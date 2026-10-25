"""
kaizen_topic_demo.py - Topic-focused assessment with an independent simulated student
=====================================================================================

Story told by this script (pick the topic with --target, default c07 "Loops & Iteration"):

  PHASE 1  Assess the target topic         -> the student struggles
  PHASE 2  Root-cause search               -> probe the most suspicious prerequisites
  PHASE 3  Personalised learning path      -> what to learn first, in order
  + graph BEFORE / AFTER, a filmstrip of the graph over the session, mastery trajectories,
    and an evaluation of the agent's diagnosis against the student's hidden ground truth.

Who is who (integrity of the experiment)
----------------------------------------
  question generator + tutor : LLM #1   (writes questions, never sees the student)
  simulated student          : LLM #2   (separate client/model; sees ONLY the question text and a
                                         persona - never the reference answer, rubric, tags or ids)
  grader                     : rules + optional judge LLM #3 (sees question + reference + answer)
  mastery estimator          : plain code (BKT) - no LLM
  graph logic                : plain code
The student's hidden true skill levels are used ONLY after the run, to score the agent's diagnosis.

Usage
-----
  python kaizen_topic_demo.py                          # fully offline (scripted student persona)
  python kaizen_topic_demo.py --provider gemini        # Gemini generator + a different Gemini student
  python kaizen_topic_demo.py --target c09 --provider gemini --student-model gemini-2.5-flash-lite
In Colab:  from kaizen_topic_demo import run_topic_demo   (see the docstring of run_topic_demo)
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import kaizen_viz as viz
from kaizen_assessment_agent import (
    AssessmentAgent, DynamicGraph, LLMClient, OfflineLLM, Question, StudentState, _SAME, make_llm,
)

# =============================================================================
# 1. STUDENT PROFILE (hidden ground truth) + PERSONA PROMPT
# =============================================================================

@dataclass
class StudentProfile:
    name: str
    know_well: List[str]                       # concept ids the student really knows
    shaky: Dict[str, str]                      # concept id -> description of partial understanding
    weak: Dict[str, str]                       # concept id -> the wrong beliefs the student holds

    def true_skill(self) -> Dict[str, float]:
        """Hidden ground truth. Used ONLY to evaluate the agent after the run."""
        out = {c: 0.90 for c in self.know_well}
        out.update({c: 0.50 for c in self.shaky})
        out.update({c: 0.10 for c in self.weak})
        return out


DEFAULT_PROFILE = StudentProfile(
    name="Riya",
    know_well=["c01", "c02", "c03", "c04"],
    shaky={"c06": "You understand if/else, but you think every elif branch whose condition is true will run, "
                  "and you sometimes write = instead of == in conditions."},
    weak={"c07": "You think range(n) produces 1..n (including n). You mix up break and continue. "
                 "You forget to update the counter in while loops. You only vaguely know what a for loop is for."},
)

STUDENT_SYSTEM = """You are role-playing a first-year student taking a Python quiz. Stay in character.

Your knowledge profile (this is ALL you know):
{profile}

Rules:
- Answer ONLY from this profile. Never use knowledge beyond it, even if you could work out the real answer.
- Topics you know well: answer correctly and briefly.
- Shaky topics: answer partly right, hesitate, or mix things up as described.
- Weak topics: answer confidently from your wrong beliefs, or say you are not sure.
- Topics not mentioned: you have not studied them yet - say you are not sure.
- Reply with the answer only: at most 2 sentences, or the exact output for code questions.
- Never mention this profile or that you are simulated."""


def persona_text(p: StudentProfile, names: Dict[str, str]) -> str:
    lines = [f"Your name is {p.name}.",
             "You KNOW WELL: " + ", ".join(f"{c} {names[c]}" for c in p.know_well) + "."]
    for c, d in p.shaky.items():
        lines.append(f"You are SHAKY on {c} {names[c]}: {d}")
    for c, d in p.weak.items():
        lines.append(f"You are WEAK on {c} {names[c]}: {d}")
    return "\n".join(lines)


# =============================================================================
# 2. THE SIMULATED STUDENT (blind, separate LLM)
# =============================================================================

class SimulatedStudent:
    """Wraps an LLM client with a persona. BLIND by construction: `answer()` forwards only the
    question text - no reference answer, rubric, misconception tag, concept id or difficulty."""

    def __init__(self, profile: StudentProfile, llm: LLMClient, concept_names: Dict[str, str]):
        self.profile, self.llm = profile, llm
        self.system = STUDENT_SYSTEM.format(profile=persona_text(profile, concept_names))

    def answer(self, q: Question) -> str:
        prompt = f"Question:\n{q.prompt}\n\nYour answer:"
        try:
            text = self.llm.complete(self.system, prompt, max_tokens=200).strip()
        except Exception:
            text = ""
        return (text or "I'm not sure.")[:400]


def assert_independent(**clients: LLMClient):
    """Integrity guard: the roles must be distinct client objects (ideally distinct models)."""
    names = list(clients)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = clients[names[i]], clients[names[j]]
            if a is b:
                raise ValueError(f"'{names[i]}' and '{names[j]}' share one client - roles must be independent")
            ma, mb = getattr(a, "model", None), getattr(b, "model", None)
            if ma and ma == mb:
                print(f"  [note] {names[i]} and {names[j]} use the same model ({ma}); "
                      f"a different model gives a stronger integrity story.")


# ---- offline stand-in for the student LLM (hand-written persona answers) ------
# (substring of the question, answer). Authored independently of the reference answers.
_SCRIPT: List[Tuple[str, str]] = [
    # c01-c03: knows well
    ("indentation", "Indentation shows which lines belong to a block, like the body of an if, loop or function."),
    ("single-line comment", "You put a # at the start of the line."),
    ("case-sensitive", "Yes, name and Name are different identifiers."),
    ("usual extension", ".py"),
    ("What does x = 10 do", "It assigns the value 10 to the variable x."),
    ("another type", "Yes, Python is dynamically typed so that is allowed."),
    ("valid variable name", "value_2"),
    ("followed by x = 8", "x refers to the value 8 now."),
    ("four basic", "int, float, str and bool."),
    ("type(5)", "It returns the int type."),
    ("int('25')", "The integer 25."),
    ("bool(0)", "False"),
    # c04: knows well
    ("7 // 2", "3"),
    ("difference between == and =", "== compares values, while = assigns a value."),
    ("operator and", "It is True only when both Boolean operands are true."),
    ("3 in [1, 2, 3]", "True"),
    ("10 % 3 + 2 ** 3", "9"),
    ("7 / 2", "3.5"),
    # c06: shaky
    ("purpose of if", "To execute a block when a condition is true."),
    ("When is elif used", "It's used to check another condition."),
    ("else block represent", "The block that runs otherwise, when the ifs are false."),
    ("5 > 2", "A Boolean value, True."),
    ("elif x > 4", "A and B"),
    ("if x = 5", "= is assignment, you need the comparison operator =="),
    # c07: weak
    ("for loop commonly used", "To loop over the items in a list or a range of numbers."),
    ("range(5) generate", "The numbers 1 to 5."),
    ("break do", "It skips to the next iteration."),
    ("continue do", "It stops the loop completely."),
    ("for i in range(3): print(i)", "1 2 3"),
    ("range(1, 4)", "10"),
    ("while i < 3", "You should add a break."),
]


class ScriptedStudentLLM(LLMClient):
    """Deterministic persona so the demo runs offline. Gets the same prompt a real student LLM would."""
    def complete(self, system, prompt, max_tokens=200):
        q = prompt.split("Question:\n", 1)[-1].split("\n\nYour answer:")[0]
        for key, ans in _SCRIPT:
            if key in q:
                return ans
        return "I'm not sure."


# =============================================================================
# 3. THE TOPIC ASSESSMENT SESSION
# =============================================================================

@dataclass
class Frame:
    label: str
    phase: str
    snapshot: dict


@dataclass
class SessionResult:
    target: str
    state: StudentState
    frames: List[Frame]
    events: List[dict]
    phases: List[Tuple[str, int, int]]
    learning_path: List[dict]
    blocked: List[str]
    evaluation: dict
    files: Dict[str, str] = field(default_factory=dict)


def _hr(title: str = "", ch: str = "═", width: int = 86):
    print(f"\n{ch * 3} {title} " + ch * max(3, width - len(title) - 5) if title else ch * width)


def _block(label: str, text: str) -> str:
    """'│  Label     : first line' then continuation lines aligned underneath (keeps code readable)."""
    lines = str(text).splitlines() or [""]
    head = f"│  {label:<10}: "
    return "\n".join([head + lines[0]] + ["│  " + " " * 10 + "  " + ln for ln in lines[1:]])


class TopicAssessment:
    def __init__(self, agent: AssessmentAgent, student: SimulatedStudent, target: str, *,
                 student_id: str = "student", min_target: int = 3, max_target: int = 4,
                 probe_min: int = 2, probe_max: int = 3, max_probed: int = 3, verbose: bool = True):
        self.agent, self.student, self.target = agent, student, target
        self.st = StudentState(student_id)
        self.k = dict(min_target=min_target, max_target=max_target, probe_min=probe_min,
                      probe_max=probe_max, max_probed=max_probed)
        self.verbose = verbose
        self.frames: List[Frame] = []
        self.events: List[dict] = []
        self.phases: List[Tuple[str, int, int]] = []

    # ---- helpers ------------------------------------------------------------
    @property
    def dg(self) -> DynamicGraph:
        return DynamicGraph(self.agent.graph, self.st)

    def _name(self, c: str) -> str:
        return self.agent.graph.name(c)

    def _snap(self, label: str, phase: str):
        self.frames.append(Frame(label, phase, self.dg.snapshot()))

    def _ask(self, cid: str, phase: str) -> Optional[dict]:
        q = self.agent.pick_question(self.st, cid)
        if q is None:
            print(f"   (no unasked questions left for {cid})")
            return None
        answer = self.student.answer(q)                     # blind call to the student LLM
        ev = self.agent.submit(self.st, q, answer)          # grade -> BKT -> propagate -> infer
        ev.update(phase=phase, answer=answer, prompt=q.prompt, reference=q.answer,
                  bank=q.source, concept_name=self._name(cid))
        self.events.append(ev)
        if self.verbose:
            self._print_event(ev)
        self._snap(f"after Q{ev['step']} ({cid})", phase)
        return ev

    def _print_event(self, ev: dict):
        cs = self.st.cs[ev["concept_id"]]
        print(f"\n┌─ Q{ev['step']} · {ev['phase'].upper()} · {ev['concept_id']} {ev['concept_name']}"
              f"   [{ev['type']} · difficulty {ev['difficulty']} · {ev['bank']}]")
        print(_block("Question", ev["prompt"]))
        print(_block("Student", f"\"{ev['answer']}\""))
        print(_block("Reference", ev["reference"]) + "   (hidden from the student)")
        mis = f"   misconception: {ev['misconception']}" if ev["misconception"] and ev["score"] < 0.5 else ""
        print(f"│  Grade     : {ev['score']:.2f}  via {ev['grader']}{mis}")
        print(f"└─ Mastery {ev['concept_id']}: {ev['p_before']:.2f} → {ev['p_after']:.2f}   "
              f"status={ev['status']}   confidence={cs.confidence:.2f}")

    # ---- the three phases -----------------------------------------------------
    def run(self) -> None:
        t, k, ag = self.target, self.k, self.agent
        self._snap("Start: nothing known", "milestone")

        # PHASE 1 -----------------------------------------------------------
        _hr(f"PHASE 1 · Assess the target topic: {t} {self._name(t)}")
        print("The agent asks a mix of recall, code-trace and debug questions. Stop rule: at least "
              f"{k['min_target']} answers, then stop as soon as the concept is clearly mastered or a gap.")
        first = self.st.step + 1
        while True:
            ev = self._ask(t, "target")
            if ev is None:
                break
            n, status = self.st.cs[t].attempts, self.dg.status(t)
            if (n >= k["min_target"] and status in ("mastered", "gap")) or n >= k["max_target"]:
                break
        self.phases.append(("1 · Target topic", first, self.st.step))
        self._snap(f"After target topic: {t} is {self.dg.status(t)}", "milestone")
        status = self.dg.status(t)
        print(f"\n→ Result: {t} is '{status}' (p = {self.st.cs[t].p:.2f}).")

        # PHASE 2 -----------------------------------------------------------
        probed: List[str] = []
        if status != "mastered":
            _hr(f"PHASE 2 · Root-cause search: is the problem in a prerequisite of {t}?")
            first = self.st.step + 1
            while len(probed) < k["max_probed"]:
                sus = [s for s in self.dg.suspects(t) if s[0] not in probed]
                if not sus:
                    break
                ranked = ", ".join(f"{c} ({sc:.2f})" for c, sc in sus[:4])
                cid = sus[0][0]
                print(f"\nSuspect ranking (path strength × (1-p) × (1-½·confidence)): {ranked}")
                print(f"→ Probing {cid} {self._name(cid)}")
                before = {c for c in self.agent.graph.concepts if self.dg.status(c) == "inferred"}
                for i in range(k["probe_max"]):
                    ev = self._ask(cid, "probe")
                    if ev is None:
                        break
                    if i + 1 >= k["probe_min"] and self.dg.status(cid) in ("mastered", "gap"):
                        break
                probed.append(cid)
                st_c = self.dg.status(cid)
                inferred = [c for c in self.agent.graph.concepts
                            if self.dg.status(c) == "inferred" and c not in before]
                msg = f"→ {cid} is '{st_c}' (p = {self.st.cs[cid].p:.2f})."
                if st_c == "mastered" and inferred:
                    msg += f"  Mastery implies its prerequisites, so {', '.join(sorted(inferred))} are now inferred (not tested)."
                print(msg)
            if probed:
                self.phases.append(("2 · Root-cause search", first, self.st.step))
                self._snap("After root-cause search of prerequisites", "milestone")

        # PHASE 3 -----------------------------------------------------------
        _hr("PHASE 3 · Personalised learning path")
        self.learning_path = self._build_path()
        self.blocked = sorted(c for c in self.agent.graph.descendants(t) if not self.dg.is_available(c))
        self._print_path()
        self._snap("Final personalised graph + learning order", "milestone")

    # ---- learning path --------------------------------------------------------
    def _build_path(self) -> List[dict]:
        g, dg, t = self.agent.graph, self.dg, self.target
        need = []
        for c in g.order:
            if c != t and c not in g.ancestors(t):
                continue
            stc, cs = dg.status(c), self.st.cs[c]
            if stc in ("mastered", "inferred"):
                continue
            if c != t and stc == "learning" and cs.p >= 0.70:
                continue                                         # borderline-OK prerequisite
            tags = sorted({e["misconception"] for e in self.events
                           if e["concept_id"] == c and e["score"] < 0.5 and e["misconception"]})
            reason = {"gap": "confirmed gap", "learning": "shaky / not yet reliable",
                      "unseen": "not tested"}.get(stc, stc)
            need.append(dict(concept_id=c, name=g.name(c), status=stc, p=round(cs.p, 3),
                             attempts=cs.attempts, reason=reason, misconceptions=tags))
        return need

    def _print_path(self):
        g, dg, t = self.agent.graph, self.dg, self.target
        verified = [c for c in g.order if c in g.ancestors(t) and dg.status(c) == "mastered"]
        inferred = [c for c in g.order if c in g.ancestors(t) and dg.status(c) == "inferred"]
        print(f"\nTarget {t} {self._name(t)}: status = {dg.status(t)}, p = {self.st.cs[t].p:.2f}")
        if verified:
            print("Verified strong prerequisites : " + ", ".join(f"{c} (p={self.st.cs[c].p:.2f})" for c in verified))
        if inferred:
            print("Inferred known (not tested)   : " + ", ".join(inferred))
        print("\nLearn in this order:")
        for i, s in enumerate(self.learning_path, 1):
            mis = f"  | misconceptions: {', '.join(s['misconceptions'])}" if s["misconceptions"] else ""
            print(f"  {i}. {s['concept_id']} {s['name']:<34} {s['reason']} (p={s['p']:.2f}, {s['attempts']} answers){mis}")
        if self.blocked:
            print("\nStay locked until the path is done: " + ", ".join(f"{c} {self._name(c)}" for c in self.blocked))

    # ---- evaluation against hidden truth ----------------------------------------
    def evaluate(self, profile: StudentProfile) -> dict:
        g, t = self.agent.graph, self.target
        truth = profile.true_skill()
        relevant = sorted(({t} | g.ancestors(t)) & set(truth), key=g.order.index)
        truth_weak = {c for c in relevant if truth[c] < 0.60}
        flagged = {s["concept_id"] for s in self.learning_path} & set(relevant)
        tp = len(truth_weak & flagged)
        prec = tp / len(flagged) if flagged else 1.0
        rec = tp / len(truth_weak) if truth_weak else 1.0
        rows = [dict(concept_id=c, true_skill=truth[c], estimated_p=round(self.st.cs[c].p, 2),
                     status=self.dg.status(c), tested=self.st.cs[c].attempts > 0,
                     verdict=("weak" if c in truth_weak else "ok") + " / " + ("flagged" if c in flagged else "not flagged"))
                for c in relevant]
        return dict(rows=rows, precision=prec, recall=rec, truth_weak=sorted(truth_weak), flagged=sorted(flagged))

    def print_evaluation(self, ev: dict):
        _hr("EVALUATION · agent's estimate vs the student's hidden true skill")
        print(f"{'concept':<30}{'true skill':>10}{'agent p':>9}   {'status':<10}{'how':<12}verdict")
        for r in ev["rows"]:
            c = r["concept_id"]
            label = textwrap.shorten(f"{c} {self._name(c)}", 29)
            print(f"{label:<30}{r['true_skill']:>10.2f}{r['estimated_p']:>9.2f}   {r['status']:<10}"
                  f"{'tested' if r['tested'] else ('inferred' if r['status'] == 'inferred' else 'untested'):<12}{r['verdict']}")
        print(f"\nTruly weak: {ev['truth_weak']}   Agent flagged: {ev['flagged']}   "
              f"precision={ev['precision']:.2f}  recall={ev['recall']:.2f}")


# =============================================================================
# 4. ENTRY POINT
# =============================================================================

def run_topic_demo(target: str = "c07", profile: StudentProfile = DEFAULT_PROFILE, *,
                   gen_llm: Optional[LLMClient] = None, student_llm: Optional[LLMClient] = None,
                   judge_llm=None, out_dir: str = "kaizen_out", show: bool = False,
                   bank_cache: Optional[str] = None, interactive_html: bool = True) -> SessionResult:
    """
    gen_llm      question generator + tutor        (default: offline stub)
    student_llm  the simulated student             (default: scripted offline persona)
    judge_llm    grader for non-exact answers      (default None = rule-based; pass a client for an LLM judge)
    bank_cache   json path: loaded if it exists, written after the run -> repeat runs need no generator calls

    Colab example:
        gen     = GeminiLLM("gemini-2.5-flash")
        student = GeminiLLM("gemini-2.5-flash-lite", temperature=0.8, min_interval=5)
        run_topic_demo("c07", gen_llm=gen, student_llm=student, show=True, bank_cache="bank.json")
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_llm = gen_llm or OfflineLLM()
    offline_student = student_llm is None
    student_llm = student_llm or ScriptedStudentLLM()
    assert_independent(generator=gen_llm, student=student_llm)

    agent = AssessmentAgent(gen_llm, grader_llm=judge_llm)
    agent.lazy_generation = True
    if bank_cache and os.path.exists(bank_cache):
        print(f"Loaded {agent.load_generated(bank_cache)} cached generated questions from {bank_cache}")
    names = {c: agent.graph.name(c) for c in agent.graph.concepts}
    student = SimulatedStudent(profile, student_llm, names)

    _hr("KAIZEN · topic assessment demo")
    print(f"Target topic : {target} {names[target]}")
    print(f"Student      : {profile.name} (hidden profile; the agent does NOT see it)")
    print(f"Generator LLM: {type(gen_llm).__name__} {getattr(gen_llm, 'model', '')}")
    print(f"Student LLM  : {'ScriptedStudentLLM (offline persona)' if offline_student else type(student_llm).__name__ + ' ' + str(getattr(student_llm, 'model', ''))}")
    print(f"Grader       : {'rule-based' if judge_llm is None else type(judge_llm).__name__}"
          f" (exact match first, then {'LLM judge' if judge_llm else 'keyword overlap'})")
    print("Mastery      : Bayesian Knowledge Tracing (no LLM)   Graph logic: plain code")

    sess = TopicAssessment(agent, student, target, student_id=profile.name.lower())
    sess.run()
    evaluation = sess.evaluate(profile)
    sess.print_evaluation(evaluation)

    # ---- artefacts ---------------------------------------------------------------
    path_ids = [s["concept_id"] for s in sess.learning_path]
    start, end = sess.frames[0].snapshot, sess.frames[-1].snapshot
    files = {}
    viz.plot_before_after(start, end, target=target, path=path_ids,
                          filename=(files.setdefault("before_after", f"{out_dir}/graph_before_after.png")),
                          suptitle=f"{profile.name}: assessment of {target} {names[target]}")
    keyframes = [(f.label, f.snapshot) for f in sess.frames
                 if f.phase == "milestone"]
    viz.plot_frames(keyframes, target=target, path=path_ids,
                    filename=files.setdefault("filmstrip", f"{out_dir}/graph_trajectory_frames.png"))
    viz.plot_trajectory(sess.events, sess.phases, target=target,
                        filename=files.setdefault("trajectory", f"{out_dir}/mastery_trajectory.png"),
                        names={c: names[c] for c in names})
    with open(files.setdefault("log", f"{out_dir}/session_log.json"), "w") as f:
        json.dump(dict(target=target, events=sess.events, learning_path=sess.learning_path,
                       blocked=sess.blocked, evaluation=evaluation, final_graph=end,
                       phases=sess.phases), f, indent=2)
    with open(files.setdefault("hidden_profile", f"{out_dir}/hidden_student_profile.json"), "w") as f:
        json.dump(dict(name=profile.name, true_skill=profile.true_skill(),
                       know_well=profile.know_well, shaky=profile.shaky, weak=profile.weak), f, indent=2)
    if interactive_html:
        html = viz.export_interactive(end, f"{out_dir}/graph_end_interactive.html", target=target, path=path_ids)
        if html:
            files["interactive"] = html
    if bank_cache:
        print(f"\nSaved {agent.save_generated(bank_cache)} generated questions to {bank_cache}")

    _hr("OUTPUT FILES")
    for k, v in files.items():
        print(f"  {k:<15} {v}")
    if show:
        plt.show()
    else:
        plt.close("all")
    res = SessionResult(target, sess.st, sess.frames, sess.events, sess.phases,
                        sess.learning_path, sess.blocked, evaluation, files)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="c07")
    ap.add_argument("--provider", default="offline", choices=["offline", "gemini", "anthropic"])
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--student-model", default=None, help="use a DIFFERENT model than the generator")
    ap.add_argument("--out", default="kaizen_out")
    ap.add_argument("--bank-cache", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.provider == "offline":
        gen = stu = None
    else:
        gen = make_llm(a.provider, a.gen_model)
        stu_kw = dict(temperature=0.8, min_interval=5.0) if a.provider == "gemini" else {}
        stu = make_llm(a.provider, a.student_model or (
            "gemini-2.5-flash-lite" if a.provider == "gemini" else None), **stu_kw)
    run_topic_demo(a.target, gen_llm=gen, student_llm=stu, out_dir=a.out, show=a.show, bank_cache=a.bank_cache)
