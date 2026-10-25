"""
KAIZEN - Track B | Assessment Agent + Dynamic Concept Graph
===========================================================

One-file reference implementation. Run it:

    python kaizen_assessment_agent.py              # scripted demo, fully offline
    python kaizen_assessment_agent.py --interactive # answer questions in the terminal
    python kaizen_assessment_agent.py --live        # use Claude API (needs ANTHROPIC_API_KEY)

Architecture (hybrid: LLM for language, plain code for state)
-------------------------------------------------------------
  GroundTruth graph (immutable, Track A)  <-- never edited per student
        |
  ConceptGraph ------ DynamicGraph (per-student overlay: p_mastery, status)
        |                    ^
  QuestionGenerator (LLM)    |  updates
  AnswerGrader  (rules+LLM) -> MasteryEstimator (BKT, no LLM)
        |
  AssessmentAgent  -- plans: assess / probe / teach / recheck / done
        |
  TutorAgent (thin)  -- explains a gap, hands control back for re-check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# =============================================================================
# 1. GROUND TRUTH DATA (copied from Track A syllabus)
# =============================================================================

DOMAIN = "Python Fundamentals"

# concept_id: (name, difficulty 1-5, description)
CONCEPTS: Dict[str, Tuple[str, int, str]] = {
    "c01": ("Python Basics & Program Structure", 1, "What Python is, how a program runs, indentation, comments, basic syntax."),
    "c02": ("Variables & Assignment", 1, "Named variables, assignment, reassignment, naming rules, basic expressions."),
    "c03": ("Data Types & Type Conversion", 2, "int, float, str, bool, None, type(), explicit conversion."),
    "c04": ("Operators & Expressions", 2, "Arithmetic, comparison, logical, assignment, membership, identity operators."),
    "c05": ("Input & Output", 2, "input(), converting input, print()."),
    "c06": ("Conditional Statements", 2, "if / elif / else decisions based on Boolean expressions."),
    "c07": ("Loops & Iteration", 3, "for and while loops, range(), loop conditions, break/continue."),
    "c08": ("Strings", 2, "Indexing, slicing, searching, formatting and string methods."),
    "c09": ("Lists, Tuples & Sets", 3, "Collections, indexing/slicing, list mutation, tuple immutability, set operations."),
    "c10": ("Dictionaries", 3, "Key-value pairs, access/update, dictionary methods, iteration."),
    "c11": ("Functions", 3, "def, parameters, return values, local scope, default arguments."),
    "c12": ("Modules & Imports", 3, "Organising code into modules, import / from-import."),
    "c13": ("Exception Handling", 4, "try, except, else, finally; handling errors deliberately."),
    "c14": ("File Handling", 3, "open(), with, common file modes."),
    "c15": ("Comprehensions", 4, "List, set and dict comprehensions."),
    "c16": ("Recursion", 5, "Recursive calls, base case, recursive case."),
}

# (from_id, to_id, weight) - relation_type is always "prerequisite"
EDGES: List[Tuple[str, str, int]] = [
    ("c01", "c02", 9), ("c02", "c03", 9), ("c03", "c04", 8), ("c03", "c05", 7),
    ("c04", "c05", 6), ("c04", "c06", 9), ("c04", "c07", 7), ("c06", "c07", 7),
    ("c03", "c08", 8), ("c04", "c08", 6), ("c03", "c09", 8), ("c04", "c09", 6),
    ("c07", "c09", 7), ("c03", "c10", 8), ("c07", "c10", 7), ("c05", "c11", 6),
    ("c06", "c11", 7), ("c07", "c11", 7), ("c11", "c12", 8), ("c06", "c13", 6),
    ("c11", "c13", 8), ("c05", "c14", 6), ("c11", "c14", 7), ("c07", "c15", 9),
    ("c09", "c15", 9), ("c10", "c15", 8), ("c11", "c16", 9), ("c06", "c16", 7),
]

# Ground-truth question bank: concept_id -> [(prompt, answer), ...]
GROUND_TRUTH_QA: Dict[str, List[Tuple[str, str]]] = {
    "c01": [
        ("What does indentation represent in Python?", "It defines code blocks such as the body of an if statement, loop, or function."),
        ("How do you write a single-line comment in Python?", "Start the comment with #."),
        ("Is Python case-sensitive?", "Yes. For example, name and Name are different identifiers."),
        ("What is the usual extension for a Python source file?", ".py"),
    ],
    "c02": [
        ("What does x = 10 do?", "It assigns the value 10 to the variable x."),
        ("Can a Python variable be reassigned to a value of another type?", "Yes. Python is dynamically typed."),
        ("Which is a valid variable name: 2value or value_2?", "value_2"),
        ("What happens after x = 5 followed by x = 8?", "x refers to the value 8."),
    ],
    "c03": [
        ("Name four basic Python data types.", "int, float, str, and bool."),
        ("What does type(5) return?", "The int type."),
        ("What does int('25') produce?", "The integer 25."),
        ("What does bool(0) produce?", "False"),
    ],
    "c04": [
        ("What is the result of 7 // 2?", "3"),
        ("What is the difference between == and =?", "== compares values; = assigns a value."),
        ("What does the operator and do?", "It is True only when both Boolean operands are true."),
        ("What does 3 in [1, 2, 3] return?", "True"),
    ],
    "c05": [
        ("What type does input() normally return?", "str"),
        ("How would you read an integer from the user?", "int(input())"),
        ("What is print('Hello') used for?", "Displaying output to the console."),
        ("Why might input() need type conversion before arithmetic?", "Because input() returns text, while arithmetic may require a numeric type."),
    ],
    "c06": [
        ("What is the purpose of if?", "To execute a block when a condition is true."),
        ("When is elif used?", "To test another condition if previous conditions were false."),
        ("What does an else block represent?", "The fallback block executed when preceding conditions are false."),
        ("What type of result is produced by a comparison such as 5 > 2?", "A Boolean value: True."),
    ],
    "c07": [
        ("What is a for loop commonly used for?", "Iterating over items in an iterable or a range of values."),
        ("What does range(5) generate?", "The sequence of integers 0 through 4."),
        ("What does break do inside a loop?", "Immediately exits the loop."),
        ("What does continue do?", "Skips the rest of the current iteration and moves to the next iteration."),
    ],
    "c08": [
        ("What is the index of the first character in a Python string?", "0"),
        ("What does 'Python'[0] return?", "'P'"),
        ("Are Python strings mutable?", "No, strings are immutable."),
        ("What does 'hello'.upper() return?", "'HELLO'"),
    ],
    "c09": [
        ("Which collection is mutable: list or tuple?", "A list is mutable; a tuple is immutable."),
        ("How do you access the first item of a list named nums?", "nums[0]"),
        ("What is one purpose of a set?", "To store unique elements and support set operations."),
        ("What does append() do to a list?", "It adds one item to the end of the list."),
    ],
    "c10": [
        ("What is stored in a dictionary?", "Key-value pairs."),
        ("How do you access the value for key 'name' in d?", "d['name']"),
        ("Can dictionary keys be duplicated?", "No. Each key is unique within a dictionary."),
        ("What does d.items() provide?", "A view of the dictionary's key-value pairs."),
    ],
    "c11": [
        ("How do you define a function in Python?", "Use the def keyword followed by the function name and parameter list."),
        ("What is the purpose of return?", "It sends a value back to the caller and ends the function's current execution."),
        ("What is a parameter?", "A named input defined by a function."),
        ("What is local scope?", "The region in which a local variable defined inside a function is accessible."),
    ],
    "c12": [
        ("What does import math do?", "It imports the math module so its functions/constants can be used."),
        ("What is a module?", "A Python file containing reusable code such as functions, classes, or variables."),
        ("What is the purpose of from math import sqrt?", "It imports sqrt directly from the math module."),
        ("Why use modules?", "To organize code and reuse functionality."),
    ],
    "c13": [
        ("Which block is used to catch an exception?", "except"),
        ("What is the purpose of finally?", "It runs after the try/except process whether an exception occurred or not."),
        ("Why should exceptions be handled specifically rather than broadly when possible?", "Specific handling makes the program's behavior clearer and avoids hiding unrelated errors."),
        ("What kind of error can try/except handle?", "Runtime exceptions, such as ValueError or FileNotFoundError."),
    ],
    "c14": [
        ("Which function is commonly used to open a file?", "open()"),
        ("What does the with statement help with when working with files?", "It ensures the file resource is properly managed and closed."),
        ("What does mode 'r' mean?", "Open for reading."),
        ("What does mode 'w' mean?", "Open for writing, replacing/truncating existing content."),
    ],
    "c15": [
        ("What is a list comprehension?", "A compact syntax for creating a list from an iterable, optionally with a condition."),
        ("What is the conceptual equivalent of [x*x for x in nums]?", "A loop that creates a new list by appending x*x for each x in nums."),
        ("Can a comprehension contain a condition?", "Yes, for example [x for x in nums if x > 0]."),
        ("Why should comprehensions not be used when they make the logic harder to read?", "Readability and maintainability are more important than unnecessary compactness."),
    ],
    "c16": [
        ("What is the base case in recursion?", "The condition that stops further recursive calls."),
        ("What happens if a recursive function has no reachable base case?", "It can keep calling itself until Python raises a RecursionError."),
        ("What is a recursive case?", "The part of the function that solves a smaller subproblem by calling itself."),
        ("Why is recursion taught after functions?", "Recursion is implemented through function calls and requires understanding parameters, return values, and control flow."),
    ],
}

QUESTION_TYPES = ("recall", "code_trace", "debug", "apply")


# =============================================================================
# 2. DATA MODELS
# =============================================================================

@dataclass
class Question:
    question_id: str
    concept_id: str
    type: str                 # recall | code_trace | debug | apply
    difficulty: int           # 1-5
    prompt: str
    answer: str
    rubric: str = ""
    misconception_tag: Optional[str] = None
    source: str = "ground_truth"      # ground_truth | generated
    verified: bool = True             # generated questions start unverified


@dataclass
class ConceptState:
    p: float = 0.30                   # P(mastery)
    attempts: int = 0
    recent: List[float] = field(default_factory=list)   # scores since last teaching
    needs_recheck: bool = False
    penalty_applied: bool = False
    last_attempt_idx: int = -1

    @property
    def confidence(self) -> float:
        """0..1, grows with the number of real attempts (one lucky answer != mastery)."""
        return self.attempts / (self.attempts + 2.0)


@dataclass
class StudentState:
    student_id: str
    cs: Dict[str, ConceptState] = field(default_factory=dict)
    asked: set = field(default_factory=set)
    history: List[dict] = field(default_factory=list)
    step: int = 0

    def __post_init__(self):
        for cid in CONCEPTS:
            self.cs.setdefault(cid, ConceptState())

    def to_json(self) -> str:
        d = asdict(self)
        d["asked"] = sorted(self.asked)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "StudentState":
        d = json.loads(s)
        st = cls(student_id=d["student_id"])
        st.cs = {k: ConceptState(**v) for k, v in d["cs"].items()}
        st.asked = set(d["asked"])
        st.history, st.step = d["history"], d["step"]
        return st


@dataclass
class GradeResult:
    score: float                      # 0..1 (partial credit)
    feedback: str = ""
    misconception: Optional[str] = None
    method: str = "rules"


@dataclass
class Plan:
    action: str                       # assess | probe | teach | recheck | done
    concept_id: Optional[str] = None
    reason: str = ""
    chain: List[str] = field(default_factory=list)


# =============================================================================
# 3. CONCEPT GRAPH (immutable ground truth)
# =============================================================================

class ConceptGraph:
    def __init__(self, concepts=CONCEPTS, edges=EDGES):
        self.concepts = concepts
        self.edges = list(edges)
        self._pre: Dict[str, List[Tuple[str, int]]] = {c: [] for c in concepts}
        self._post: Dict[str, List[Tuple[str, int]]] = {c: [] for c in concepts}
        for u, v, w in self.edges:
            self._pre[v].append((u, w))
            self._post[u].append((v, w))
        self.validate()
        self.order = self.topological_order()

    def prereqs(self, c): return self._pre[c]
    def dependents(self, c): return self._post[c]
    def name(self, c): return self.concepts[c][0]
    def difficulty(self, c): return self.concepts[c][1]

    def ancestors(self, c: str) -> set:
        """All transitive prerequisites of c."""
        seen, stack = set(), [c]
        while stack:
            for u, _ in self._pre[stack.pop()]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        return seen

    def descendants(self, c: str) -> set:
        """All concepts that (transitively) depend on c."""
        seen, stack = set(), [c]
        while stack:
            for v, _ in self._post[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def validate(self):
        for u, v, w in self.edges:
            assert u in self.concepts and v in self.concepts, f"unknown id in edge {u}->{v}"
            assert 1 <= w <= 10, f"weight out of range on {u}->{v}"
        assert len(self.topological_order()) == len(self.concepts), "graph has a cycle"

    def topological_order(self) -> List[str]:
        indeg = {c: len(self._pre[c]) for c in self.concepts}
        ready = sorted(c for c, d in indeg.items() if d == 0)
        out = []
        while ready:
            c = ready.pop(0)
            out.append(c)
            for v, _ in self._post[c]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
            ready.sort()
        return out

    def export_ground_truth(self) -> dict:
        return {
            "nodes": [{"concept_id": c, "name": n, "description": d, "difficulty": df,
                       "domain_tag": DOMAIN} for c, (n, df, d) in self.concepts.items()],
            "edges": [{"from_id": u, "to_id": v, "relation_type": "prerequisite", "weight": w}
                      for u, v, w in self.edges],
        }


# =============================================================================
# 4. LLM CLIENTS (swap freely; everything works with OfflineLLM)
# =============================================================================

class LLMClient:
    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        raise NotImplementedError


class AnthropicLLM(LLMClient):
    def __init__(self, model: Optional[str] = None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("KAIZEN_MODEL", "claude-sonnet-4-6")

    def complete(self, system, prompt, max_tokens=1500):
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")


class GeminiLLM(LLMClient):
    """Google Gemini (`pip install google-genai`). Throttled for free-tier limits.

    Free-tier quotas are per model, so giving the *student* a different model than the
    *question generator* also gives each its own rate-limit budget.
    """
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None,
                 min_interval: float = 13.0, retries: int = 4, temperature: float = 0.4):
        from google import genai
        from google.genai import types
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key is None:
            try:                                   # Colab "Secrets" panel
                from google.colab import userdata
                key = userdata.get("GEMINI_API_KEY")
            except Exception:
                pass
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model, self.min_interval = model, min_interval
        self.retries, self.temperature = retries, temperature
        self._last, self.calls = 0.0, 0

    def complete(self, system, prompt, max_tokens=1500):
        cfg = dict(system_instruction=system, max_output_tokens=max_tokens * 2,
                   temperature=self.temperature)
        if "flash" in self.model:                  # disable "thinking" for speed (not allowed on pro)
            cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        for _ in range(self.retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                self.calls += 1
                r = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._types.GenerateContentConfig(**cfg))
                return r.text or ""
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retry in ([\d.]+)s", msg)
                    time.sleep(float(m.group(1)) + 2 if m else 40)
                    continue
                raise
        return ""


def make_llm(provider: str = "offline", model: Optional[str] = None, **kw) -> LLMClient:
    """provider: offline | gemini | anthropic"""
    if provider == "offline":
        return OfflineLLM()
    if provider == "gemini":
        return GeminiLLM(model or "gemini-2.5-flash", **kw)
    if provider == "anthropic":
        return AnthropicLLM(model)
    raise ValueError(f"unknown provider {provider!r}")


# Canned generator output so the demo runs with no API key.
_OFFLINE_QUESTIONS = {
    "c04": [
        {"type": "code_trace", "difficulty": 3, "prompt": "What does print(10 % 3 + 2 ** 3) output?",
         "answer": "9", "rubric": "10 % 3 = 1, 2 ** 3 = 8, total 9.", "misconception_tag": "operator_precedence"},
        {"type": "code_trace", "difficulty": 2, "prompt": "What does print(7 / 2) output?",
         "answer": "3.5", "rubric": "/ is true division and returns a float.", "misconception_tag": "division_vs_floor_division"},
    ],
    "c06": [
        {"type": "code_trace", "difficulty": 3,
         "prompt": "x = 5\nif x > 3:\n    print('A')\nelif x > 4:\n    print('B')\nelse:\n    print('C')\nWhat is printed?",
         "answer": "A", "rubric": "Only the first true branch runs.", "misconception_tag": "elif_runs_after_true"},
        {"type": "debug", "difficulty": 3,
         "prompt": "What is wrong with: if x = 5: print('five')",
         "answer": "= is assignment; the comparison operator == should be used.",
         "rubric": "Must identify = vs ==.", "misconception_tag": "assignment_vs_comparison"},
    ],
    "c07": [
        {"type": "code_trace", "difficulty": 3, "prompt": "for i in range(3): print(i)\nWhat is printed (in order)?",
         "answer": "0 1 2", "rubric": "range(3) stops before 3.", "misconception_tag": "range_excludes_end"},
        {"type": "code_trace", "difficulty": 3,
         "prompt": "total = 0\nfor n in range(1, 4):\n    total += n\nprint(total)\nWhat is printed?",
         "answer": "6", "rubric": "1+2+3.", "misconception_tag": "range_excludes_end"},
        {"type": "debug", "difficulty": 4,
         "prompt": "i = 0\nwhile i < 3:\n    print(i)\nWhy does this loop never end and how do you fix it?",
         "answer": "i is never incremented, so the condition stays true; add i += 1 inside the loop.",
         "rubric": "Must say the loop variable is never updated.", "misconception_tag": "missing_loop_update"},
    ],
}


class OfflineLLM(LLMClient):
    """Deterministic stand-in so the demo needs no network/API key."""
    def complete(self, system, prompt, max_tokens=1500):
        if "TASK: GENERATE_QUESTIONS" in prompt:
            cid = re.search(r"CONCEPT_ID: (c\d\d)", prompt).group(1)
            return json.dumps({"questions": _OFFLINE_QUESTIONS.get(cid, [])})
        if "TASK: TUTOR" in prompt:
            return "(offline tutor) See the worked example below."
        return "{}"


# =============================================================================
# 5. QUESTION GENERATOR (LLM + schema validation)
# =============================================================================

GEN_SYSTEM = ("You are an assessment designer for an intro Python course. "
              "Reply with a single JSON object and nothing else - no prose, no markdown fences.")

GEN_TEMPLATE = """TASK: GENERATE_QUESTIONS
CONCEPT_ID: {cid}
CONCEPT: {name} (difficulty {diff}/5)
DESCRIPTION: {desc}
REQUESTED_TYPE: {qtype}
COUNT: {n}

Verified seed questions (match their style and answer length):
{seeds}

Rules:
- Test only this concept; assume only its prerequisites are known: {prereqs}
- type must be one of: recall, code_trace, debug, apply
- code_trace: show 1-6 lines of code; the answer must be the exact printed output
- debug: show broken code; the answer names the bug and the fix
- difficulty is an integer 1-5; answer must be unambiguous
- include misconception_tag: a short snake_case label for the most likely wrong belief

Return JSON: {{"questions": [{{"type": "...", "difficulty": 3, "prompt": "...",
"answer": "...", "rubric": "...", "misconception_tag": "..."}}]}}"""


class QuestionGenerator:
    REQUIRED = ("type", "difficulty", "prompt", "answer")

    def __init__(self, llm: LLMClient, graph: ConceptGraph):
        self.llm, self.graph = llm, graph

    def generate(self, concept_id: str, existing: List[Question],
                 qtype: str = "any", n: int = 3) -> List[Question]:
        name, diff, desc = self.graph.concepts[concept_id]
        seeds = "\n".join(f"- Q: {q.prompt} | A: {q.answer}"
                          for q in existing if q.concept_id == concept_id and q.verified)
        prereqs = ", ".join(u for u, _ in self.graph.prereqs(concept_id)) or "none"
        prompt = GEN_TEMPLATE.format(cid=concept_id, name=name, diff=diff, desc=desc,
                                     qtype=qtype, n=n, seeds=seeds or "(none)", prereqs=prereqs)
        raw = self.llm.complete(GEN_SYSTEM, prompt)
        return self._validate(concept_id, raw, existing)

    def _validate(self, concept_id, raw, existing) -> List[Question]:
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            items = json.loads(raw).get("questions", [])
        except (json.JSONDecodeError, AttributeError):
            return []
        seen = {q.prompt.strip().lower() for q in existing}
        out, k = [], sum(1 for q in existing if q.concept_id == concept_id and q.source == "generated")
        for it in items:
            if not all(it.get(f) not in (None, "") for f in self.REQUIRED):
                continue
            if it["type"] not in QUESTION_TYPES:
                continue
            try:
                d = int(it["difficulty"])
            except (TypeError, ValueError):
                continue
            if not 1 <= d <= 5 or it["prompt"].strip().lower() in seen:
                continue
            k += 1
            out.append(Question(
                question_id=f"{concept_id}-g{k:02d}", concept_id=concept_id, type=it["type"],
                difficulty=d, prompt=it["prompt"].strip(), answer=str(it["answer"]).strip(),
                rubric=it.get("rubric", ""), misconception_tag=it.get("misconception_tag"),
                source="generated", verified=False))
            seen.add(it["prompt"].strip().lower())
        return out


def load_ground_truth_bank(graph: ConceptGraph) -> List[Question]:
    bank = []
    for cid, qa in GROUND_TRUTH_QA.items():
        for i, (p, a) in enumerate(qa, 1):
            bank.append(Question(f"{cid}-q{i:02d}", cid, "recall", graph.difficulty(cid), p, a,
                                 rubric=a, source="ground_truth", verified=True))
    return bank


# =============================================================================
# 6. ANSWER GRADER (rules first, LLM for free text, partial credit)
# =============================================================================

_STOP = set("a an the of to is are it its in on for and or by be as at this that with from "
            "so can if when then which what does do not than into their there also".split())

GRADE_SYSTEM = "You are a strict but fair Python tutor grading a student answer. Reply with JSON only."
GRADE_TEMPLATE = """TASK: GRADE
Question: {q}
Reference answer: {a}
Rubric: {r}
Student answer: {s}

Return JSON: {{"score": <0..1, partial credit allowed>, "feedback": "<one sentence>",
"misconception": "<short snake_case label or null>"}}"""


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[`'\"]", "", s)
    s = re.sub(r"[\s,;]+", " ", s)
    return s.strip(" .")


def _content_tokens(s: str) -> set:
    return {t[:6] for t in re.findall(r"[a-z0-9_]+", s.lower()) if t not in _STOP}


class AnswerGrader:
    """Grading pipeline:  exact match  ->  LLM + rubric (if a real LLM is given)  ->  offline heuristics.
    Pass llm=None for a purely rule-based grader."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    @property
    def _real_llm(self) -> bool:
        return self.llm is not None and not isinstance(self.llm, OfflineLLM)

    def grade(self, q: Question, student_answer: str) -> GradeResult:
        s_raw = (student_answer or "").strip()
        if not s_raw or _norm(s_raw) in {"idk", "i dont know", "i don't know", "not sure", "?"}:
            return GradeResult(0.0, "No answer given.", q.misconception_tag, "rules")
        a_norm, s_norm = _norm(q.answer), _norm(s_raw)

        # 1) exact match -> full credit without spending an LLM call
        if self._exact_hit(q, a_norm, s_norm):
            return GradeResult(1.0, "Correct.", method="exact")

        # 2) LLM judge with the ground-truth answer as rubric (partial credit)
        if self._real_llm:
            try:
                raw = self.llm.complete(GRADE_SYSTEM, GRADE_TEMPLATE.format(
                    q=q.prompt, a=q.answer, r=q.rubric or q.answer, s=s_raw))
                d = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
                return GradeResult(max(0.0, min(1.0, float(d["score"]))), d.get("feedback", ""),
                                   d.get("misconception") or None, "llm")
            except Exception:
                pass                                   # fall through to heuristics
        return self._offline(q, s_raw, a_norm, s_norm)

    _FILLER = re.compile(r"^(it (prints|outputs|returns|gives|shows|produces)|the (output|result|answer) is|"
                         r"output|result|answer|prints|returns|it is|its|is)\b[:\s]*")

    @classmethod
    def _exact_hit(cls, q: Question, a_norm: str, s_norm: str) -> bool:
        """Strict for outputs and 1-2 word answers ('A' must not match 'A and B'); looser for sentences."""
        if a_norm == s_norm:
            return True
        core = s_norm
        for _ in range(2):                                   # strip "it prints", "the result is", ...
            core = cls._FILLER.sub("", core).strip()
        if core == a_norm:
            return True
        if q.type == "code_trace" or len(a_norm.split()) <= 2:
            return False
        return bool(re.search(rf"(?<!\w){re.escape(a_norm)}(?!\w)", s_norm))

    @staticmethod
    def _offline(q, s_raw, a_norm, s_norm) -> GradeResult:
        a_tok, s_tok = _content_tokens(q.answer), _content_tokens(s_raw)
        recall = len(a_tok & s_tok) / max(1, len(a_tok))
        miss = GradeResult(0.0, f"Expected: {q.answer}", q.misconception_tag, "overlap")
        if len(q.answer.split()) <= 4:
            # Short answers: outputs / symbols must match exactly; short wordy recall answers get keyword credit.
            wordy = sum(1 for t in a_tok if re.fullmatch(r"[a-z_]{3,}", t)) >= 2
            if q.type != "recall" or not wordy:
                return miss
            score = 1.0 if recall >= 1 else (0.5 if recall >= 0.5 else 0.0)
        else:
            first = a_norm.split()[0]
            if first in ("yes", "no"):
                if not s_norm.startswith(first):
                    return miss
                score = 0.6 + 0.4 * recall
            else:
                score = max(0.0, min(1.0, (recall - 0.25) / 0.5))
        score = round(score, 2)
        fb = "Correct." if score >= 0.8 else ("Partly correct." if score > 0 else f"Expected: {q.answer}")
        return GradeResult(score, fb, q.misconception_tag if score < 0.5 else None, "overlap")


# =============================================================================
# 7. MASTERY ESTIMATOR (Bayesian Knowledge Tracing, difficulty-aware, soft evidence)
# =============================================================================

@dataclass
class BKTParams:
    p_init: float = 0.30
    p_learn: float = 0.20
    mastered_p: float = 0.85
    gap_p: float = 0.40

    @staticmethod
    def guess(d: int) -> float:   # easy questions are easier to guess
        return max(0.10, 0.30 - 0.04 * (d - 1))

    @staticmethod
    def slip(d: int) -> float:    # hard questions are missed by students who know the concept
        return 0.05 + 0.04 * (d - 1)


class MasteryEstimator:
    def __init__(self, params: BKTParams = BKTParams()):
        self.k = params

    def update(self, cs: ConceptState, score: float, difficulty: int, step: int) -> Tuple[float, float]:
        """score in [0,1] is treated as soft evidence (partial credit). Returns (old_p, new_p)."""
        old, g, s = cs.p, BKTParams.guess(difficulty), BKTParams.slip(difficulty)
        p_c = old * (1 - s) + (1 - old) * g
        p_w = old * s + (1 - old) * (1 - g)
        post_c, post_w = old * (1 - s) / p_c, old * s / p_w
        post = score * post_c + (1 - score) * post_w
        cs.p = min(0.99, max(0.01, post + (1 - post) * self.k.p_learn))
        cs.attempts += 1
        cs.recent.append(score)
        cs.last_attempt_idx = step
        cs.needs_recheck = False
        return old, cs.p


# =============================================================================
# 8. DYNAMIC GRAPH (per-student overlay on the fixed ground truth)
# =============================================================================

STATUS_COLORS = {"unseen": "#BDBDBD", "inferred": "#A5D6A7", "learning": "#FFC107",
                 "mastered": "#4CAF50", "gap": "#F44336"}


class DynamicGraph:
    def __init__(self, graph: ConceptGraph, student: StudentState, params: BKTParams = BKTParams()):
        self.g, self.st, self.k = graph, student, params

    # ---- node status ---------------------------------------------------------
    def status(self, c: str) -> str:
        cs = self.st.cs[c]
        if cs.attempts == 0:
            return "inferred" if cs.p >= 0.75 else "unseen"   # inferred = implied by a mastered dependent
        if cs.p >= self.k.mastered_p and cs.attempts >= 2:
            return "mastered"
        if len(cs.recent) >= 2 and cs.p < self.k.gap_p:
            return "gap"
        return "learning"

    # ---- unlocking: weak edges use a softer threshold than strong ones -------
    @staticmethod
    def threshold(weight: int) -> float:
        return 0.70 if weight >= 8 else 0.55

    def prereqs_satisfied(self, c: str) -> bool:
        return all(self.st.cs[u].p >= self.threshold(w) for u, w in self.g.prereqs(c))

    def is_available(self, c: str) -> bool:
        return self.prereqs_satisfied(c)

    # ---- propagation ---------------------------------------------------------
    def propagate(self, c: str, score: float):
        """Success is evidence for prerequisites; a confirmed gap weakens unseen dependents."""
        if score >= 0.8:
            for u, w in self.g.prereqs(c):
                cs = self.st.cs[u]
                if cs.attempts < 2:                      # never overwrite real evidence
                    cs.p += 0.12 * (w / 10) * (1 - cs.p)
        cs_c = self.st.cs[c]
        if self.status(c) == "gap" and not cs_c.penalty_applied:
            cs_c.penalty_applied = True
            for d, w in self.g.dependents(c):
                cs = self.st.cs[d]
                if cs.attempts == 0:
                    cs.p = max(0.05, cs.p * (1 - 0.15 * w / 10))
        elif self.status(c) != "gap":
            cs_c.penalty_applied = False

    def infer_from_mastery(self, floor: float = 0.80):
        """Mastering a concept implies its (transitive) prerequisites, so we don't have to test them."""
        for m in self.g.order:
            if self.status(m) == "mastered":
                for a in self.g.ancestors(m):
                    cs = self.st.cs[a]
                    if cs.attempts == 0:
                        cs.p = max(cs.p, floor)

    # ---- diagnostic back-tracing ---------------------------------------------
    def backtrace(self, c: str) -> List[str]:
        """Walk from a failing concept to its most likely root cause.
        At each step choose the not-yet-mastered prerequisite with the highest
        weight * (1 - p) * (1 - 0.5 * confidence). Stops when no weak prereq remains."""
        chain, cur = [c], c
        while True:
            best, best_score = None, 0.0
            for u, w in self.g.prereqs(cur):
                if self.status(u) in ("mastered", "inferred"):
                    continue
                cs = self.st.cs[u]
                if cs.p >= self.threshold(w) and cs.attempts >= 2:
                    continue
                score = w * (1 - cs.p) * (1 - 0.5 * cs.confidence)
                if score > best_score:
                    best, best_score = u, score
            if best is None:
                return chain
            chain.append(best)
            cur = best

    def suspects(self, c: str) -> List[Tuple[str, float]]:
        """Prerequisites (direct or transitive) that could explain a failure on c, most likely first.
        score = path_strength * (1 - p) * (1 - 0.5 * confidence)
        path_strength = product of (weight/10) along the strongest dependency path to c."""
        strength = {c: 1.0}
        for v in reversed(self.g.order):               # dependents are processed before prerequisites
            if v in strength:
                for u, w in self.g.prereqs(v):
                    strength[u] = max(strength.get(u, 0.0), strength[v] * w / 10)
        idx = {x: i for i, x in enumerate(self.g.order)}
        out = []
        for u, st_ in strength.items():
            if u == c or self.status(u) in ("mastered", "inferred"):
                continue
            cs = self.st.cs[u]
            out.append((u, round(st_ * (1 - cs.p) * (1 - 0.5 * cs.confidence), 4)))
        return sorted(out, key=lambda t: (-t[1], idx[t[0]]))

    # ---- next-concept ranking ------------------------------------------------
    def rank_next(self) -> List[str]:
        cands = []
        for idx, c in enumerate(self.g.order):
            if self.status(c) == "mastered" or not self.is_available(c):
                continue
            downstream = sum(w for _, w in self.g.dependents(c))
            priority = (1 - self.st.cs[c].p) * (1 + 0.1 * downstream) - 0.001 * idx
            cands.append((priority, c))
        return [c for _, c in sorted(cands, reverse=True)]

    # ---- exports -------------------------------------------------------------
    def snapshot(self) -> dict:
        nodes = []
        for c in self.g.order:
            cs = self.st.cs[c]
            nodes.append({"concept_id": c, "name": self.g.name(c), "status": self.status(c),
                          "p_mastery": round(cs.p, 3), "confidence": round(cs.confidence, 3),
                          "attempts": cs.attempts, "available": self.is_available(c),
                          "color": STATUS_COLORS[self.status(c)]})
        return {"student_id": self.st.student_id, "nodes": nodes,
                "edges": [{"from": u, "to": v, "weight": w} for u, v, w in self.g.edges]}

    def to_dot(self) -> str:
        """Paste into https://dreampuf.github.io/GraphvizOnline or `dot -Tpng`."""
        lines = ["digraph kaizen {", "  rankdir=LR; node [style=filled, shape=box, fontname=Helvetica];"]
        for n in self.snapshot()["nodes"]:
            lines.append(f'  {n["concept_id"]} [label="{n["concept_id"]} {n["name"]}\\np={n["p_mastery"]:.2f}", '
                         f'fillcolor="{n["color"]}"];')
        for u, v, w in self.g.edges:
            lines.append(f'  {u} -> {v} [label="{w}", penwidth={0.5 + w / 5:.1f}];')
        lines.append("}")
        return "\n".join(lines)

    def render_text(self) -> str:
        sym = {"unseen": "·", "inferred": "≈", "learning": "~", "mastered": "✔", "gap": "✘"}
        rows = [f"{'ID':<4} {'Concept':<34} {'Status':<9} {'p(mastery)':<22} {'n':<3} avail"]
        for n in self.snapshot()["nodes"]:
            bar = "█" * round(n["p_mastery"] * 10) + "░" * (10 - round(n["p_mastery"] * 10))
            rows.append(f"{n['concept_id']:<4} {n['name'][:33]:<34} {sym[n['status']]} {n['status']:<7} "
                        f"{bar} {n['p_mastery']:.2f}  {n['attempts']:<3} {'yes' if n['available'] else 'no'}")
        return "\n".join(rows)


# =============================================================================
# 9. TUTOR AGENT (thin: explain a diagnosed gap, then hand back for re-check)
# =============================================================================

WORKED_EXAMPLES = {
    "c04": "7 // 2 is 3 (floor division) but 7 / 2 is 3.5 (true division). % gives the remainder: 7 % 2 == 1.",
    "c06": "x = 5\nif x > 3:      # True -> runs\n    print('A')\nelif x > 4:    # skipped: a branch already ran\n    print('B')",
    "c07": "for i in range(3):   # i = 0, 1, 2  (stops BEFORE 3)\n    print(i)\n\ni = 0\nwhile i < 3:\n    print(i)\n    i += 1           # without this the loop never ends",
}

TUTOR_SYSTEM = "You are a friendly Python tutor. Be concise: max 120 words, one worked example."


class TutorAgent:
    def __init__(self, llm: LLMClient, graph: ConceptGraph):
        self.llm, self.graph = llm, graph

    def teach(self, concept_id: str, misconception: Optional[str] = None) -> str:
        name, _, desc = self.graph.concepts[concept_id]
        prompt = (f"TASK: TUTOR\nConcept: {name}\nDescription: {desc}\n"
                  f"Student misconception: {misconception or 'unknown'}\n"
                  "Explain the idea, address the misconception, give one short worked example.")
        try:
            text = self.llm.complete(TUTOR_SYSTEM, prompt, max_tokens=500)
        except Exception:
            text = ""                                   # never crash a session because the tutor LLM failed
        if isinstance(self.llm, OfflineLLM) or not text.strip():
            text = (f"{name}: {desc}\nCommon slip: {misconception or 'n/a'}\n"
                    f"Worked example:\n{WORKED_EXAMPLES.get(concept_id, '(see course notes)')}")
        return text


# =============================================================================
# 10. ASSESSMENT AGENT (orchestrator)
# =============================================================================

_SAME = object()      # sentinel: "use the same client as the generator"


class AssessmentAgent:
    TYPE_ORDER = ("recall", "code_trace", "debug", "apply")

    def __init__(self, llm: Optional[LLMClient] = None, grader_llm=_SAME):
        """llm        -> question generation + tutoring
        grader_llm -> answer grading. Default: same client. Pass None for rule-based only,
                      or a separate client to keep generator and judge independent."""
        self.llm = llm or OfflineLLM()
        self.graph = ConceptGraph()
        self.bank: List[Question] = load_ground_truth_bank(self.graph)
        self.generator = QuestionGenerator(self.llm, self.graph)
        self.grader = AnswerGrader(self.llm if grader_llm is _SAME else grader_llm)
        self.lazy_generation = False        # True: generate LLM questions the first time a concept is needed
        self._expanded: set = set()
        self.bkt = MasteryEstimator()
        self.tutor = TutorAgent(self.llm, self.graph)

    # ---- bank management -----------------------------------------------------
    def expand_bank(self, concept_ids: List[str], qtype: str = "any", n: int = 3) -> int:
        added = 0
        for cid in concept_ids:
            new = self.generator.generate(cid, self.bank, qtype, n)
            self.bank.extend(new)
            added += len(new)
        return added

    def question_by_id(self, qid: str) -> Question:
        return next(q for q in self.bank if q.question_id == qid)

    # ---- planning ------------------------------------------------------------
    def plan(self, st: StudentState) -> Plan:
        dg = DynamicGraph(self.graph, st)
        for c in self.graph.order:                               # 1. pending re-checks first
            if st.cs[c].needs_recheck:
                return Plan("recheck", c, "Verify understanding after tutoring.")
        gaps = [c for c in self.graph.order if dg.status(c) == "gap"]
        if gaps:                                                 # 2. diagnose gaps
            failing = max(gaps, key=lambda c: st.cs[c].last_attempt_idx)
            chain = dg.backtrace(failing)
            root = chain[-1]
            if root == failing or dg.status(root) == "gap":
                return Plan("teach", root, f"Confirmed gap in {root}.", chain)
            return Plan("probe", root, f"{failing} keeps failing; probing weak prerequisite {root}.", chain)
        ranked = dg.rank_next()                                  # 3. otherwise move forward
        if not ranked:
            return Plan("done", None, "Nothing left to assess.")
        return Plan("assess", ranked[0], "Highest-priority unlocked concept.")

    # ---- question selection --------------------------------------------------
    def pick_question(self, st: StudentState, cid: str, allow_generate: bool = True) -> Optional[Question]:
        """Next unasked question for a concept. Rotates question type with the number of attempts
        (recall -> code_trace -> debug -> apply) so 'memorised' and 'understands' can be told apart."""
        def pool_():
            return [q for q in self.bank if q.concept_id == cid and q.question_id not in st.asked]
        if allow_generate and self.lazy_generation and cid not in self._expanded:
            self._expanded.add(cid)
            if not any(q.concept_id == cid and q.source == "generated" for q in self.bank):
                self.expand_bank([cid])
        pool = pool_()
        if not pool and allow_generate:                      # pool exhausted -> generate more
            self.expand_bank([cid])
            pool = pool_()
        if not pool:
            return None
        k = st.cs[cid].attempts % len(self.TYPE_ORDER)
        pref = self.TYPE_ORDER[k:] + self.TYPE_ORDER[:k]
        want_d = 1 + 4 * st.cs[cid].p
        pool.sort(key=lambda q: (pref.index(q.type), abs(q.difficulty - want_d)))
        return pool[0]

    def next_question(self, st: StudentState) -> Tuple[Plan, Optional[Question]]:
        plan = self.plan(st)
        if plan.action in ("done", "teach"):
            return plan, None
        targets = [plan.concept_id] if plan.action in ("probe", "recheck") else \
            DynamicGraph(self.graph, st).rank_next()
        for cid in targets:
            q = self.pick_question(st, cid)
            if q is not None:
                plan.concept_id = cid
                return plan, q
            if plan.action in ("probe", "recheck"):           # nothing left to ask -> teach instead
                return Plan("teach", cid, "Question pool exhausted; teach instead.", plan.chain), None
        return Plan("done", None, "No questions left."), None

    # ---- question-bank persistence (so a live demo needs zero generator calls) ----
    def save_generated(self, path: str) -> int:
        data = [asdict(q) for q in self.bank if q.source == "generated"]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return len(data)

    def load_generated(self, path: str) -> int:
        have = {q.question_id for q in self.bank}
        n = 0
        with open(path) as f:
            for d in json.load(f):
                if d["question_id"] not in have:
                    self.bank.append(Question(**d))
                    n += 1
        return n

    # ---- answering -----------------------------------------------------------
    def submit(self, st: StudentState, q: Question, answer: str) -> dict:
        grade = self.grader.grade(q, answer)
        st.asked.add(q.question_id)
        st.step += 1
        cs = st.cs[q.concept_id]
        old, new = self.bkt.update(cs, grade.score, q.difficulty, st.step)
        dg = DynamicGraph(self.graph, st)
        dg.propagate(q.concept_id, grade.score)
        dg.infer_from_mastery()
        event = {"step": st.step, "question_id": q.question_id, "concept_id": q.concept_id,
                 "type": q.type, "difficulty": q.difficulty, "score": grade.score,
                 "feedback": grade.feedback, "misconception": grade.misconception,
                 "p_before": round(old, 3), "p_after": round(new, 3),
                 "status": dg.status(q.concept_id), "grader": grade.method}
        st.history.append(event)
        return event

    # ---- tutoring hand-off ---------------------------------------------------
    def teach(self, st: StudentState, concept_id: str) -> str:
        wrong = [h["misconception"] for h in st.history
                 if h["concept_id"] == concept_id and h["score"] < 0.5 and h["misconception"]]
        text = self.tutor.teach(concept_id, wrong[-1] if wrong else None)
        cs = st.cs[concept_id]
        cs.recent.clear()                        # gap is re-evaluated from fresh evidence
        cs.p = max(cs.p, 0.45)
        cs.needs_recheck = True
        cs.penalty_applied = False
        return text

    # ---- reporting -----------------------------------------------------------
    def report(self, st: StudentState) -> dict:
        dg = DynamicGraph(self.graph, st)
        snap = dg.snapshot()
        by = {s: [n["concept_id"] for n in snap["nodes"] if n["status"] == s]
              for s in ("mastered", "inferred", "learning", "gap", "unseen")}
        return {"student_id": st.student_id, "summary": by,
                "next_recommended": dg.rank_next()[:3], "graph": snap}


# =============================================================================
# 11. DEMO: scripted student (strong basics, weak loops) + interactive mode
# =============================================================================

class SimulatedStudent:
    """Answers correctly with probability = skill[concept]. Wrong => 'not sure'."""
    def __init__(self, skill: Dict[str, float], seed: int = 7):
        self.skill, self.rng = dict(skill), random.Random(seed)

    def answer(self, q: Question) -> str:
        return q.answer if self.rng.random() < self.skill.get(q.concept_id, 0.5) else "not sure"

    def learn(self, cid: str):
        self.skill[cid] = 0.95


def run_demo(llm: Optional[LLMClient] = None, max_steps: int = 40):
    agent = AssessmentAgent(llm)
    added = agent.expand_bank(["c04", "c06", "c07"])
    st = StudentState("demo_student")
    sim = SimulatedStudent({"c01": 1, "c02": .95, "c03": .9, "c04": .85, "c05": .8, "c06": .55,
                            "c07": .1, "c08": .8, "c09": .8, "c10": .8, "c11": .8, "c12": .8,
                            "c13": .75, "c14": .8, "c15": .7, "c16": .7})
    print(f"KAIZEN assessment demo | bank = {len(agent.bank)} questions ({added} generated)\n")

    for _ in range(max_steps):
        plan, q = agent.next_question(st)
        if plan.action == "done":
            print("\n[agent] assessment complete:", plan.reason)
            break
        if plan.action == "teach":
            print(f"\n>>> TEACH {plan.concept_id} ({agent.graph.name(plan.concept_id)}) - {plan.reason}")
            if plan.chain and len(plan.chain) > 1:
                print("    back-trace chain:", " -> ".join(plan.chain))
            print("    " + agent.teach(st, plan.concept_id).replace("\n", "\n    "))
            sim.learn(plan.concept_id)
            continue
        ans = sim.answer(q)
        ev = agent.submit(st, q, ans)
        tag = {"assess": "ASSESS", "probe": "PROBE ", "recheck": "RECHECK"}[plan.action]
        print(f"[{tag}] {q.question_id} ({q.type}, d{q.difficulty}) score={ev['score']:.2f} "
              f"p {ev['p_before']:.2f}->{ev['p_after']:.2f} [{ev['status']}]"
              + (f"  <- {plan.reason}" if plan.action == "probe" else ""))
        if len(st.history) % 8 == 0:
            print()

    print("\n" + DynamicGraph(agent.graph, st).render_text())
    rep = agent.report(st)
    print("\nSummary:", json.dumps(rep["summary"]))
    print("Next recommended:", rep["next_recommended"])
    with open("kaizen_graph_snapshot.json", "w") as f:
        json.dump(rep["graph"], f, indent=2)
    with open("kaizen_graph.dot", "w") as f:
        f.write(DynamicGraph(agent.graph, st).to_dot())
    print("\nWrote kaizen_graph_snapshot.json and kaizen_graph.dot (render with Graphviz).")


def run_interactive(llm: Optional[LLMClient] = None):
    agent = AssessmentAgent(llm)
    agent.expand_bank(["c04", "c06", "c07"])
    st = StudentState(input("Student id: ").strip() or "student")
    print("Answer each question (type 'quit' to stop).\n")
    while True:
        plan, q = agent.next_question(st)
        if plan.action == "done":
            break
        if plan.action == "teach":
            print(f"\n--- Let's review {agent.graph.name(plan.concept_id)} ---")
            print(agent.teach(st, plan.concept_id), "\n")
            continue
        print(f"[{plan.action}] ({agent.graph.name(q.concept_id)}) {q.prompt}")
        ans = input("> ")
        if ans.strip().lower() == "quit":
            break
        ev = agent.submit(st, q, ans)
        print(f"  score {ev['score']:.2f} | {ev['feedback']}\n")
    print("\n" + DynamicGraph(agent.graph, st).render_text())
    open(f"{st.student_id}_state.json", "w").write(st.to_json())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the Claude API instead of the offline stub")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    client = AnthropicLLM() if args.live else OfflineLLM()
    run_interactive(client) if args.interactive else run_demo(client)
