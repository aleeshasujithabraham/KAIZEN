"""
kaizen_viz.py - visual helpers for the KAIZEN dynamic concept graph
===================================================================

Pure matplotlib (preinstalled in Colab). Everything takes the plain-dict `snapshot`
produced by `DynamicGraph.snapshot()`, so it works with any front end you build later.

    draw_graph(ax, snapshot, ...)        one graph on a matplotlib axis
    plot_before_after(...)               start vs end graph, stacked
    plot_frames(...)                     graph "filmstrip" across the session
    plot_trajectory(...)                 p(mastery) per concept over time
    export_interactive(...)              optional draggable HTML graph (pip install pyvis)
"""
from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

# ---- style ------------------------------------------------------------------
FACE = {"unseen": "#CFD8DC", "inferred": "#C8E6C9", "learning": "#FFE082",
        "mastered": "#66BB6A", "gap": "#EF5350"}
RIM = {"unseen": "#90A4AE", "inferred": "#81C784", "learning": "#F9A825",
       "mastered": "#2E7D32", "gap": "#B71C1C"}
EDGE = {"unseen": "#B0BEC5", "inferred": "#66BB6A", "learning": "#F9A825",
        "mastered": "#2E7D32", "gap": "#C62828"}
LABEL = {"unseen": "not tested yet", "inferred": "inferred known (not tested)",
         "learning": "learning / shaky", "mastered": "mastered (verified)", "gap": "confirmed gap"}
PATH_COLOR, TARGET_COLOR = "#FB8C00", "#1A237E"

SHORT_NAMES = {
    "c01": "Python Basics", "c02": "Variables", "c03": "Data Types", "c04": "Operators",
    "c05": "Input & Output", "c06": "Conditionals", "c07": "Loops", "c08": "Strings",
    "c09": "Lists/Tuples/Sets", "c10": "Dictionaries", "c11": "Functions", "c12": "Modules",
    "c13": "Exceptions", "c14": "File Handling", "c15": "Comprehensions", "c16": "Recursion",
}
W, H, XSTEP, YSTEP = 1.12, 0.74, 1.55, 1.0


# ---- layout -----------------------------------------------------------------
def layered_layout(node_ids: Sequence[str], edges: List[dict]) -> Dict[str, Tuple[float, float]]:
    """Left-to-right layered layout: x = longest prerequisite chain, y ordered by barycentre."""
    pre = {n: [] for n in node_ids}
    for e in edges:
        pre[e["to"]].append(e["from"])
    depth: Dict[str, int] = {}

    def d(n):
        if n not in depth:
            depth[n] = 0 if not pre[n] else 1 + max(d(u) for u in pre[n])
        return depth[n]

    layers: Dict[int, List[str]] = {}
    for n in node_ids:
        layers.setdefault(d(n), []).append(n)
    pos, ypos = {}, {}
    for L in sorted(layers):
        nodes = layers[L]
        if L > 0:
            nodes = sorted(nodes, key=lambda n: sum(ypos[u] for u in pre[n]) / len(pre[n]), reverse=True)
        k = len(nodes)
        for i, n in enumerate(nodes):
            ypos[n] = (k - 1) / 2 - i
            pos[n] = (L * XSTEP, ypos[n] * YSTEP)
    return pos


# ---- single graph -----------------------------------------------------------
def draw_graph(ax, snapshot: dict, *, title: str = "", target: Optional[str] = None,
               path: Optional[Sequence[str]] = None, legend: bool = True):
    nodes = {n["concept_id"]: n for n in snapshot["nodes"]}
    pos = layered_layout(list(nodes), snapshot["edges"])
    path = list(path or [])

    for e in snapshot["edges"]:                                   # edges first (below nodes)
        (x1, y1), (x2, y2) = pos[e["from"]], pos[e["to"]]
        src = nodes[e["from"]]["status"]
        rad = 0.0
        if abs(x2 - x1) > XSTEP * 1.01:                  # long edge: bow it clear of the nodes in between
            chord = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            rad = min(0.45, 1.3 / chord) * (-1 if y2 >= y1 else 1)
        ax.add_patch(FancyArrowPatch(
            (x1 + W / 2, y1), (x2 - W / 2, y2), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=7, lw=0.5 + e["weight"] / 7,
            color=EDGE[src], alpha=0.75, zorder=1))

    for cid, n in nodes.items():
        x, y = pos[cid]
        st = n["status"]
        locked = st == "unseen" and not n["available"]
        is_target = cid == target
        ax.add_patch(FancyBboxPatch(
            (x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=FACE[st], ec=TARGET_COLOR if is_target else RIM[st],
            lw=3.0 if is_target else 1.2, ls="--" if locked else "-",
            alpha=0.6 if locked else 1.0, zorder=2))
        ax.text(x, y + 0.19, cid, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)
        ax.text(x, y - 0.01, SHORT_NAMES.get(cid, textwrap.shorten(n["name"], 16)),
                ha="center", va="center", fontsize=7, zorder=3)
        ax.text(x, y - 0.21, f"p={n['p_mastery']:.2f}", ha="center", va="center",
                fontsize=7, color="#37474F", zorder=3)
        if is_target:
            ax.text(x, y + H / 2 + 0.13, "★ target", ha="center", fontsize=7.5,
                    color=TARGET_COLOR, fontweight="bold", zorder=3)
        if cid in path:
            ax.add_patch(Circle((x - W / 2 + 0.02, y + H / 2 - 0.02), 0.17, fc=PATH_COLOR,
                                ec="white", lw=1.5, zorder=4))
            ax.text(x - W / 2 + 0.02, y + H / 2 - 0.02, str(path.index(cid) + 1), ha="center",
                    va="center", fontsize=8, color="white", fontweight="bold", zorder=5)

    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.75, max(ys) + 0.85)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if legend:
        add_legend(ax, with_path=bool(path))
    return ax


def add_legend(ax, with_path: bool = False):
    handles = [Patch(fc=FACE[s], ec=RIM[s], label=LABEL[s]) for s in FACE]
    handles.append(Patch(fc="white", ec="#90A4AE", ls="--", label="locked (prerequisites not satisfied)"))
    if with_path:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=PATH_COLOR,
                              markersize=10, label="recommended learning order"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
              ncol=len(handles), fontsize=7.5, frameon=False)


# ---- figures ----------------------------------------------------------------
def plot_before_after(start: dict, end: dict, *, target: str, path: Sequence[str],
                      filename: Optional[str] = None, suptitle: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(15, 11.5))
    draw_graph(axes[0], start, title="BEFORE - nothing known about the student", target=target, legend=False)
    draw_graph(axes[1], end, title="AFTER - personalised graph after the assessment", target=target, path=path)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=160, bbox_inches="tight")
    return fig


def plot_frames(frames: List[Tuple[str, dict]], *, target: str, path: Sequence[str],
                filename: Optional[str] = None, ncols: int = 2):
    """frames = [(title, snapshot), ...]; the learning-path badges appear only on the last frame."""
    nrows = (len(frames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5 * ncols, 6.2 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (title, snap) in enumerate(frames):
        last = i == len(frames) - 1
        draw_graph(axes[i], snap, title=f"{i + 1}. {title}", target=target,
                   path=path if last else None, legend=(i == 0 or last))
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=140, bbox_inches="tight")
    return fig


def plot_trajectory(events: List[dict], phases: List[Tuple[str, int, int]], *, target: str,
                    prior: float = 0.30, mastered_p: float = 0.85, gap_p: float = 0.40,
                    filename: Optional[str] = None, names: Optional[Dict[str, str]] = None):
    """p(mastery) of every concept the agent touched, one dot per answered question."""
    names = names or SHORT_NAMES
    fig, ax = plt.subplots(figsize=(12, 5.2))
    palette = plt.cm.tab10.colors
    concepts = []
    for ev in events:
        if ev["concept_id"] not in concepts:
            concepts.append(ev["concept_id"])
    shade = ["#E3F2FD", "#FFF3E0", "#F3E5F5", "#E8F5E9"]
    for i, (label, a, b) in enumerate(phases):
        ax.axvspan(a - 0.5, b + 0.5, color=shade[i % len(shade)], alpha=0.8, lw=0)
        ax.text((a + b) / 2, 1.04, label, ha="center", fontsize=9, fontweight="bold", color="#455A64")
    ax.axhline(mastered_p, ls="--", lw=0.8, color="#2E7D32")
    ax.text(0.55, mastered_p + 0.012, "mastered ≥ 0.85", fontsize=7.5, color="#2E7D32")
    ax.axhline(gap_p, ls="--", lw=0.8, color="#B71C1C")
    ax.text(0.55, gap_p - 0.035, "gap < 0.40", fontsize=7.5, color="#B71C1C")
    for ci, cid in enumerate(concepts):
        evs = [e for e in events if e["concept_id"] == cid]
        xs = [evs[0]["step"] - 1] + [e["step"] for e in evs]
        ys = [evs[0]["p_before"]] + [e["p_after"] for e in evs]
        col = palette[ci % len(palette)]
        ax.plot(xs, ys, "-", color=col, lw=3.0 if cid == target else 1.8, alpha=0.9,
                label=f"{cid} {names.get(cid, '')}" + ("  ★ target" if cid == target else ""))
        for e in evs:
            mk = "#2E7D32" if e["score"] >= 0.7 else ("#C62828" if e["score"] <= 0.3 else "#F9A825")
            ax.scatter(e["step"], e["p_after"], s=70, color=mk, edgecolor=col, linewidth=2, zorder=5)
    ax.set_xlim(0.4, max(e["step"] for e in events) + 0.6)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("question # (dot fill: green = correct, amber = partial, red = wrong)")
    ax.set_ylabel("P(mastery)")
    ax.set_title("Mastery trajectory per concept", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=160, bbox_inches="tight")
    return fig


# ---- optional interactive export --------------------------------------------
def export_interactive(snapshot: dict, html_path: str, *, target: Optional[str] = None,
                       path: Optional[Sequence[str]] = None) -> Optional[str]:
    """Draggable/zoomable HTML graph (hover a node for details). Needs `pip install pyvis`."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed -> skipping interactive export (pip install pyvis)")
        return None
    path = list(path or [])
    net = Network(height="720px", width="100%", directed=True, cdn_resources="in_line")
    net.set_options('{"layout":{"hierarchical":{"enabled":true,"direction":"LR","sortMethod":"directed",'
                    '"levelSeparation":170,"nodeSpacing":110}},"physics":{"enabled":false}}')
    for n in snapshot["nodes"]:
        cid = n["concept_id"]
        order = f"  [learn #{path.index(cid) + 1}]" if cid in path else ""
        net.add_node(cid, label=f"{cid}\n{SHORT_NAMES.get(cid, n['name'])}\np={n['p_mastery']:.2f}",
                     title=f"{n['name']}<br>status: {n['status']}<br>p(mastery): {n['p_mastery']:.2f}"
                           f"<br>attempts: {n['attempts']}{order}",
                     color={"background": FACE[n["status"]],
                            "border": PATH_COLOR if cid in path else RIM[n["status"]]},
                     shape="box", borderWidth=4 if (cid == target or cid in path) else 1)
    for e in snapshot["edges"]:
        net.add_edge(e["from"], e["to"], width=0.5 + e["weight"] / 3, title=f"weight {e['weight']}",
                     color=EDGE[next(n["status"] for n in snapshot["nodes"] if n["concept_id"] == e["from"])])
    try:
        net.write_html(html_path)
    except Exception:
        net.save_graph(html_path)
    return html_path
