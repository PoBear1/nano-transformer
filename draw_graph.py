#!/usr/bin/env python3
"""
git_history_svg.py

Renders a repo's commit graph as an SVG in the "Learn Git Branching" style:
circles for commits, arrows for parent links, boxes for branch/tag refs,
and greyed-out floating commits for anything orphaned (e.g. left behind by
a rebase or reset) that's only reachable via the reflog.

Usage:
    python3 git_history_svg.py /path/to/repo -o history.svg
    python3 git_history_svg.py .                  # writes ./git_history.svg
    python3 git_history_svg.py . --labels subject  # label nodes with commit
                                                    # subject instead of hash
"""
import argparse
import subprocess
import sys
from collections import defaultdict

# ---------- layout / style constants ----------
ROW_H = 90
LANE_W = 90
NODE_R = 32
MARGIN_X = 80
MARGIN_TOP = 60
MARGIN_BOTTOM = 60
FONT = "Menlo, Consolas, 'Courier New', monospace"

BG = "#6fa8dc"
GREEN = "#8ee6b8"
GREEN_STROKE = "#3fae7a"
GREY = "#9fb6c9"
GREY_STROKE = "#7f97ab"
ARROW = "#2b2b2b"
GREY_ARROW = "#b9c8d6"
TAG_GREEN = "#7ee08a"
TEXT_DARK = "#1a1a1a"


def run_git(repo, args):
    result = subprocess.run(
        ["git", "-C", repo] + args, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def get_current_branch(repo):
    out = run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    return out if out != "HEAD" else None


def parse_log(repo, extra_args):
    """Return list of commit dicts, newest first, topologically ordered."""
    fmt = "%H%x01%P%x01%s%x01%D"
    out = run_git(
        repo, ["log", "--topo-order", "--date-order", f"--pretty=format:{fmt}"] + extra_args
    )
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        h, p, subject, refs = line.split("\x01", 3)
        commits.append(
            {
                "hash": h,
                "short": h[:7],
                "parents": p.split() if p else [],
                "subject": subject,
                "refs": [r.strip() for r in refs.split(",")] if refs else [],
            }
        )
    return commits


def get_orphaned_commits(repo, known_hashes):
    """Commits only reachable via reflog (e.g. pre-rebase state)."""
    try:
        out = run_git(
            repo,
            ["log", "--walk-reflogs", "--all", "--pretty=format:%H%x01%P%x01%s"],
        )
    except RuntimeError:
        return []
    seen = set()
    orphaned = []
    for line in out.splitlines():
        if not line.strip():
            continue
        h, p, subject = line.split("\x01", 2)
        if h in known_hashes or h in seen:
            continue
        seen.add(h)
        orphaned.append({"hash": h, "short": h[:7], "parents": p.split() if p else [], "subject": subject})
    return orphaned


def assign_lanes(commits):
    """Simplified git-log --graph style lane assignment.
    Returns {hash: lane_index}."""
    lanes = []  # lanes[i] = hash this lane is waiting to place next, or None
    positions = {}
    for c in commits:
        h = c["hash"]
        lane_idx = None
        for i, waiting in enumerate(lanes):
            if waiting == h:
                lane_idx = i
                break
        if lane_idx is None:
            for i, waiting in enumerate(lanes):
                if waiting is None:
                    lane_idx = i
                    break
            if lane_idx is None:
                lane_idx = len(lanes)
                lanes.append(None)
        positions[h] = lane_idx

        parents = c["parents"]
        if parents:
            lanes[lane_idx] = parents[0]
            for extra in parents[1:]:
                free = None
                for i, waiting in enumerate(lanes):
                    if waiting is None:
                        free = i
                        break
                if free is None:
                    free = len(lanes)
                    lanes.append(extra)
                else:
                    lanes[free] = extra
        else:
            lanes[lane_idx] = None
    return positions


def esc(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def truncate(s, n=22):
    return s if len(s) <= n else s[: n - 1] + "…"


def build_svg(repo, label_mode="hash"):
    current_branch = get_current_branch(repo)
    commits = parse_log(repo, ["--all"])
    known = {c["hash"] for c in commits}
    orphaned = get_orphaned_commits(repo, known)

    lanes = assign_lanes(commits)
    max_main_lane = max(lanes.values(), default=0)

    orphan_lanes = assign_lanes(orphaned) if orphaned else {}
    orphan_lane_offset = max_main_lane + 2
    for h in orphan_lanes:
        orphan_lanes[h] += orphan_lane_offset

    all_positions = {}
    row = 0
    for c in commits:
        all_positions[c["hash"]] = (lanes[c["hash"]], row)
        row += 1
    orphan_row_start = 0
    for c in orphaned:
        all_positions[c["hash"]] = (orphan_lanes[c["hash"]], orphan_row_start)
        orphan_row_start += 1

    n_lanes = max(list(lanes.values()) + list(orphan_lanes.values()) + [0]) + 1
    n_rows = max(len(commits), len(orphaned))

    width = MARGIN_X * 2 + n_lanes * LANE_W + 260  # extra room for ref labels
    height = MARGIN_TOP + MARGIN_BOTTOM + n_rows * ROW_H

    def xy(hash_):
        lane, r = all_positions[hash_]
        x = MARGIN_X + lane * LANE_W + NODE_R
        y = height - MARGIN_BOTTOM - r * ROW_H - NODE_R
        return x, y

    svg = []
    svg.append(
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="{FONT}">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>')
    svg.append(
        """<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/>
  </marker>
  <marker id="arrowGrey" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/>
  </marker>
</defs>"""
        % (ARROW, GREY_ARROW)
    )

    # ---- edges: main graph ----
    for c in commits:
        x1, y1 = xy(c["hash"])
        for p in c["parents"]:
            if p not in all_positions:
                continue
            x2, y2 = xy(p)
            if x1 == x2:
                svg.append(
                    f'<line x1="{x1}" y1="{y1-NODE_R}" x2="{x2}" y2="{y2+NODE_R+8}" '
                    f'stroke="{ARROW}" stroke-width="3" marker-end="url(#arrow)"/>'
                )
            else:
                midy = (y1 + y2) / 2
                path = f"M {x1} {y1-NODE_R} C {x1} {midy}, {x2} {midy}, {x2} {y2+NODE_R+8}"
                svg.append(
                    f'<path d="{path}" fill="none" stroke="{ARROW}" stroke-width="3" marker-end="url(#arrow)"/>'
                )

    # ---- edges: orphaned chain (dashed grey) ----
    for c in orphaned:
        x1, y1 = xy(c["hash"])
        for p in c["parents"]:
            if p in all_positions:
                x2, y2 = xy(p)
                midy = (y1 + y2) / 2
                path = (
                    f"M {x1} {y1-NODE_R} C {x1} {midy}, {x2} {midy}, {x2} {y2+NODE_R+8}"
                    if x1 != x2
                    else f"M {x1} {y1-NODE_R} L {x2} {y2+NODE_R+8}"
                )
                svg.append(
                    f'<path d="{path}" fill="none" stroke="{GREY_ARROW}" stroke-width="3" '
                    f'stroke-dasharray="2,6" marker-end="url(#arrowGrey)"/>'
                )

    # ---- ref labels (branches/tags/HEAD) ----
    for c in commits:
        x, y = xy(c["hash"])
        labels = []
        for r in c["refs"]:
            if not r:
                continue
            name = r.replace("HEAD -> ", "")
            is_head = r.startswith("HEAD ->") or name == current_branch
            if name.startswith("tag:"):
                name = name.strip()
            labels.append((name, is_head))
        for i, (name, is_head) in enumerate(labels):
            lx = x + NODE_R + 40 + i * 4
            ly = y - (len(labels) - 1 - i) * 34
            disp = esc(name) + ("*" if is_head else "")
            box_w = 18 + len(disp) * 11
            svg.append(
                f'<rect x="{lx}" y="{ly-16}" width="{box_w}" height="32" rx="8" '
                f'fill="{TAG_GREEN}" stroke="#2f8f4e" stroke-width="2"/>'
            )
            svg.append(
                f'<text x="{lx+box_w/2}" y="{ly+6}" text-anchor="middle" '
                f'font-size="16" font-weight="bold" fill="{TEXT_DARK}">{disp}</text>'
            )
            svg.append(
                f'<path d="M {lx-2} {ly} L {x+NODE_R+14} {y}" stroke="#3fae7a" '
                f'stroke-width="4" marker-end="url(#arrow)"/>'
            )

    # ---- nodes ----
    def draw_node(c, x, y, grey=False):
        fill = GREY if grey else GREEN
        stroke = GREY_STROKE if grey else GREEN_STROKE
        svg.append(f'<circle cx="{x}" cy="{y}" r="{NODE_R}" fill="{fill}" stroke="{stroke}" stroke-width="4"/>')
        label = c["short"] if label_mode == "hash" else truncate(c["subject"])
        fs = 14 if label_mode == "hash" else 11
        svg.append(
            f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="{fs}" '
            f'font-weight="bold" fill="{TEXT_DARK}">{esc(label)}</text>'
        )
        svg.append(f"<title>{esc(c['hash'])}: {esc(c['subject'])}</title>")

    for c in commits:
        x, y = xy(c["hash"])
        draw_node(c, x, y)
    for c in orphaned:
        x, y = xy(c["hash"])
        draw_node(c, x, y, grey=True)

    svg.append("</svg>")
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description="Render a git repo's commit history as an SVG.")
    ap.add_argument("repo", nargs="?", default=".", help="Path to the git repo (default: current dir)")
    ap.add_argument("-o", "--output", default="git_history.svg", help="Output SVG path")
    ap.add_argument(
        "--labels",
        choices=["hash", "subject"],
        default="hash",
        help="Label commit circles with short hash (default) or commit subject",
    )
    args = ap.parse_args()

    try:
        svg = build_svg(args.repo, label_mode=args.labels)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w") as f:
        f.write(svg)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()