#!/usr/bin/env python3
"""Worst-case stack depth from GCC call-graph files (-fcallgraph-info=su).

Compile the module with the EXACT flags of the target build plus
``-fcallgraph-info=su`` (see tools/px4_v6x_stack_report.sh); this script
merges the .ci files, and for each entry point reports the deepest call
chain and its byte total. Functions outside the analysed translation
units (libm, NuttX, PX4 libraries) have unknown frames and are listed so
their contribution can be bounded separately. Recursion and indirect
calls are reported, never silently ignored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NODE = re.compile(r'node: \{ title: "([^"]+)" label: "([^"]*)"')
EDGE = re.compile(r'edge: \{ sourcename: "([^"]+)" targetname: "([^"]+)"')
FRAME = re.compile(r'\\n(\d+) bytes \(([a-z,]+)\)')


def _key(title: str) -> str:
    # Static (local) symbols are prefixed by the file path: keep the name.
    return title.rsplit(":", 1)[-1] if title.startswith("/") else title


def load(paths: list[Path]) -> tuple[dict, dict, dict]:
    frames: dict[str, tuple[int, str]] = {}
    labels: dict[str, str] = {}
    edges: dict[str, set[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for title, label in NODE.findall(text):
            k = _key(title)
            labels.setdefault(k, label.split("\\n", 1)[0])
            m = FRAME.search(label)
            if m:
                frames[k] = (int(m.group(1)), m.group(2))
        for src, dst in EDGE.findall(text):
            edges.setdefault(_key(src), set()).add(_key(dst))
    return frames, labels, edges


def worst(root: str, frames: dict, edges: dict) -> dict:
    memo: dict[str, tuple[int, list[str]]] = {}
    unknown: set[str] = set()
    flags = {"recursion": False, "indirect": False, "dynamic": False}

    def visit(node: str, stack: tuple[str, ...]) -> tuple[int, list[str]]:
        if node in stack:
            flags["recursion"] = True
            return 0, [node + " (recursion)"]
        if node in memo:
            return memo[node]
        if node == "__indirect_call":
            flags["indirect"] = True
        own, kind = frames.get(node, (0, "unknown"))
        if node not in frames and node != "__indirect_call":
            unknown.add(node)
        if "dynamic" in kind and "bounded" not in kind:
            flags["dynamic"] = True
        best = (0, [])
        for child in sorted(edges.get(node, ())):
            depth, path = visit(child, stack + (node,))
            if depth > best[0]:
                best = (depth, path)
        result = (own + best[0], [f"{node} [{own} B]"] + best[1])
        memo[node] = result
        return result

    total, path = visit(root, ())
    return {"root": root, "worst_bytes": total, "chain": path,
            "unknown_callees": sorted(unknown), **flags}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ci_files", nargs="+", type=Path)
    parser.add_argument("--root", action="append", required=True,
                        help="Mangled entry point; substring match allowed")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    frames, labels, edges = load(args.ci_files)
    reports = []
    for pattern in args.root:
        matches = sorted(k for k in frames if pattern in k or pattern in labels.get(k, ""))
        if not matches:
            sys.exit(f"entry point not found: {pattern}")
        for root in matches:
            r = worst(root, frames, edges)
            r["label"] = labels.get(root, root)
            reports.append(r)
            print(f"{r['worst_bytes']:6d} B  {r['label']}")
            for step in r["chain"]:
                name = step.split(" [", 1)[0]
                print(f"          {labels.get(name, name)}  {step[len(name):]}")
            if r["unknown_callees"]:
                print("          unknown frames:", ", ".join(r["unknown_callees"][:12]))
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
