#!/usr/bin/env python3
"""Assignment 4 analysis: standard loop vs. code <-> pseudocode loop.

Reads every ``*_results.json`` written by ``evaloop evaluate`` in a results directory and writes,
into ``--out_dir``:

  summary.csv               one row per (model, configuration) run (the required deliverable)
  metrics_full.csv          + ASL_bias, pass_drop, pass@k curve, mean prompt length
  paired_comparison.csv     per model: extension vs. standard loop on the same tasks
                            (win/tie/loss counts, paired-bootstrap CI of the ASL difference)
  pseudocode_leakage.csv    how much Python the "pseudocode" prompts contain
  failure_labels.csv        heuristic pre-label of every failed loop (input for manual review)
  failure_label_counts.csv  pre-label distribution per run
  manual_inspection.md      sampled failed loops with full context, for manual labelling
  manual_labels.csv         template to record your manual labels for those samples
  cases.md                  candidate "broken by the extension" / "survived" examples
  datapackage_check.csv     (with --data_package) our standard runs vs. the provided records
  survival_curves.png       pass@k vs. k for every run

Only the loop counts are used for the ASL numbers (no LLM judge): ``asl_score`` is ASL_pow,
i.e. the leaderboard's ``aslScore`` column. Usage:

    python a4/analyze_a4.py --results_dir results/a4 --out_dir results/a4/analysis \
        --data_package ../A4/assignment2-data
"""

import argparse
import ast
import csv
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse on model code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaloop.analysis.metrics import asl_bias, asl_pow, pass_at_k, pass_drop  # noqa: E402

CONFIG_NAMES = {
    "code_generation_summarization": "standard-mbpp-plus",
    "code_generation_pseudocode": "pseudocode-mbpp-plus",
    "code_generation_pseudocode_noassert": "pseudocode-noassert-mbpp-plus",
}
BASELINE = "standard-mbpp-plus"

# Maps our HuggingFace model paths to data-package model ids (for the cross-check)
DATA_PACKAGE_IDS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder-7b-instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "qwen2.5-coder-14b-instruct",
    "ByteDance-Seed/Seed-Coder-8B-Instruct": "seed-coder-8b-instruct",
    "infly/OpenCoder-8B-Instruct": "opencoder-8b-instruct",
    "Qwen/CodeQwen1.5-7B-Chat": "codeqwen1.5-7b-chat",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemma-3n-E4B-it": "gemma-3n-e4b-it",
    "openbmb/MiniCPM4-8B": "minicpm4-8b",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "deepseek-coder-v2-lite-instruct",
}


# --------------------------------------------------------------------------- loading

def load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Loads every results file into a flat run record."""
    runs = []
    for path in sorted(results_dir.glob("*_results.json")):
        data = json.loads(path.read_text())
        for model_result in data if isinstance(data, list) else [data]:
            cfg = model_result.get("config", {})
            loop_type = model_result.get("loop_type") or cfg.get("loop_type") or _loop_from_name(path.name)
            runs.append({
                "path": path,
                "model": cfg.get("model") or model_result.get("model", path.stem),
                "loop_type": loop_type,
                "configuration": CONFIG_NAMES.get(loop_type, loop_type),
                "max_cycles": model_result.get("max_cycles", 10),
                "tasks": {p["task_id"]: p for p in model_result["prompt_results"]},
            })
    if not runs:
        sys.exit(f"No *_results.json files in {results_dir}")
    return runs


def _loop_from_name(name: str) -> str:
    for loop_type in sorted(CONFIG_NAMES, key=len, reverse=True):
        if loop_type in name:
            return loop_type
    return "unknown"


def load_entry_points() -> Dict[str, str]:
    """Entry points from evalplus when available (otherwise inferred per task)."""
    try:
        from evalplus.data import get_mbpp_plus
        return {tid: p["entry_point"] for tid, p in get_mbpp_plus().items()}
    except Exception:
        return {}


def assert_line(prompt: str) -> str:
    lines = (prompt or "").splitlines()
    return lines[-2] if len(lines) >= 2 else ""


def infer_entry_point(task: Dict[str, Any], entry_points: Dict[str, str]) -> Optional[str]:
    if task["task_id"] in entry_points:
        return entry_points[task["task_id"]]
    line = assert_line(task["initial_prompt"])
    called = re.findall(r"([A-Za-z_]\w*)\s*\(", line)
    for cycle in task["cycles"]:
        defined = defined_functions(cycle.get("task_1_output", ""))
        for name in called:
            if name in defined:
                return name
    builtins = {"set", "math", "isclose", "abs", "sorted", "list", "tuple", "round", "str", "len"}
    return next((n for n in called if n not in builtins), None)


# --------------------------------------------------------------------------- code helpers

def parse(code: str) -> Optional[ast.Module]:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def defined_functions(code: str) -> set:
    tree = parse(code)
    if tree is None:
        return set()
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def has_toplevel_assert(code: str) -> bool:
    tree = parse(code)
    return tree is not None and any(isinstance(n, ast.Assert) for n in tree.body)


def returns_only_literal(code: str, entry_point: Optional[str]) -> bool:
    """True if the entry function just returns a constant (hard-coding the visible assert)."""
    tree = parse(code)
    if tree is None or not entry_point:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            body = [s for s in node.body if not isinstance(s, ast.Expr)]
            return (len(body) == 1 and isinstance(body[0], ast.Return)
                    and body[0].value is not None and _is_literal(body[0].value))
    return False


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        return False


def prompt_body(prompt: str) -> str:
    """The description inside the docstring wrapper, without the assert line."""
    lines = [ln for ln in (prompt or "").strip().splitlines() if ln.strip() != '"""']
    return "\n".join(ln for ln in lines if not ln.lstrip().startswith("assert "))


TOKEN = re.compile(r"[A-Za-z_]\w*|==|!=|<=|>=|\*\*|//|[-+*/%<>=\[\](){}:,.]")


def leakage(pseudocode: str, code: str) -> Dict[str, float]:
    """How close a pseudocode prompt is to the Python it was derived from."""
    body = prompt_body(pseudocode)
    tree = parse(body)
    parses = tree is not None and any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
    code_lines = {ln.strip() for ln in code.splitlines() if ln.strip()}
    p_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    copied = sum(ln in code_lines for ln in p_lines) / len(p_lines) if p_lines else 0.0
    a, b = set(TOKEN.findall(body)), set(TOKEN.findall(code))
    jaccard = len(a & b) / len(a | b) if a | b else 0.0
    python_markers = sum(bool(re.search(r"^\s*(def |return |elif |for \w+ in |import )|:\s*$", ln))
                         for ln in body.splitlines())
    return {
        "parses_as_python": float(parses),
        "line_copy_ratio": copied,
        "token_jaccard": jaccard,
        "python_marker_lines": python_markers / max(len(p_lines), 1),
        "chars": len(body),
    }


# --------------------------------------------------------------------------- failures

def failing_cycle(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for cycle in task["cycles"]:
        if not cycle.get("success"):
            return cycle
    return None


def label_failure(task: Dict[str, Any], entry_point: Optional[str]) -> Dict[str, Any]:
    """Heuristic pre-label of a failed loop; manual review decides the final category."""
    fail = failing_cycle(task)
    if fail is None:
        return {}
    code = fail.get("task_1_output", "")
    tests = fail.get("test_results", {})
    if any(k.endswith("_error") for k in fail):
        label = "generation_error"
    elif parse(code) is None:
        label = "syntax_or_extraction"
    elif entry_point and entry_point not in defined_functions(code):
        label = "signature_drift"
    elif returns_only_literal(code, entry_point):
        label = "hardcoded_assert"
    elif tests.get("base_success") and not tests.get("plus_success"):
        label = "edge_case_regression"
    else:
        label = "wrong_behaviour"
    # A pasted copy of the visible assert never causes a failure by itself (a correct function
    # passes it), so it is recorded as a flag, not as the failure category.
    return {"fail_cycle": fail["cycle"], "auto_label": label, "code_embeds_assert": has_toplevel_assert(code),
            "base_passed": tests.get("base_success"), "plus_passed": tests.get("plus_success")}


def cycle_prompt(task: Dict[str, Any], cycle_number: int) -> str:
    """The generation input of a cycle (original prompt, else previous cycle's output)."""
    if cycle_number == 1:
        return task["initial_prompt"]
    return task["cycles"][cycle_number - 2].get("task_2_output", "")


# --------------------------------------------------------------------------- stats

def counts(run: Dict[str, Any], task_ids: Optional[List[str]] = None) -> List[int]:
    ids = task_ids if task_ids is not None else list(run["tasks"])
    return [run["tasks"][t]["successful_cycles"] for t in ids]


def asl(values: List[int], m: int) -> float:
    return asl_pow(values, m)


def paired_bootstrap(a: List[int], b: List[int], m: int, n_boot: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(asl([b[i] for i in idx], m) - asl([a[i] for i in idx], m))
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def wilcoxon_p(a: List[int], b: List[int]) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon
        if all(x == y for x, y in zip(a, b)):
            return 1.0
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return None


# --------------------------------------------------------------------------- outputs

def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None):
    if not rows:
        return
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_package", help="assignment data folder (with loops/ and leaderboard.json)")
    ap.add_argument("--samples_per_run", type=int, default=12,
                    help="failed loops sampled per extension run for manual inspection")
    ap.add_argument("--baseline_samples_per_run", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.results_dir))
    entry_points = load_entry_points()

    # ---- summary.csv and full metrics
    summary, full = [], []
    for run in sorted(runs, key=lambda r: (r["model"], r["configuration"] != BASELINE, r["configuration"])):
        values, m = counts(run), run["max_cycles"]
        n = len(values)
        prompt_chars = [len(prompt_body(c.get("task_2_output", "")))
                        for t in run["tasks"].values() for c in t["cycles"] if c.get("task_2_output")]
        row = {"model": run["model"], "configuration": run["configuration"], "n_tasks": n,
               "success_rate": round(pass_at_k(values, 1) / n, 3), "asl_score": round(asl(values, m), 3)}
        summary.append(row)
        full.append({**row, "asl_bias": round(asl_bias(values, m), 3),
                     "pass_drop": round(pass_drop(values, m), 3),
                     "mean_sustained_loops": round(sum(values) / n, 3),
                     "mean_prompt_chars": round(sum(prompt_chars) / max(len(prompt_chars), 1), 1),
                     **{f"pass@{k}": pass_at_k(values, k) for k in range(1, m + 1)}})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "metrics_full.csv", full)

    by_key = {(r["model"], r["configuration"]): r for r in runs}
    models = sorted({r["model"] for r in runs})
    configs = sorted({r["configuration"] for r in runs}, key=lambda c: (c != BASELINE, c))

    # ---- paired comparison against the standard loop
    paired = []
    for model in models:
        base = by_key.get((model, BASELINE))
        for config in configs:
            ext = by_key.get((model, config))
            if base is None or ext is None or config == BASELINE:
                continue
            ids = sorted(set(base["tasks"]) & set(ext["tasks"]))
            a, b, m = counts(base, ids), counts(ext, ids), base["max_cycles"]
            lo, hi = paired_bootstrap(a, b, m, seed=args.seed)
            paired.append({
                "model": model, "configuration": config, "n_tasks": len(ids),
                "asl_baseline": round(asl(a, m), 3), "asl_extension": round(asl(b, m), 3),
                "asl_diff": round(asl(b, m) - asl(a, m), 3), "diff_ci95_low": round(lo, 3),
                "diff_ci95_high": round(hi, 3),
                "tasks_extension_longer": sum(y > x for x, y in zip(a, b)),
                "tasks_equal": sum(y == x for x, y in zip(a, b)),
                "tasks_extension_shorter": sum(y < x for x, y in zip(a, b)),
                "pass1_both": sum(x >= 1 and y >= 1 for x, y in zip(a, b)),
                "full10_baseline_only": sum(x == m and y < m for x, y in zip(a, b)),
                "full10_extension_only": sum(y == m and x < m for x, y in zip(a, b)),
                "wilcoxon_p": fmt(wilcoxon_p(a, b), 4),
            })
    write_csv(out / "paired_comparison.csv", paired)

    # ---- ranking per configuration
    ranking = []
    for config in configs:
        scored = sorted(((by_key[(m_, config)], m_) for m_ in models if (m_, config) in by_key),
                        key=lambda x: -asl(counts(x[0]), x[0]["max_cycles"]))
        for rank, (run, model) in enumerate(scored, 1):
            ranking.append({"configuration": config, "rank": rank, "model": model,
                            "asl_score": round(asl(counts(run), run["max_cycles"]), 3),
                            "success_rate": round(pass_at_k(counts(run), 1) / len(run["tasks"]), 3)})
    write_csv(out / "ranking.csv", ranking)

    # ---- pseudocode leakage (and the same statistics for standard summaries, as a reference)
    leak_rows = []
    for run in runs:
        stats = defaultdict(list)
        for task in run["tasks"].values():
            for cycle in task["cycles"]:
                if cycle.get("task_2_output"):
                    for k, v in leakage(cycle["task_2_output"], cycle.get("task_1_output", "")).items():
                        stats[k].append(v)
        if stats:
            leak_rows.append({"model": run["model"], "configuration": run["configuration"],
                              "n_prompts": len(stats["chars"]),
                              **{k: round(sum(v) / len(v), 3) for k, v in stats.items()}})
    write_csv(out / "pseudocode_leakage.csv", leak_rows)

    # ---- failure pre-labels
    label_rows = []
    for run in runs:
        for task in run["tasks"].values():
            info = label_failure(task, infer_entry_point(task, entry_points))
            if info:
                label_rows.append({"model": run["model"], "configuration": run["configuration"],
                                   "task_id": task["task_id"], **info})
    write_csv(out / "failure_labels.csv", label_rows)
    label_counts = []
    for (model, config), rows in sorted(_group(label_rows, ("model", "configuration")).items()):
        loop_rows = [r for r in rows if r["fail_cycle"] > 1]
        c = Counter(r["auto_label"] for r in loop_rows)
        label_counts.append({"model": model, "configuration": config,
                             "failed_at_cycle1": len(rows) - len(loop_rows),
                             "failed_in_loop": len(loop_rows), **c,
                             "code_embeds_assert": sum(bool(r["code_embeds_assert"]) for r in loop_rows)})
    fields = ["model", "configuration", "failed_at_cycle1", "failed_in_loop",
              "syntax_or_extraction", "signature_drift", "hardcoded_assert",
              "edge_case_regression", "wrong_behaviour", "generation_error", "code_embeds_assert"]
    write_csv(out / "failure_label_counts.csv", label_counts, fields)

    # ---- manual-inspection sample (loop failures only: cycle >= 2)
    rng = random.Random(args.seed)
    md, manual = ["# Failed loops sampled for manual inspection\n",
                  "Label each with one of: requirement_dropped, detail_hallucinated, "
                  "cannot_implement_own_description, signature_drift, formatting_extraction, "
                  "assert_overreliance, edge_case_lost, other. Record labels in manual_labels.csv.\n"], []
    for run in sorted(runs, key=lambda r: (r["configuration"] == BASELINE, r["model"])):
        k = args.baseline_samples_per_run if run["configuration"] == BASELINE else args.samples_per_run
        pool = [r for r in label_rows if r["model"] == run["model"]
                and r["configuration"] == run["configuration"] and r["fail_cycle"] > 1]
        for r in sorted(rng.sample(pool, min(k, len(pool))), key=lambda r: r["task_id"]):
            task = run["tasks"][r["task_id"]]
            f = r["fail_cycle"]
            manual.append({**r, "manual_label": "", "notes": ""})
            md += [f"\n---\n\n## {run['model']} · {run['configuration']} · {r['task_id']} · "
                   f"failed at cycle {f} (auto: `{r['auto_label']}`, base={r['base_passed']}, "
                   f"plus={r['plus_passed']})\n",
                   "**Original prompt**\n", _block(task["initial_prompt"]),
                   f"**Last passing code (cycle {f - 1})**\n", _block(task["cycles"][f - 2].get("task_1_output", ""), "python"),
                   f"**Prompt that failed (written in cycle {f - 1})**\n", _block(cycle_prompt(task, f)),
                   f"**Failing code (cycle {f})**\n", _block(task["cycles"][f - 1].get("task_1_output", ""), "python")]
    (out / "manual_inspection.md").write_text("\n".join(md))
    write_csv(out / "manual_labels.csv", manual,
              ["model", "configuration", "task_id", "fail_cycle", "auto_label", "manual_label", "notes"])

    # ---- representative cases
    cases = ["# Representative cases\n"]
    for model in models:
        base = by_key.get((model, BASELINE))
        for config in configs:
            ext = by_key.get((model, config))
            if base is None or ext is None or config == BASELINE:
                continue
            m = base["max_cycles"]
            ids = sorted(set(base["tasks"]) & set(ext["tasks"]))
            broke = [t for t in ids if base["tasks"][t]["successful_cycles"] == m
                     and 1 <= ext["tasks"][t]["successful_cycles"] < m]
            held = [t for t in ids if ext["tasks"][t]["successful_cycles"] == m
                    and 1 <= base["tasks"][t]["successful_cycles"] < m]
            both = [t for t in ids if ext["tasks"][t]["successful_cycles"] == m == base["tasks"][t]["successful_cycles"]]
            cases.append(f"\n## {model} · {config}\n")
            cases.append(f"- Survived 10 in the standard loop, broken by {config} ({len(broke)}): "
                         + ", ".join(f"{t} (failed at {ext['tasks'][t]['successful_cycles'] + 1})" for t in broke))
            cases.append(f"- Survived 10 under {config}, broken in the standard loop ({len(held)}): "
                         + ", ".join(f"{t} (failed at {base['tasks'][t]['successful_cycles'] + 1})" for t in held))
            cases.append(f"- Survived 10 in both: {len(both)} tasks\n")
    (out / "cases.md").write_text("\n".join(cases))

    # ---- data-package cross-check of our own standard runs
    if args.data_package:
        rows = []
        pkg = Path(args.data_package)
        leaderboard = {m_["model_id"]: m_ for m_ in json.loads((pkg / "leaderboard.json").read_text())["models"]}
        for model in models:
            run = by_key.get((model, BASELINE))
            pid = DATA_PACKAGE_IDS.get(model)
            if run is None or not pid or not (pkg / "loops" / f"{pid}.json").exists():
                continue
            ref = {t["task_id"]: t["successful_cycles"]
                   for t in json.loads((pkg / "loops" / f"{pid}.json").read_text())["tasks"]}
            ids = sorted(set(ref) & set(run["tasks"]))
            ours, theirs = counts(run, ids), [ref[t] for t in ids]
            lb = leaderboard.get(pid, {})
            rows.append({"model": model, "data_package_id": pid, "n_tasks": len(ids),
                         "asl_ours": round(asl(ours, 10), 3), "asl_package": round(asl(theirs, 10), 3),
                         "asl_leaderboard": lb.get("aslScore"),
                         "pass1_ours": round(pass_at_k(ours, 1) / len(ids), 3),
                         "pass1_package": round(pass_at_k(theirs, 1) / len(ids), 3),
                         "exact_task_agreement": round(sum(x == y for x, y in zip(ours, theirs)) / len(ids), 3),
                         "spearman_tasks": fmt(_spearman(ours, theirs))})
        write_csv(out / "datapackage_check.csv", rows)

    _plot_survival(runs, out / "survival_curves.png")

    print((out / "summary.csv").read_text())
    for name in ("paired_comparison.csv", "pseudocode_leakage.csv", "failure_label_counts.csv",
                 "datapackage_check.csv"):
        if (out / name).exists():
            print(f"== {name}\n{(out / name).read_text()}")
    print(f"Wrote analysis to {out}")


def _group(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def _block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{(text or '').rstrip()}\n```\n"


def _spearman(a, b) -> Optional[float]:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        return None


def _plot_survival(runs, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    styles = {BASELINE: "-", "pseudocode-mbpp-plus": "--", "pseudocode-noassert-mbpp-plus": ":"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {}
    for run in sorted(runs, key=lambda r: (r["model"], r["configuration"])):
        values, m = counts(run), run["max_cycles"]
        color = colors.setdefault(run["model"], f"C{len(colors)}")
        ax.plot(range(1, m + 1), [pass_at_k(values, k) / len(values) for k in range(1, m + 1)],
                styles.get(run["configuration"], "-."), color=color, marker="o", markersize=3,
                label=f"{run['model'].split('/')[-1]} · {run['configuration']}")
    ax.set_xlabel("loop k")
    ax.set_ylabel("fraction of tasks still passing (pass@k / T)")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
