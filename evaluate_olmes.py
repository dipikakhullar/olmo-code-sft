#!/usr/bin/env python
"""
evaluate.py
Run OLMES Python-only coding benchmarks (MBPP, MBPP-Plus, HumanEval,
HumanEval-Plus, LeetCode-Hard) on any Hugging Face model.

➜ Examples
# Full suite, 4 x A100, pass@10
python evaluate.py \
      --model allenai/OLMo-2-0425-70B \
      --gpus 4 \
      --num-samples 10 \
      --output results/70B_full

# Single task, 1 GPU, quick smoke test
python evaluate.py --model checkpoints/step-8000 \
      --benchmarks human_eval --gpus 1
"""
import argparse, json, subprocess, os, sys
from datetime import datetime
from pathlib import Path

PY_TASKS = {
    "mbpp":           "mbpp::evalplus",
    "mbppplus":       "mbppplus::evalplus",
    "human_eval":     "human_eval::olmes",
    "human_evalplus": "human_eval_plus::olmes",
    "leetcode":       "leetcode_hard::olmes",
}
DEFAULT = list(PY_TASKS)

def resolve_tasks(sel):
    if sel == ["all"]:  return [PY_TASKS[k] for k in DEFAULT]
    bad = [s for s in sel if s not in PY_TASKS]
    if bad:  sys.exit(f"Unknown benchmark(s): {bad}")
    return [PY_TASKS[s] for s in sel]



p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF repo or local checkpoint")
p.add_argument("--benchmarks", nargs="+", default=["all"],
               help=f"{list(PY_TASKS)} or 'all'")
p.add_argument("--num-samples", type=int, default=10, help="k for pass@k")
p.add_argument("--temperature", type=float, default=0.2)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--gpus", type=int, default=1,
               help="Number of GPUs to reserve and to pass as "
                    "--tensor-parallel to OLMES")
p.add_argument("--output", default="results", help="Top-level output dir")
args = p.parse_args()
tasks = resolve_tasks(args.benchmarks)

run_dir = Path(args.output) / datetime.now().strftime("%Y%m%dT%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    "olmes",
    "--engine", "vllm",
    "--model", args.model,
    "--tensor-parallel", str(args.gpus),
    "--task", *tasks,
    "--num-samples", str(args.num_samples),
    "--temperature", str(args.temperature),
    "--max-new-tokens", str(args.max_tokens),
    "--output-dir", str(run_dir),
    "--overwrite",
    "--quiet",
]
print("Executing:\n ", " \\\n  ".join(cmd), "\n")
subprocess.run(cmd, check=True)

# quick aggregate
scores = {}
for js in run_dir.glob("*/scores.json"):
    with open(js) as f:
        block = json.load(f)
    task_name = js.parent.name.split("__")[0]
    scores[task_name] = round(block["results"][block["primary_metric"]], 4)
with open(run_dir / "aggregate.json", "w") as f:
    json.dump(scores, f, indent=2)
print("Done. Aggregate results:", scores)
