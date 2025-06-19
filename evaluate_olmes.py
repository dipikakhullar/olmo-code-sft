from __future__ import annotations
import argparse, json, os, subprocess, sys, textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------- 1. CLI & benchmark list ---------------- #
TASKS = {
    "mbpp": "mbpp",
    "mbppplus": "mbppplus",
    "human_eval": "human_eval",
    "human_evalplus": "human_eval_plus",
    "leetcode": "leetcode_hard",
}
ORDER = list(TASKS)

def pick(lst: List[str]) -> List[str]:
    return ORDER if lst == ["all"] else lst

cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--model", required=True)
cli.add_argument("--benchmarks", nargs="+", default=["all"])
cli.add_argument("--backend", choices=["hf", "vllm"], default="hf")
cli.add_argument("--gpus", type=int, default=1)
cli.add_argument("--num-samples", type=int, default=10, help="k for pass@k")
cli.add_argument("--temperature", type=float, default=0.8)
cli.add_argument("--top-p", type=float, default=0.95)
cli.add_argument("--max-tokens", type=int, default=256)
cli.add_argument("--output", default="results")
args = cli.parse_args()
TODO = pick(args.benchmarks)

# -------------- 2. Run folder & HF cache --------------- #
run_root = Path(args.output) / datetime.now().strftime("%Y%m%dT%H%M%S")
run_root.mkdir(parents=True, exist_ok=True)
hf_cache = run_root / "hf_cache"
hf_cache.mkdir()
os.environ["HF_HOME"] = os.environ["HF_HUB_CACHE"] = str(hf_cache)
print(f"[+] Outputs → {run_root}")

from huggingface_hub import snapshot_download           # after env set

# -------------- 3. Ensure model is local --------------- #
model_path = Path(args.model)
if not model_path.exists():
    model_path = run_root / "model"
    snapshot_download(args.model, local_dir=model_path, resume_download=True)
model_path = model_path.resolve()

# -------- 4. Generation & task-config helpers ---------- #
k = max(1, args.num_samples)
GEN: Dict[str, Any] = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_new_tokens": args.max_tokens,
    "repeats": k,
    "do_sample": True,        # always sample – needed for real pass@k
}

MODEL_ARGS: Dict[str, Any] = {"dtype": "float16", "trust_remote_code": True}

# removed "```" to prevent instant termination on tiny models
STOP = {
    "mbpp": ['\n"""', "\nassert", "\n#"],
    "mbppplus": ['\n"""', "\nassert", "\n#"],
    "human_eval": ['\n"""', "\ndef", "\nassert", "\nclass"],
    "human_evalplus": ['\n"""', "\ndef", "\nassert", "\nclass"],
    "leetcode": ["\nclass", "\n#"],
}

def task_json(key: str) -> str:
    return json.dumps({
        "task_name": TASKS[key],
        "primary_metric": "pass_at_1",
        "use_chat_format": False,
        "context_kwargs": {},
        "generation_kwargs": GEN | {"stop_sequences": STOP[key]},
        "metadata": {"alias": f"{TASKS[key]}::olmes"},
    })

def run_task(key: str, backend: str) -> subprocess.CompletedProcess[str]:
    margs = MODEL_ARGS | ({"tensor_parallel_size": args.gpus} if backend == "vllm" else {})
    cmd = [
        "python", "-m", "oe_eval.run_eval",
        "--model", str(model_path),
        "--model-type", backend,
        "--model-args", json.dumps(margs),
        "--task", task_json(key),
        "--output-dir", str(run_root),
        "--gpus", str(args.gpus),
        "--save-raw-requests", "true",
        "--num-workers", "1",
    ]
    print(textwrap.indent(" \\\n".join(cmd), prefix="  "))
    print()
    return subprocess.run(cmd, text=True, capture_output=True)

# ---------------- 5. Execute with fallback ------------- #
for key in TODO:
    print(f"\n===== {key.upper()} =====")
    proc = run_task(key, args.backend)
    if proc.returncode == 0:
        continue
    # fallback for models unsupported by vLLM
    if args.backend == "vllm" and "Unrecognized model" in proc.stderr:
        print("[!] vLLM failed → falling back to HF.")
        proc = run_task(key, "hf")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"[x] {key} failed.")

# ---------------- 6. Aggregate metrics ---------------- #
scores: Dict[str, float] = {}

for jf in run_root.glob("*-metrics.json"):
    try:
        blk = json.loads(jf.read_text())
        metrics_dict = blk.get("metrics") or blk.get("results")  # OE-Eval ≥0.3 vs legacy
        if not metrics_dict:
            continue
        primary = blk.get("primary_metric") or blk.get("task_config", {}).get("primary_metric")
        if primary is None or primary not in metrics_dict:
            continue
        task_name = blk.get("task_name") or jf.stem.split("-")[2]
        scores[task_name] = round(float(metrics_dict[primary]), 4)
    except (KeyError, json.JSONDecodeError):
        # skip malformed or error files
        continue

(run_root / "aggregate.json").write_text(json.dumps(scores, indent=2))
print("\n[✓] Aggregate results:")
print(json.dumps(scores, indent=2))
print(f"[✓] All files in {run_root}")
