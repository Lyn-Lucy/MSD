"""Distribute hidden-state data generation across explicitly selected GPUs."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model_type", required=True, choices=("llava_v15_v", "llava_v15_t", "qwen2_vl_v", "qwen2_vl_t"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--image_data_path", required=True)
    parser.add_argument("--json_data_path", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Exclusive end index; defaults to the annotation count.")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must contain distinct non-negative GPU indices")
    with open(args.json_data_path) as handle:
        count = len(json.load(handle))
    end = count if args.end is None else args.end
    if not 0 <= args.start < end <= count:
        parser.error("--start and --end must specify a nonempty range inside the annotations")
    script = "ge_data_all_llava15.py" if args.model_type.startswith("llava") else "ge_data_all_qwen2_5vl.py"
    script_path = Path(__file__).resolve().parent / script
    output = Path(args.outdir) / f"{args.model_type}_{args.start}_{end - 1}"
    output.mkdir(parents=True, exist_ok=True)
    workers = min(len(args.gpus), end - args.start)
    commands = []
    for index, gpu in enumerate(args.gpus[:workers]):
        start = args.start + (end - args.start) * index // workers
        stop = args.start + (end - args.start) * (index + 1) // workers
        commands.append([
            sys.executable, str(script_path), "--start", str(start), "--end", str(stop),
            "--index", str(index), "--gpu_index", str(gpu), "--outdir", str(output),
            "--model", args.model, "--image_data_path", args.image_data_path,
            "--json_data_path", args.json_data_path,
        ])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in commands]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
