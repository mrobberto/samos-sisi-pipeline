#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Union, Sequence

import config


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def run_script(stage: Union[str, tuple[str, Sequence[str]]]) -> None:
    if isinstance(stage, tuple):
        relpath, extra_args = stage
        extra_args = list(extra_args)
    else:
        relpath = stage
        extra_args = []

    script = PIPELINE_ROOT / relpath
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    cmd = [sys.executable, str(script)] + extra_args
    print("\n[CMD]", " ".join(cmd), flush=True)

    result = subprocess.run(cmd, cwd=PIPELINE_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {relpath}")


def select_stages(stages, from_step=None, to_step=None, only=None):
    keys = list(stages.keys())

    if only:
        if only not in stages:
            raise ValueError(f"Unknown stage: {only}. Valid: {keys}")
        return [only]

    from_step = from_step or keys[0]
    to_step = to_step or keys[-1]

    if from_step not in stages:
        raise ValueError(f"Unknown --from-step {from_step}. Valid: {keys}")
    if to_step not in stages:
        raise ValueError(f"Unknown --to-step {to_step}. Valid: {keys}")

    i0 = keys.index(from_step)
    i1 = keys.index(to_step)

    if i1 < i0:
        raise ValueError("--to-step must be after --from-step")

    return keys[i0:i1 + 1]


def matching_qc_keys(stage, qc_stages):
    keys = []

    if stage in qc_stages:
        keys.append(stage)

    # Allows QC_STAGES["07"] to run after stage "07c"
    short = stage[:2]
    if short in qc_stages and short not in keys:
        keys.append(short)

    return keys


def main(STAGES, QC_STAGES):
    parser = argparse.ArgumentParser(description="Run a SISI reduction.")
    parser.add_argument("--from-step", default=None)
    parser.add_argument("--to-step", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--run-qc", action="store_true")
    args = parser.parse_args()

    if hasattr(config, "ensure_dirs"):
        config.ensure_dirs()
    elif hasattr(config, "build_sisi_paths"):
        config.build_sisi_paths()

    if hasattr(config, "print_config"):
        config.print_config()
    else:
        print("Active SISI reduction:")
        print(f"  TARGET       = {getattr(config, 'SISI_TARGET', 'UNKNOWN')}")
        print(f"  FIELD        = {getattr(config, 'SISI_FIELD', 'UNKNOWN')}")
        print(f"  SISI_NIGHT   = {getattr(config, 'SISI_NIGHT', 'UNKNOWN')}")
        print(f"  SISI_REDUCED = {getattr(config, 'SISI_REDUCED', 'UNKNOWN')}")

    selected = select_stages(
        STAGES,
        from_step=args.from_step,
        to_step=args.to_step,
        only=args.only,
    )

    print("\nPlanned SISI stages:")
    for s in selected:
        print(f"  {s}: {STAGES[s]}")

    for s in selected:
        run_script(STAGES[s])

    if args.run_qc:
        print("\nRunning QC:")
        for s in selected:
            for qkey in matching_qc_keys(s, QC_STAGES):
                for relpath in as_list(QC_STAGES[qkey]):
                    run_script(relpath)

    print("\n=== SISI pipeline run complete ===")


if __name__ == "__main__":
    raise SystemExit(
        "Use a dataset-specific runner, e.g. run_sisi_pipeline_T00.py or T01.py"
    )