#!/usr/bin/env python3
"""Prepare and run the frozen Article 50 watermark experiment.

The default workflow is intentionally fail-closed:

  python scripts/run_article50.py preflight --model gemma
  python scripts/run_article50.py freeze-prompts --model gemma
  python scripts/run_article50.py plan --prompts results/article50/prompts.jsonl
  python scripts/run_article50.py generate --model gemma --pilot-prompts 20

Pilot outputs live under ``results/article50/pilot/`` and are never read by
the confirmatory analysis path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.article50 import (  # noqa: E402
    GENERATION_CONFIG,
    MODEL_REVISIONS,
    build_generation_plan,
    create_run_manifest,
    generation_key,
    manifest_dict,
    prompt_id,
    read_jsonl,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


DATASET_CONFIGS = {
    "cnn_dailymail": {
        "path": "cnn_dailymail",
        "name": "3.0.0",
        "split": "test",
        "field": "article",
        "prefix": "Summarize the following article:\n\n",
        "max_prompt_tokens": 200,
    },
    "writing_prompts": {
        "path": "euclaise/writingprompts",
        "name": None,
        "split": "test",
        "field": "prompt",
        "prefix": "Continue the following story prompt:\n\n",
        "max_prompt_tokens": 60,
    },
    "trivia_qa": {
        "path": "trivia_qa",
        "name": "rc",
        "split": "validation",
        "field": "question",
        "prefix": "Answer the following question in detail:\n\n",
        "max_prompt_tokens": 50,
    },
}


def _hf_identity() -> tuple[bool, str]:
    for command in (["hf", "auth", "whoami"], ["huggingface-cli", "whoami"]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and "not logged in" not in text.lower():
            return True, text
    return False, "not logged in to Hugging Face"


def preflight(model_alias: str, *, require_gpu: bool = True) -> int:
    import torch

    authenticated, identity = _hf_identity()
    has_cuda = torch.cuda.is_available()
    pinned = MODEL_REVISIONS[model_alias]
    report = {
        "model": pinned,
        "hugging_face_authenticated": authenticated,
        "hugging_face_identity": identity,
        "cuda_available": has_cuda,
        "cuda_device": torch.cuda.get_device_name(0) if has_cuda else None,
        "status": "ready" if authenticated and (has_cuda or not require_gpu) else "blocked",
    }
    print(json.dumps(report, indent=2))
    if not authenticated:
        print(
            "\nBLOCKED: the pinned model repository is gated. Credential entry is "
            "an explicit personal gate; no login was attempted.",
            file=sys.stderr,
        )
    if require_gpu and not has_cuda:
        print(
            "\nBLOCKED: no CUDA GPU is available. Use the same command in a no-cost "
            "Colab/Kaggle GPU runtime.",
            file=sys.stderr,
        )
    return 0 if report["status"] == "ready" else 2


def freeze_prompts(model_alias: str, output: Path, n_per_source: int) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    pinned = MODEL_REVISIONS[model_alias]
    tokenizer = AutoTokenizer.from_pretrained(
        pinned["model_id"],
        revision=pinned["revision"],
    )
    frozen = []
    exclusions = {}
    for source, config in DATASET_CONFIGS.items():
        dataset = load_dataset(
            config["path"],
            config["name"],
            split=config["split"],
        )
        eligible = []
        counters = {"empty": 0, "tokenization_failure": 0, "too_long": 0, "duplicate": 0}
        seen_text = set()
        for source_index, row in enumerate(dataset):
            raw = row.get(config["field"])
            if raw is None or not str(raw).strip():
                counters["empty"] += 1
                continue
            prompt = config["prefix"] + str(raw).strip()
            if prompt in seen_text:
                counters["duplicate"] += 1
                continue
            try:
                token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            except Exception:
                counters["tokenization_failure"] += 1
                continue
            if len(token_ids) > config["max_prompt_tokens"]:
                counters["too_long"] += 1
                continue
            seen_text.add(prompt)
            identifier = prompt_id(source, source_index, prompt)
            eligible.append(
                {
                    "prompt_id": identifier,
                    "source": source,
                    "source_index": source_index,
                    "prompt": prompt,
                    "prompt_tokens": len(token_ids),
                    "selection_hash": sha256_text(f"sample:42:{identifier}"),
                }
            )
        chosen = sorted(eligible, key=lambda item: item["selection_hash"])[:n_per_source]
        if len(chosen) != n_per_source:
            raise RuntimeError(
                f"{source} yielded {len(chosen)} eligible prompts; expected {n_per_source}"
            )
        frozen.extend(chosen)
        exclusions[source] = {
            **counters,
            "eligible": len(eligible),
            "selected": len(chosen),
        }
    frozen.sort(key=lambda item: (item["source"], item["selection_hash"]))
    write_jsonl_atomic(output, frozen)
    write_json_atomic(
        output.with_suffix(".metadata.json"),
        {
            "model_id": pinned["model_id"],
            "tokenizer_revision": pinned["revision"],
            "n_per_source": n_per_source,
            "sources": DATASET_CONFIGS,
            "exclusions": exclusions,
            "prompt_file_sha256": sha256_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in frozen)
            ),
        },
    )
    print(f"Frozen {len(frozen)} prompts at {output}")


def _pilot_subset(prompts: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("pilot prompt count must be positive")
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in prompts:
        by_source.setdefault(str(row["source"]), []).append(row)
    sources = sorted(by_source)
    selected = []
    for index in range(count):
        source = sources[index % len(sources)]
        source_rows = sorted(by_source[source], key=lambda row: str(row["selection_hash"]))
        offset = index // len(sources)
        if offset >= len(source_rows):
            raise ValueError("not enough frozen prompts for requested pilot")
        selected.append(source_rows[offset])
    return selected


def write_plan(prompts_path: Path, output: Path, pilot_prompts: int | None) -> list[dict]:
    prompts = read_jsonl(prompts_path)
    if pilot_prompts is not None:
        prompts = _pilot_subset(prompts, pilot_prompts)
    plan = build_generation_plan(prompts)
    write_jsonl_atomic(output, plan)
    counts: dict[str, int] = {}
    for row in plan:
        key = f"{row['split']}:{row['scheme']}"
        counts[key] = counts.get(key, 0) + 1
    write_json_atomic(output.with_suffix(".summary.json"), {"counts": counts})
    print(json.dumps({"plan": str(output), "counts": counts}, indent=2))
    return plan


def _load_existing(path: Path) -> tuple[list[dict], set[tuple]]:
    if not path.exists():
        return [], set()
    rows = read_jsonl(path)
    return rows, {generation_key(row) for row in rows}


def generate(model_alias: str, prompts_path: Path, output_dir: Path, pilot_prompts: int | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from watermark.logits_processor import WatermarkLogitsProcessor
    from watermark.synthid import generation_config as synthid_generation_config

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "generation_plan.jsonl"
    plan = write_plan(prompts_path, plan_path, pilot_prompts)
    manifest = create_run_manifest(REPO, model_alias, pilot=pilot_prompts is not None)
    write_json_atomic(output_dir / "run_manifest.json", manifest_dict(manifest))

    pinned = MODEL_REVISIONS[model_alias]
    tokenizer = AutoTokenizer.from_pretrained(
        pinned["model_id"],
        revision=pinned["revision"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        pinned["model_id"],
        revision=pinned["revision"],
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    model_device = next(model.parameters()).device
    kirchenbauer = WatermarkLogitsProcessor(
        vocab_size=len(tokenizer),
        delta=2.0,
        gamma=0.5,
        seed=42,
    )

    outputs_path = output_dir / "completions.jsonl"
    failures_path = output_dir / "failures.jsonl"
    _, completed = _load_existing(outputs_path)
    _, failed = _load_existing(failures_path)
    completed.update(failed)
    with outputs_path.open("a", encoding="utf-8") as outputs, failures_path.open(
        "a", encoding="utf-8"
    ) as failures:
        for index, item in enumerate(plan, start=1):
            key = generation_key(item)
            if key in completed:
                continue
            try:
                inputs = tokenizer(
                    item["prompt"],
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(model_device)
                generator = torch.Generator(device=model_device).manual_seed(int(item["seed"]))
                kwargs = {
                    **inputs,
                    **GENERATION_CONFIG,
                    "generator": generator,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if item["scheme"] == "kirchenbauer":
                    kwargs["logits_processor"] = [kirchenbauer]
                elif item["scheme"] == "synthid":
                    kwargs["watermarking_config"] = synthid_generation_config()
                with torch.inference_mode():
                    generated = model.generate(**kwargs)
                prompt_length = inputs["input_ids"].shape[1]
                completion_ids = generated[0, prompt_length:].detach().cpu().tolist()
                result = {
                    **item,
                    "model_id": pinned["model_id"],
                    "model_revision": pinned["revision"],
                    "tokenizer_vocab_size": len(tokenizer),
                    "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                    "token_ids": completion_ids,
                    "n_tokens": len(completion_ids),
                }
                outputs.write(json.dumps(result, sort_keys=True) + "\n")
                outputs.flush()
                completed.add(key)
            except Exception as error:
                failure = {
                    **item,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                failures.write(json.dumps(failure, sort_keys=True) + "\n")
                failures.flush()
            if index % 10 == 0:
                print(f"Processed {index}/{len(plan)} planned generations")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--model", choices=MODEL_REVISIONS, default="gemma")
    preflight_parser.add_argument("--allow-cpu", action="store_true")

    freeze_parser = subparsers.add_parser("freeze-prompts")
    freeze_parser.add_argument("--model", choices=MODEL_REVISIONS, default="gemma")
    freeze_parser.add_argument(
        "--output", type=Path, default=REPO / "results/article50/prompts.jsonl"
    )
    freeze_parser.add_argument("--n-per-source", type=int, default=150)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument(
        "--prompts", type=Path, default=REPO / "results/article50/prompts.jsonl"
    )
    plan_parser.add_argument(
        "--output", type=Path, default=REPO / "results/article50/generation_plan.jsonl"
    )
    plan_parser.add_argument("--pilot-prompts", type=int)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--model", choices=MODEL_REVISIONS, default="gemma")
    generate_parser.add_argument(
        "--prompts", type=Path, default=REPO / "results/article50/prompts.jsonl"
    )
    generate_parser.add_argument("--pilot-prompts", type=int)
    generate_parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return preflight(args.model, require_gpu=not args.allow_cpu)
    if args.command == "freeze-prompts":
        freeze_prompts(args.model, args.output, args.n_per_source)
        return 0
    if args.command == "plan":
        write_plan(args.prompts, args.output, args.pilot_prompts)
        return 0
    if args.command == "generate":
        pilot = args.pilot_prompts is not None
        output_dir = args.output_dir or REPO / "results/article50" / (
            f"pilot/{args.model}" if pilot else f"confirmatory/{args.model}"
        )
        generate(args.model, args.prompts, output_dir, args.pilot_prompts)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
