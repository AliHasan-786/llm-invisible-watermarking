#!/usr/bin/env python3
"""Run the frozen deterministic or model-backed Article 50 attacks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.article50_attacks import (  # noqa: E402
    PARAPHRASE_PROMPT,
    PARAPHRASER_REVISIONS,
    REGENERATE_PROMPT,
    SUMMARY_PROMPT,
    TRANSLATOR,
    attack_key,
    attack_manifest,
    attack_seed,
    attack_specs,
    deterministic_attack_rows,
)
from pipeline.article50 import (  # noqa: E402
    MODEL_REVISIONS,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


def _target_tokenizer(model_alias: str):
    from transformers import AutoTokenizer

    pinned = MODEL_REVISIONS[model_alias]
    return AutoTokenizer.from_pretrained(
        pinned["model_id"],
        revision=pinned["revision"],
    )


def run_deterministic(completions: Path, model_alias: str, output_dir: Path) -> None:
    tokenizer = _target_tokenizer(model_alias)
    special = set(tokenizer.all_special_ids)
    valid_ids = [token for token in range(len(tokenizer)) if token not in special]
    rows = read_jsonl(completions)
    outputs, ineligible = deterministic_attack_rows(
        rows,
        valid_non_special_ids=valid_ids,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "deterministic_attacks.jsonl", outputs)
    write_jsonl_atomic(output_dir / "deterministic_ineligible.jsonl", ineligible)
    write_json_atomic(output_dir / "attack_manifest.json", attack_manifest())
    print(
        json.dumps(
            {"outputs": len(outputs), "ineligible": len(ineligible)},
            indent=2,
        )
    )


class CausalAttackModel:
    def __init__(self, alias: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        pinned = PARAPHRASER_REVISIONS[alias]
        self.alias = alias
        self.tokenizer = AutoTokenizer.from_pretrained(
            pinned["model_id"], revision=pinned["revision"]
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            pinned["model_id"],
            revision=pinned["revision"],
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=generator,
            )
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()


class CausalModelManager:
    """Keep at most one 7B causal attack model resident at a time."""

    def __init__(self):
        self.alias = None
        self.model = None

    def get(self, alias: str) -> CausalAttackModel:
        if self.alias != alias:
            self.clear()
            self.model = CausalAttackModel(alias)
            self.alias = alias
        return self.model

    def clear(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self.alias = None
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass


def _run_paraphrase(
    clean: dict,
    spec,
    models: CausalModelManager,
) -> str:
    passage = str(clean["completion"])
    max_tokens = min(int(clean["n_tokens"]), 250)
    for pass_index, alias in enumerate(spec.parameters["models"]):
        passage = models.get(alias).generate(
            PARAPHRASE_PROMPT.format(passage=passage),
            max_new_tokens=max_tokens,
            seed=attack_seed(str(clean["prompt_id"]), spec.condition, pass_index),
        )
    return passage


def _run_laundering(
    clean: dict,
    spec,
    models: CausalModelManager,
) -> str:
    model = models.get(str(spec.parameters["model"]))
    summary = model.generate(
        SUMMARY_PROMPT.format(passage=clean["completion"]),
        max_new_tokens=80,
        seed=attack_seed(str(clean["prompt_id"]), spec.condition, 0),
    )
    return model.generate(
        REGENERATE_PROMPT.format(
            summary=summary,
            target_tokens=int(clean["n_tokens"]),
        ),
        max_new_tokens=min(int(clean["n_tokens"]), 250),
        seed=attack_seed(str(clean["prompt_id"]), spec.condition, 1),
    )


class BackTranslator:
    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATOR["model_id"], revision=TRANSLATOR["revision"]
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATOR["model_id"],
            revision=TRANSLATOR["revision"],
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def translate(self, text: str, source: str, target: str) -> str:
        import torch

        self.tokenizer.src_lang = source
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target),
                num_beams=4,
                do_sample=False,
                max_new_tokens=512,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def roundtrip(self, text: str, pivot: str) -> str:
        pivot_text = self.translate(text, "eng_Latn", pivot)
        return self.translate(pivot_text, pivot, "eng_Latn")


def run_model_backed(
    completions: Path,
    model_alias: str,
    output_dir: Path,
    families: set[str],
) -> None:
    target_tokenizer = _target_tokenizer(model_alias)
    clean_rows = [row for row in read_jsonl(completions) if row["split"] == "heldout"]
    specs = [
        spec
        for spec in attack_specs()
        if spec.model_backed and spec.family in families
    ]
    models = CausalModelManager()
    translator = None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_attacks.jsonl"
    failure_path = output_dir / "model_attack_failures.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    failures = read_jsonl(failure_path) if failure_path.exists() else []
    completed = {attack_key(row) for row in existing + failures}
    with output_path.open("a", encoding="utf-8") as output, failure_path.open(
        "a", encoding="utf-8"
    ) as failure:
        for spec in specs:
            if spec.family == "backtranslation":
                models.clear()
                if translator is None:
                    translator = BackTranslator()
            elif translator is not None:
                del translator
                translator = None
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
            for clean in clean_rows:
                key = (str(clean["prompt_id"]), str(clean["scheme"]), spec.condition)
                if key in completed:
                    continue
                try:
                    if spec.family == "paraphrase":
                        attacked = _run_paraphrase(clean, spec, models)
                    elif spec.family == "laundering":
                        attacked = _run_laundering(clean, spec, models)
                    elif spec.family == "backtranslation":
                        attacked = translator.roundtrip(
                            str(clean["completion"]), str(spec.parameters["pivot"])
                        )
                    else:
                        raise AssertionError(spec.family)
                    token_ids = target_tokenizer.encode(attacked, add_special_tokens=False)
                    row = {
                        "prompt_id": clean["prompt_id"],
                        "source": clean["source"],
                        "scheme": clean["scheme"],
                        "split": "heldout",
                        "condition": spec.condition,
                        "attack_family": spec.family,
                        "attack_parameters": spec.parameters,
                        "attack_seed": attack_seed(
                            str(clean["prompt_id"]), spec.condition
                        ),
                        "completion": attacked,
                        "token_ids": token_ids,
                        "n_tokens": len(token_ids),
                        "source_n_tokens": int(clean["n_tokens"]),
                        "length_retention": len(token_ids) / max(1, int(clean["n_tokens"])),
                        "tokenizer_vocab_size": len(target_tokenizer),
                        "eos_token_id": target_tokenizer.eos_token_id,
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                except Exception as error:
                    row = {
                        "prompt_id": clean["prompt_id"],
                        "scheme": clean["scheme"],
                        "condition": spec.condition,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    failure.write(json.dumps(row, sort_keys=True) + "\n")
                    failure.flush()
    models.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("completions", type=Path)
    parser.add_argument("--model", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("deterministic", "model"), default="deterministic"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("paraphrase", "backtranslation", "laundering"),
        default=("paraphrase", "backtranslation", "laundering"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.completions.parent / "attacks"
    if args.mode == "deterministic":
        run_deterministic(args.completions, args.model, output_dir)
    else:
        run_model_backed(
            args.completions,
            args.model,
            output_dir,
            set(args.families),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
