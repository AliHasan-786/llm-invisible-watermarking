"""
Corpus generation pipeline.

Loads prompts from CNN/DailyMail, WritingPrompts, and TriviaQA,
then generates both watermarked and unwatermarked completions
from a target model, saving results for downstream evaluation.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermark.logits_processor import WatermarkLogitsProcessor


DATASET_CONFIGS = {
    "cnn_dailymail": {
        "path": "cnn_dailymail",
        "name": "3.0.0",
        "split": "test",
        "prompt_field": "article",
        "prompt_prefix": "Summarize the following article:\n\n",
        "max_prompt_tokens": 200,
    },
    "writing_prompts": {
        "path": "euclaise/writingprompts",
        "name": None,
        "split": "test",
        "prompt_field": "prompt",
        "prompt_prefix": "Continue the following story prompt:\n\n",
        "max_prompt_tokens": 60,
    },
    "trivia_qa": {
        "path": "trivia_qa",
        "name": "rc",
        "split": "validation",
        "prompt_field": "question",
        "prompt_prefix": "Answer the following question in detail:\n\n",
        "max_prompt_tokens": 50,
    },
}


class CorpusGenerator:
    def __init__(
        self,
        model_name: str,
        delta: float = 2.0,
        gamma: float = 0.5,
        seed: int = 42,
        max_new_tokens: int = 200,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.delta = delta
        self.gamma = gamma
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer and model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        self.watermark_processor = WatermarkLogitsProcessor(
            vocab_size=len(self.tokenizer),
            delta=self.delta,
            gamma=self.gamma,
            seed=self.seed,
        )

    def _load_prompts(self, n_per_dataset: int = 150) -> list[dict]:
        prompts = []
        for dataset_name, cfg in DATASET_CONFIGS.items():
            print(f"Loading {dataset_name}...")
            try:
                ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], trust_remote_code=True)
                indices = random.sample(range(len(ds)), min(n_per_dataset * 3, len(ds)))
                for idx in indices:
                    row = ds[idx]
                    text = cfg["prompt_prefix"] + str(row[cfg["prompt_field"]])
                    tokens = self.tokenizer.encode(text, truncation=True, max_length=cfg["max_prompt_tokens"])
                    prompt_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    prompts.append({"source": dataset_name, "prompt": prompt_text})
                    if len([p for p in prompts if p["source"] == dataset_name]) >= n_per_dataset:
                        break
            except Exception as e:
                print(f"  Warning: could not load {dataset_name}: {e}")
        random.shuffle(prompts)
        return prompts

    def _generate(self, prompt: str, watermark: bool) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if watermark:
            gen_kwargs["logits_processor"] = [self.watermark_processor]

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_len:].tolist()
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return {
            "prompt": prompt,
            "completion": completion_text,
            "token_ids": completion_ids,
            "n_tokens": len(completion_ids),
            "watermarked": watermark,
        }

    def generate_corpus(
        self,
        n_per_dataset: int = 150,
        output_path: str = "corpus.jsonl",
        resume: bool = True,
    ) -> list[dict]:
        """
        Generate watermarked and unwatermarked completions and save to JSONL.

        Args:
            n_per_dataset: Number of prompts to sample per dataset.
            output_path: Path to output JSONL file.
            resume: If True and the file exists, skip already-completed (source, prompt, watermarked) triples.
        Returns:
            List of all result dicts (existing + newly generated).
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume
        existing: list[dict] = []
        completed_keys: set[tuple] = set()
        if resume and output_file.exists():
            with output_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        existing.append(item)
                        completed_keys.add((item["source"], item["prompt"], item["watermarked"]))
            print(f"Resuming: found {len(existing)} existing samples, {len(completed_keys)} completed (source, prompt, wm) pairs.")

        prompts = self._load_prompts(n_per_dataset)
        total = len(prompts) * 2
        skipped = 0
        generated = 0
        start_time = time.time()

        file_mode = "a" if (resume and output_file.exists()) else "w"
        results = list(existing)

        with output_file.open(file_mode) as f:
            for i, item in enumerate(prompts):
                for wm in [True, False]:
                    key = (item["source"], item["prompt"], wm)
                    if key in completed_keys:
                        skipped += 1
                        continue

                    try:
                        result = self._generate(item["prompt"], watermark=wm)
                        result["source"] = item["source"]
                        result["model"] = self.model_name
                        result["delta"] = self.delta
                        result["gamma"] = self.gamma
                        result["seed"] = self.seed
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        results.append(result)
                        generated += 1

                        # Progress log every 10 new generations
                        if generated % 10 == 0:
                            elapsed = time.time() - start_time
                            remaining = total - skipped - generated
                            eta = (elapsed / generated * remaining) if generated > 0 else 0
                            mem_str = ""
                            if torch.cuda.is_available():
                                mem_gb = torch.cuda.memory_allocated() / 1e9
                                mem_str = f"  GPU mem: {mem_gb:.1f}GB"
                            print(
                                f"  [{generated} new / {skipped} skipped / {remaining} remaining]"
                                f"  ETA: {eta/60:.1f}min{mem_str}"
                            )
                    except Exception as e:
                        print(f"\n  Error on prompt {i} (wm={wm}): {e}")

        print(f"\nDone. {generated} new samples generated, {skipped} skipped. Total in file: {len(results)}.")
        return results


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
