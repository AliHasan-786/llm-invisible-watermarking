"""
Corpus generation pipeline.

Loads prompts from CNN/DailyMail, WritingPrompts, and TriviaQA,
then generates both watermarked and unwatermarked completions
from a target model, saving results for downstream evaluation.
"""

import json
import random
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
                # shuffle and sample
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
    ) -> list[dict]:
        """
        Generate watermarked and unwatermarked completions and save to JSONL.

        Returns list of result dicts.
        """
        prompts = self._load_prompts(n_per_dataset)
        print(f"Generating from {len(prompts)} prompts (watermarked + control)...")

        results = []
        output_file = Path(output_path)

        with output_file.open("w") as f:
            for i, item in enumerate(prompts):
                print(f"  [{i+1}/{len(prompts)}] {item['source']}", end="\r")
                for wm in [True, False]:
                    try:
                        result = self._generate(item["prompt"], watermark=wm)
                        result["source"] = item["source"]
                        result["model"] = self.model_name
                        result["delta"] = self.delta
                        result["gamma"] = self.gamma
                        result["seed"] = self.seed
                        f.write(json.dumps(result) + "\n")
                        results.append(result)
                    except Exception as e:
                        print(f"\n  Error on prompt {i}: {e}")

        print(f"\nSaved {len(results)} samples to {output_path}")
        return results


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
