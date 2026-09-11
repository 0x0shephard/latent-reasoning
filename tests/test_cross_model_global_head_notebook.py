from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_model_notebook_has_complete_preregistered_protocol():
    path = ROOT / "notebooks" / "kaggle_cross_model_global_head_benchmark.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
    assert payload["nbformat"] == 4
    for required in (
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-2-2b-it",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", "MODEL_KEY",
        "SEED_INDEX", "PAPER_SEEDS", "three_seed_protocol_complete",
        "wikitext-2-raw-v1", "openai/gsm8k", "HuggingFaceH4/MATH-500",
        "ARC-Challenge", "google-research-datasets/mbpp", "cnn_dailymail",
        '"facebook/xnli", "ar"', "weight_svd_frozen", "asvd_style_frozen",
        "svdllm_style_whitened_frozen", "random_full_distillation",
        "learned_no_whitening", "whitened_clean_no_onpolicy",
        "slimspec_style_random_kl", "full_whitened_distilled_onpolicy",
        "dense_perplexity", "top1_agreement", "topk_overlap_fraction",
        "teacher_kl", "mean_absolute_margin_error", "rare_teacher_top1_agreement",
        "exact_sequence_agreement", "common_prefix_fraction", "task_score",
        "benchmark_head_latency", "speedup_over_dense", "tied_embeddings",
        "aggregate_manifest.json", "official_codi_gpt2_anchor",
        "reference_implementation_boundary", '"checkout", "--detach"',
    ):
        assert required in source
