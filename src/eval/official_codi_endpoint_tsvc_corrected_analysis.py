"""Combine and render corrected source-faithful CODI endpoint TSV-C gates."""
from __future__ import annotations

from typing import Mapping

from src.eval.official_codi_endpoint_tsvc_analysis import PRIMARY_CONTROLS


CORRECTED_ENDPOINT_ANALYSIS_SCHEMA_VERSION = 2


def combine_corrected_endpoint_reports(
    all_states: Mapping,
    layer11: Mapping,
) -> dict:
    if all_states.get("scope") != "endpoint_all_states":
        raise ValueError("corrected primary report must be endpoint_all_states")
    if layer11.get("scope") != "endpoint_layer11":
        raise ValueError("corrected secondary report must be endpoint_layer11")
    primary_request = all_states.get("request", {})
    secondary_request = layer11.get("request", {})
    identity_fields = (
        "contract",
        "checkpoint_sha256",
        "dataset_fingerprint",
        "basis_sha256",
        "basis_request_sha256",
        "rank",
        "update_indices",
        "validation_indices",
        "batch_size",
        "relative_update_norm",
    )
    if any(primary_request.get(field) is None for field in identity_fields):
        raise ValueError("corrected summaries are missing matched request identity")
    if any(
        primary_request.get(field) != secondary_request.get(field)
        for field in identity_fields
    ):
        raise ValueError("corrected endpoint scopes are not matched")
    if primary_request.get("native_parity_gate", {}).get("status") != "passed":
        raise ValueError("corrected report lacks a passing native parity gate")

    primary = bool(all_states["gate"]["supported"])
    secondary = bool(layer11["gate"]["supported"])
    if primary:
        status = "corrected_endpoint_tsvc_training_authorized"
    elif secondary:
        status = "corrected_layer11_requires_fresh_confirmation"
    else:
        status = "corrected_endpoint_tsvc_not_supported"
    return {
        "schema_version": CORRECTED_ENDPOINT_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_tsvc_corrected_utility_gate",
        "status": status,
        "training_authorized": primary,
        "primary_scope": "endpoint_all_states",
        "secondary_scope": "endpoint_layer11",
        "native_parity_gate": primary_request["native_parity_gate"],
        "scopes": {
            "endpoint_all_states": dict(all_states),
            "endpoint_layer11": dict(layer11),
        },
        "interpretation_boundary": (
            "This is a local, one-step, equal-update-norm test at the final official "
            "CODI checkpoint after exact source-level endpoint-loss and gradient parity. "
            "A positive primary gate authorizes a separately preregistered training "
            "experiment; it does not itself establish long-run accuracy improvement."
        ),
    }


def render_corrected_endpoint_markdown(report: Mapping) -> str:
    parity = report["native_parity_gate"]
    lines = [
        "# Corrected official CODI answer-cue endpoint TSV-C utility gate",
        "",
        "## Outcome",
        "",
        f"Predefined decision: **{str(report['status']).replace('_', ' ')}**.",
        "",
        "The teacher and student are both measured at the colon in `The answer is:`.",
        "The student colon occurs after six continuous latents and EOT.",
        "",
        "## Native parity gate",
        "",
        f"Status: **{parity['status']}**.",
        "",
        f"Loss absolute error: {parity['absolute_loss_error']:.3e}.",
        f"Gradient relative L2 error: {parity['gradient_relative_l2_error']:.3e}.",
        f"Gradient cosine: {parity['gradient_cosine']:.9f}.",
    ]
    for scope_name in ("endpoint_all_states", "endpoint_layer11"):
        scope = report["scopes"][scope_name]
        lines.extend(
            [
                "",
                f"## {scope_name}",
                "",
                f"Gate: **{scope['gate']['status'].replace('_', ' ')}**.",
                "",
                "| Learned top-77 versus | Mean advantage | 95% CI | Holm p | Pass |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name in PRIMARY_CONTROLS:
            value = scope["learned_top77_comparisons"][name]
            low, high = value["bootstrap_95ci"]
            lines.append(
                f"| {name} | {value['mean_advantage']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{value['holm_adjusted_p']:.6g} | "
                f"{'yes' if value['passes'] else 'no'} |"
            )
        full = scope["learned_top77_vs_full"]
        low, high = full["bootstrap_95ci"]
        lines.extend(
            [
                "",
                (
                    "Learned top-77 minus full-target utility: "
                    f"{full['mean_advantage']:+.6f} "
                    f"(95% CI [{low:+.6f}, {high:+.6f}])."
                ),
                (
                    "Median learned-subspace answer-gradient cosine: "
                    f"{scope['gradient_alignment']['median_cosine']:+.6f}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Rank: 77 of 768 hidden dimensions",
            "- Primary: embedding state plus all 12 transformer-block states at the colon",
            "- Secondary: transformer block 11 at the same colon",
            "- Loss: released SmoothL1 divided by unbiased teacher standard deviation",
            "- Bootstrap and randomization unit: paired update batch",
            "- Every auxiliary gradient is matched to the native full-endpoint gradient norm",
            "- Every total parameter update has the same L2 norm",
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)

