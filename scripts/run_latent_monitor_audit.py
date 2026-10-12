"""Latent Monitor Audit driver (ledger §123).  No training.

One command, one card.  Runs the twin audit on both profiles (§110/§112), the paired
bootstrap, the masking audit (§120 B) with carrying slots derived from the long twins,
and the monitor-faithfulness analysis (§120 A), then assembles the standardised card
(``src/mech/monitor_audit.py``) with the environment fingerprint and the predictions
fingerprints.  ``--from-existing`` assembles a card from runs already on disk (the §120
Kaggle layout ``<prefix>_twins_standard`` etc. is recognised) without touching a GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.chain_pools import DEFAULT_POOL, POOLS
from src.mech.monitor_audit import (
    CARD_CONTRACT,
    build_card,
    card_markdown,
    carrying_slots,
    environment_fingerprint,
    predictions_fingerprint,
)

STAGES = ("twins_standard", "twins_long", "masking", "bootstrap", "monitor", "card")
TWINS_SCRIPT = "scripts/run_codi_directed_counterfactual.py"
MASKING_SCRIPT = "scripts/run_codi_masking_audit.py"


def layout(root: Path) -> dict:
    return {"twins_standard": root / "twins_standard", "twins_long": root / "twins_long", "masking": root / "masking",
            "bootstrap": root / "bootstrap.json", "monitor": root / "monitor_faithfulness.json",
            "card": root / "card.json", "card_md": root / "card.md", "stage_seconds": root / "stage_seconds.json", "logs": root / "logs"}


def locate_existing(root: Path, prefix: str | None = None) -> dict:
    """Find runs under ``root`` in the driver layout or the §120 Kaggle layout (``<prefix>_<stage>``)."""
    root = Path(root)
    out = layout(root)

    def first(candidates, is_dir):
        for c in candidates:
            if c is not None and c.exists() and c.is_dir() == is_dir:
                return c
        return None

    for stage in ("twins_standard", "twins_long", "masking"):
        globbed = sorted(p for p in root.glob(f"*_{stage}") if p.is_dir() and not p.name.endswith("_smoke"))
        out[stage] = first([root / stage, root / f"{prefix}_{stage}" if prefix else None] + globbed, True)
    for key, suffix in (("bootstrap", "bootstrap.json"), ("monitor", "monitor_faithfulness.json")):
        globbed = sorted(root.glob(f"*_{suffix}"))
        out[key] = first([root / suffix, root / f"{prefix}_{suffix}" if prefix else None] + globbed, False)
    return out


def plan(pool: str, long: str = "auto", stages=STAGES) -> list[str]:
    chosen = [s for s in STAGES if s in set(stages)]
    if long == "off" or (long == "auto" and not POOLS[pool]["long_profile"]):
        chosen = [s for s in chosen if s != "twins_long"]
    return chosen


def twins_command(args, profile: str, out: Path, *, smoke: bool) -> list[str]:
    cmd = [sys.executable, "-u", str(ROOT / TWINS_SCRIPT), "--config", str(args.config), "--reproduction-summary", str(args.reproduction_summary),
           "--output-dir", str(out), "--batch-size", str(args.batch_size), "--precision", str(args.precision), "--device", str(args.device),
           "--profile", profile, "--pool", str(args.pool)]
    if getattr(args, "checkpoint_path", None):
        cmd += ["--checkpoint-path", str(args.checkpoint_path)]
    if smoke:
        cmd.append("--smoke")
    return cmd


def masking_command(args, out: Path, carrying: list[int], *, smoke: bool) -> list[str]:
    cmd = [sys.executable, "-u", str(ROOT / MASKING_SCRIPT), "--config", str(args.config), "--reproduction-summary", str(args.reproduction_summary),
           "--output-dir", str(out), "--batch-size", str(args.batch_size), "--precision", str(args.precision), "--device", str(args.device),
           "--pool", str(args.pool), "--carrying-slots", *[str(s) for s in carrying]]
    if getattr(args, "checkpoint_path", None):
        cmd += ["--checkpoint-path", str(args.checkpoint_path)]
    if smoke:
        cmd.append("--smoke")
    return cmd


def run_stage(cmd: list[str], log_path: Path, *, attempts: int = 2, wait: int = 30) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for attempt in range(1, attempts + 1):
        with log_path.open("a") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line); log.write(line)
            code = proc.wait()
        if code == 0:
            return time.perf_counter() - started
        if attempt == attempts:
            raise subprocess.CalledProcessError(code, cmd)
        print(f"stage exited with {code}; retrying in {wait}s"); time.sleep(wait)
    return time.perf_counter() - started


def _load(path: Path | None) -> dict | None:
    return None if path is None or not Path(path).exists() else json.loads(Path(path).read_text())


def gate_from(summary_path: Path | None, cfg) -> dict | None:
    payload = _load(summary_path)
    if payload is None:
        return None
    gate = payload.get("accuracy_gate", {})
    return {"accuracy": payload.get("datasets", {}).get("gsm8k"), "published": float(cfg.accuracy_gate.published_accuracy.gsm8k),
            "tolerance": float(cfg.accuracy_gate.absolute_tolerance), "direction": str(getattr(cfg.accuracy_gate, "direction", "two_sided")),
            "status": gate.get("status"), "evaluated": payload.get("evaluated_counts", {}).get("gsm8k"), "summary": str(summary_path)}


def model_block(cfg, config_path: Path, twins: dict | None, name: str | None) -> dict:
    pre = (twins or {}).get("preregistration", {})
    return {"name": name or Path(config_path).stem.replace("official_", ""), "config": str(config_path), "checkpoint_repo": str(cfg.checkpoint.repo_id),
            "checkpoint_revision": str(cfg.checkpoint.revision), "checkpoint_sha256": (twins or {}).get("checkpoint_sha256") or str(cfg.checkpoint.sha256),
            "base_model": str(cfg.model.base_model), "layers": pre.get("backbone_layers"), "latent_iterations": int(cfg.eval.latent_iterations)}


def assemble(args, lay: dict, *, stage_seconds: dict, cfg) -> dict:
    twins_std = _load(None if lay["twins_standard"] is None else Path(lay["twins_standard"]) / "summary.json")
    twins_long = _load(None if lay["twins_long"] is None else Path(lay["twins_long"]) / "summary.json")
    masking = _load(None if lay["masking"] is None else Path(lay["masking"]) / "summary.json")
    runs = {p: Path(lay[f"twins_{p}"]) for p in ("standard", "long") if lay[f"twins_{p}"] is not None and (Path(lay[f"twins_{p}"]) / "predictions.pt").exists()}

    bootstrap = _load(lay["bootstrap"])
    if bootstrap is None and runs and "bootstrap" in args.stages:
        from scripts.analyze_directed_counterfactual_bootstrap import analyze as boot_analyze
        t = time.perf_counter(); bootstrap = {p: boot_analyze(d) for p, d in runs.items()}; stage_seconds["bootstrap"] = round(time.perf_counter() - t, 1)
        Path(lay["bootstrap"]).write_text(json.dumps(bootstrap, indent=2))
    monitor = _load(lay["monitor"])
    if monitor is None and runs and "monitor" in args.stages:
        from scripts.analyze_monitor_faithfulness import analyze as mon_analyze, load_model_pieces
        t = time.perf_counter()
        tokenizer, embeddings, readout = load_model_pieces(Path(args.config))
        monitor = {p: mon_analyze(d, tokenizer, embeddings, readout) for p, d in runs.items()}
        stage_seconds["monitor"] = round(time.perf_counter() - t, 1)
        Path(lay["monitor"]).write_text(json.dumps(monitor, indent=2))

    fingerprints = {}
    for p, d in runs.items():
        try:
            fingerprints[f"twins_{p}"] = predictions_fingerprint(d)
        except Exception as exc:  # a run without predictions.pt still yields a card
            fingerprints[f"twins_{p}"] = {"error": str(exc)}
    source = twins_std or twins_long or masking or {}
    pool_meta = source.get("pool") or {"pool": str(args.pool), "hash": (source.get("preregistration") or {}).get("sampling_hash_fit"), "n": None,
                                       "steps_histogram": None, "source": POOLS[str(args.pool)]["source"]}
    existing_card = _load(Path(args.from_existing) / "card.json") if getattr(args, "from_existing", None) else None
    if existing_card and existing_card.get("environment"):
        environment = dict(existing_card["environment"]); environment["environment_source"] = "kernel card"
    elif getattr(args, "from_existing", None):
        # runs made before the card existed: the local machine's versions say nothing about the GPU that ran them
        environment = {k: None for k in environment_fingerprint()} | {"dtype": str(args.precision), "batch_size": args.batch_size,
                                                                      "device": str(args.device), "environment_source": "unknown (pre-card run)"}
    else:
        environment = environment_fingerprint(device=args.device, dtype=args.precision, batch_size=args.batch_size)
        environment["environment_source"] = "this run"
    environment["from_existing"] = bool(getattr(args, "from_existing", None))
    if getattr(args, "device_name", None):
        environment["device_name"] = args.device_name   # the GPU the runs were made on, when assembling off-line
    gate = gate_from(getattr(args, "reproduction_summary", None), cfg)
    if gate is None and existing_card:
        gate = existing_card.get("gate")   # the kernel's own card carries the gate it ran under
    card = build_card(model=model_block(cfg, args.config, twins_std or twins_long, getattr(args, "model_name", None)), pool=pool_meta,
                      environment=environment, gate=gate,
                      twins_standard=twins_std, twins_long=twins_long, bootstrap=bootstrap, masking=masking, monitor=monitor,
                      fingerprints=fingerprints, stage_seconds=stage_seconds)
    card["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    card["smoke"] = bool(getattr(args, "smoke", False))
    return card


def run(args) -> dict:
    from src.utils.config import load_config

    cfg = load_config(str(args.config))
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict = {}
    if args.from_existing:
        lay = locate_existing(Path(args.from_existing), args.prefix)
        for key in ("card", "card_md", "stage_seconds"):
            lay[key] = layout(root)[key]
        for p in ("standard", "long"):
            s = _load(None if lay[f"twins_{p}"] is None else Path(lay[f"twins_{p}"]) / "summary.json")
            if s:
                stage_seconds[f"twins_{p}"] = round(float(s.get("seconds", 0)), 1)
        m = _load(None if lay["masking"] is None else Path(lay["masking"]) / "summary.json")
        if m:
            stage_seconds["masking"] = round(float(m.get("seconds", 0)), 1)
    else:
        lay = layout(root)
        stages = plan(str(args.pool), args.long, args.stages)
        print("stages", stages)
        for profile in ("standard", "long"):
            if f"twins_{profile}" in stages and not (lay[f"twins_{profile}"] / "summary.json").exists():
                stage_seconds[f"twins_{profile}"] = round(run_stage(twins_command(args, profile, lay[f"twins_{profile}"], smoke=args.smoke),
                                                                    lay["logs"] / f"twins_{profile}.log"), 1)
        if "masking" in stages and not (lay["masking"] / "summary.json").exists():
            if args.carrying == "auto":
                carrying = carrying_slots(_load(lay["twins_long"] / "summary.json")) if (lay["twins_long"] / "summary.json").exists() else []
            else:
                carrying = [int(s) for s in args.carrying.split(",") if s.strip()]
            print("carrying slots for masking", carrying)
            stage_seconds["masking"] = round(run_stage(masking_command(args, lay["masking"], carrying, smoke=args.smoke), lay["logs"] / "masking.log"), 1)
        for key in ("twins_standard", "twins_long", "masking"):
            if not (Path(lay[key]) / "summary.json").exists():
                lay[key] = None
    card = assemble(args, lay, stage_seconds=stage_seconds, cfg=cfg)
    Path(lay["card"]).write_text(json.dumps(card, indent=2))
    Path(lay["card_md"]).write_text(card_markdown(card) + "\n")
    Path(lay["stage_seconds"]).write_text(json.dumps(stage_seconds, indent=2))
    print(card_markdown(card))
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--pool", choices=sorted(POOLS), default=DEFAULT_POOL)
    parser.add_argument("--reproduction-summary", type=Path, help="full GSM8K reproduction summary (the gate); required unless --from-existing")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--carrying", default="auto", help="'auto' derives carriers from the long twins (ledger §123 threshold), or a comma list")
    parser.add_argument("--long", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--from-existing", type=Path, help="assemble the card from runs already under this directory")
    parser.add_argument("--prefix", help="run prefix in the §120 Kaggle layout (<prefix>_twins_standard ...)")
    parser.add_argument("--model-name", help="card label (defaults to the config stem)")
    parser.add_argument("--device-name", help="GPU name to record when assembling from existing runs")
    args = parser.parse_args()
    if not args.from_existing and args.reproduction_summary is None:
        parser.error("--reproduction-summary is required unless --from-existing is given")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
