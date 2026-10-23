"""Command-line entry point for the MIRROR-Rad research workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bidirectional import BidirectionalAuditor, ReportRiskCalibrator
from .config import PROJECT_ROOT
from .product import (
    ENV_ARTIFACT,
    ENV_CALIBRATOR,
    ENV_DEVICE,
    ENV_EVALUATION_LEDGER,
    ENV_RELEASE_MANIFEST,
    PRODUCT_TAGLINE,
)
from .release import validate_release_manifest
from .verifier import RadTraceVerifier

DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "mirror-rad-system" / "tuned_verifier"
DEFAULT_CALIBRATOR = PROJECT_ROOT / "artifacts" / "mirror-rad-risk-calibrator.pkl"
DEFAULT_LEDGER = PROJECT_ROOT / "reports" / "public" / "evaluation_summary.json"
DEFAULT_RELEASE_MANIFEST = PROJECT_ROOT / "artifacts" / "mirror-rad-release.json"
SCRIPT_COMMANDS = {
    "prepare": "prepare_data.py",
    "train-transformer": "train_transformer.py",
    "ablations": "run_ablations.py",
    "evaluate-public": "evaluate_public_benchmarks.py",
}


def _load_text(value: str | None, file_path: str | None, name: str) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    if value:
        return value.strip()
    raise SystemExit(f"Provide --{name} or --{name}-file.")


def _run_script(name: str, arguments: Sequence[str]) -> int:
    script = PROJECT_ROOT / "scripts" / name
    if not script.is_file():
        raise SystemExit(
            f"scripts/{name} is not in this checkout. Training and benchmark scripts need the "
            "licensed datasets described in README.md (Data and licensing)."
        )
    command = [sys.executable, str(script), *arguments]
    return subprocess.call(command, cwd=PROJECT_ROOT)


def _require_release(
    manifest_path: Path,
    *,
    artifact: Path | None = None,
    calibrator: Path | None = None,
    evaluation_ledger: Path | None = None,
) -> dict[str, Any]:
    validation = validate_release_manifest(manifest_path, PROJECT_ROOT)
    if not validation["valid"]:
        raise SystemExit(
            "Sealed release validation failed: " + "; ".join(validation["errors"])
        )
    assets = validation.get("assets", {})
    if artifact is not None:
        expected = assets.get("verifier_config", {}).get("path")
        actual = (artifact.resolve() / "verifier_config.json").resolve()
        if expected is None or Path(expected).resolve() != actual:
            raise SystemExit("The requested verifier is not part of the sealed release.")
    if calibrator is not None:
        expected = assets.get("report_router", {}).get("path")
        if expected is None or Path(expected).resolve() != calibrator.resolve():
            raise SystemExit("The requested report router is not part of the sealed release.")
    if evaluation_ledger is not None:
        expected = assets.get("evaluation_ledger", {}).get("path")
        if expected is None or Path(expected).resolve() != evaluation_ledger.resolve():
            raise SystemExit("The requested evaluation ledger is not part of the sealed release.")
    return validation


def _audit(args: argparse.Namespace) -> int:
    reference = _load_text(args.reference, args.reference_file, "reference")
    candidate = _load_text(args.candidate, args.candidate_file, "candidate")
    release = _require_release(
        args.release_manifest,
        artifact=args.artifact,
        calibrator=args.calibrator,
    )
    verifier = RadTraceVerifier.load_artifact(args.artifact, device_override=args.device)
    available = getattr(verifier.nli_predictor, "available", True)
    load_error = getattr(verifier.nli_predictor, "load_error", None)
    if not available or load_error:
        detail = load_error or "backend reported unavailable"
        raise SystemExit(f"Clinical NLI checkpoint failed to initialize: {detail}")
    if verifier.fusion_model is None or not verifier.fusion_model.is_fitted:
        raise SystemExit("The inference artifact does not contain a fitted fusion model.")
    calibrator = ReportRiskCalibrator.load(args.calibrator)
    result = BidirectionalAuditor(verifier, calibrator=calibrator).audit(reference, candidate)
    payload = {
        "schema": "mirror-rad.bidirectional-audit.v2",
        "release_id": release.get("release_id"),
        "neural_fallback_used": False,
        "audit": result.to_dict(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved audit to {args.output}")
    else:
        print(rendered)
    return 0


def _serve(args: argparse.Namespace) -> int:
    _require_release(
        args.release_manifest,
        artifact=args.artifact,
        calibrator=args.calibrator,
        evaluation_ledger=args.ledger,
    )
    environment = os.environ.copy()
    environment[ENV_ARTIFACT] = str(args.artifact.resolve())
    environment[ENV_CALIBRATOR] = str(args.calibrator.resolve())
    environment[ENV_DEVICE] = args.device
    environment[ENV_EVALUATION_LEDGER] = str(args.ledger.resolve())
    environment[ENV_RELEASE_MANIFEST] = str(args.release_manifest.resolve())
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.address",
        args.address,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--server.maxUploadSize",
        "10",
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)


def _info(_: argparse.Namespace) -> int:
    summary_path = PROJECT_ROOT / "data" / "processed" / "dataset_summary.json"
    payload = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "dataset_ready": summary_path.exists(),
        "default_artifact_ready": DEFAULT_ARTIFACT.is_dir(),
        "default_calibrator_ready": DEFAULT_CALIBRATOR.is_file(),
        "evaluation_ledger_ready": DEFAULT_LEDGER.is_file(),
    }
    release = validate_release_manifest(DEFAULT_RELEASE_MANIFEST, PROJECT_ROOT)
    payload["sealed_release_ready"] = bool(release["valid"])
    payload["release_id"] = release.get("release_id")
    payload["release_errors"] = release.get("errors", [])
    if summary_path.exists():
        manifest = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["corpus"] = manifest.get("corpus", {})
        payload["splits"] = manifest.get("splits", {})
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        payload["torch"] = None
        payload["cuda_available"] = False
        payload["gpu"] = None
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mirror-rad",
        description=PRODUCT_TAGLINE,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Parse and split the Open-i XML reports.")
    prepare.add_argument("arguments", nargs=argparse.REMAINDER)
    prepare.set_defaults(handler=lambda args: _run_script("prepare_data.py", args.arguments))

    train_transformer = subparsers.add_parser("train-transformer", help="Fine-tune a transformer NLI model.")
    train_transformer.add_argument("arguments", nargs=argparse.REMAINDER)
    train_transformer.set_defaults(
        handler=lambda args: _run_script("train_transformer.py", args.arguments)
    )

    ablations = subparsers.add_parser(
        "ablations",
        help="Run claim-level baselines, ablations, calibration, and frozen controlled testing.",
    )
    ablations.add_argument("arguments", nargs=argparse.REMAINDER)
    ablations.set_defaults(handler=lambda args: _run_script("run_ablations.py", args.arguments))

    evaluate_public = subparsers.add_parser(
        "evaluate-public",
        help="Run frozen ReXErr and RadEvalX benchmark evaluation.",
    )
    evaluate_public.add_argument("arguments", nargs=argparse.REMAINDER)
    evaluate_public.set_defaults(
        handler=lambda args: _run_script("evaluate_public_benchmarks.py", args.arguments)
    )

    audit = subparsers.add_parser("audit", help="Run the full bidirectional audit on one report pair.")
    audit.add_argument("--reference", "--findings", dest="reference")
    audit.add_argument("--reference-file", "--findings-file", dest="reference_file")
    audit.add_argument("--candidate", "--impression", dest="candidate")
    audit.add_argument("--candidate-file", "--impression-file", dest="candidate_file")
    audit.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    audit.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    audit.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    audit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=_audit)

    serve = subparsers.add_parser("serve", help="Launch the Streamlit dashboard.")
    serve.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    serve.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    serve.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    serve.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    serve.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--address", default="127.0.0.1")
    serve.set_defaults(handler=_serve)

    info = subparsers.add_parser("info", help="Show environment, data and artifact status.")
    info.set_defaults(handler=_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in SCRIPT_COMMANDS:
        return _run_script(SCRIPT_COMMANDS[arguments[0]], arguments[1:])
    args = build_parser().parse_args(arguments)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
