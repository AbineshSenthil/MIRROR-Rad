"""Checksum-bearing release manifests for fail-closed runtime assembly."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radtrace.product import PRODUCT_NAME

RELEASE_SCHEMA = "mirror-rad.release.v1"
REQUIRED_RELEASE_ASSETS = frozenset(
    {
        "application",
        "engine",
        "requirements_lock",
        "verifier_config",
        "fusion_model",
        "nli_config",
        "nli_weights",
        "nli_calibration",
        "nli_special_tokens",
        "nli_tokenizer_config",
        "nli_tokenizer",
        "nli_vocab",
        "report_router",
        "evaluation_ledger",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _asset(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required release asset is missing: {path}")
    return {
        "kind": "file",
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _directory_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    files = [
        value
        for value in path.rglob("*")
        if value.is_file()
        and "__pycache__" not in value.parts
        and value.suffix not in {".pyc", ".pyo"}
    ]
    for value in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative_name = value.relative_to(path).as_posix()
        file_hash = file_sha256(value)
        size = value.stat().st_size
        total += size
        digest.update(f"{relative_name}\0{size}\0{file_hash}\n".encode())
    return total, digest.hexdigest()


def _directory_asset(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Required release directory is missing: {path}")
    size, digest = _directory_digest(path)
    return {
        "kind": "directory",
        "path": _relative(path, root),
        "bytes": size,
        "sha256": digest,
    }


def build_release_manifest(
    root: str | Path,
    *,
    artifact: str | Path,
    calibrator: str | Path,
    evaluation_ledger: str | Path,
    include_evaluation_predictions: bool = False,
) -> dict[str, Any]:
    """Build a manifest only after every final asset is present."""

    project_root = Path(root).resolve()
    artifact_path = Path(artifact).resolve()
    config_path = artifact_path / "verifier_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = Path(str(config.get("nli_model", "")).replace("\\", "/"))
    if not model_path.is_absolute():
        model_path = project_root / model_path
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        tokenizer_path = model_path / "vocab.txt"
    assets: dict[str, dict[str, Any]] = {
        "application": _asset(project_root / "app.py", project_root),
        "engine": _directory_asset(project_root / "src" / "radtrace", project_root),
        "requirements_lock": _asset(project_root / "requirements-lock.txt", project_root),
        "verifier_config": _asset(config_path, project_root),
        "fusion_model": _asset(artifact_path / "fusion_model.pkl", project_root),
        "nli_config": _asset(model_path / "config.json", project_root),
        "nli_weights": _asset(model_path / "model.safetensors", project_root),
        "nli_calibration": _asset(model_path / "calibration.json", project_root),
        "nli_special_tokens": _asset(model_path / "special_tokens_map.json", project_root),
        "nli_tokenizer_config": _asset(model_path / "tokenizer_config.json", project_root),
        "nli_tokenizer": _asset(tokenizer_path, project_root),
        "nli_vocab": _asset(model_path / "vocab.txt", project_root),
        "report_router": _asset(Path(calibrator).resolve(), project_root),
        "evaluation_ledger": _asset(Path(evaluation_ledger).resolve(), project_root),
    }
    if include_evaluation_predictions:
        assets["evaluation_predictions"] = _directory_asset(
            project_root / "reports" / "generated" / "public_benchmarks",
            project_root,
        )
    identity = "|".join(f"{name}:{entry['sha256']}" for name, entry in sorted(assets.items()))
    return {
        "schema": RELEASE_SCHEMA,
        "project": PRODUCT_NAME,
        "release_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "assets": assets,
    }


def write_release_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Publish the completed release marker with an atomic same-volume replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.pending")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_release_manifest(path: str | Path, root: str | Path) -> dict[str, Any]:
    """Validate schema, path containment, sizes, and hashes for every asset."""

    manifest_path = Path(path)
    project_root = Path(root).resolve()
    result: dict[str, Any] = {
        "valid": False,
        "release_id": None,
        "errors": [],
        "assets": {},
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        result["errors"].append(f"Manifest could not be read: {type(exc).__name__}: {exc}")
        return result
    if manifest.get("schema") != RELEASE_SCHEMA:
        result["errors"].append("Unsupported release manifest schema.")
    if manifest.get("project") != PRODUCT_NAME:
        result["errors"].append("Release manifest project identity does not match.")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        result["errors"].append("Release manifest has no assets.")
        return result
    missing_assets = sorted(REQUIRED_RELEASE_ASSETS.difference(assets))
    if missing_assets:
        result["errors"].append(
            "Release manifest is incomplete; missing assets: " + ", ".join(missing_assets)
        )
    for name, entry in assets.items():
        if not isinstance(entry, dict):
            result["errors"].append(f"Asset {name} has an invalid entry.")
            continue
        configured = Path(str(entry.get("path", "")))
        target = configured if configured.is_absolute() else project_root / configured
        target = target.resolve()
        try:
            target.relative_to(project_root)
        except ValueError:
            result["errors"].append(f"Asset {name} resolves outside the project root.")
            continue
        kind = str(entry.get("kind", "file"))
        if kind == "directory":
            if not target.is_dir():
                result["errors"].append(f"Asset {name} directory is missing.")
                continue
            actual_size, actual_hash = _directory_digest(target)
        else:
            if not target.is_file():
                result["errors"].append(f"Asset {name} is missing.")
                continue
            actual_size = target.stat().st_size
            actual_hash = file_sha256(target)
        expected_size = int(entry.get("bytes", -1))
        expected_hash = str(entry.get("sha256", ""))
        matched = actual_size == expected_size and actual_hash == expected_hash
        result["assets"][name] = {
            "path": target,
            "bytes": actual_size,
            "sha256": actual_hash,
            "matched": matched,
        }
        if not matched:
            result["errors"].append(f"Asset {name} does not match the sealed release.")
    result["release_id"] = manifest.get("release_id")
    result["valid"] = not result["errors"]
    return result
