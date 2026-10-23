"""The sealed release manifest guards the deployed app; keep it in sync."""

from pathlib import Path

from conftest import ROOT, checkpoint_available
from radtrace.release import build_release_manifest, validate_release_manifest, write_release_manifest


def _make_release_tree(root: Path) -> None:
    (root / "src" / "radtrace").mkdir(parents=True)
    (root / "src" / "radtrace" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (root / "app.py").write_text("print('app')\n", encoding="utf-8")
    (root / "requirements-lock.txt").write_text("numpy==2.4.6\n", encoding="utf-8")
    model = root / "artifacts" / "model"
    model.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "calibration.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ):
        (model / name).write_text(name, encoding="utf-8")
    verifier = root / "artifacts" / "verifier"
    verifier.mkdir()
    (verifier / "verifier_config.json").write_text('{"nli_model": "artifacts/model"}', encoding="utf-8")
    (verifier / "fusion_model.pkl").write_bytes(b"fusion")
    (root / "artifacts" / "router.pkl").write_bytes(b"router")
    (root / "ledger.json").write_text("{}", encoding="utf-8")


def test_manifest_detects_any_changed_source_file(tmp_path):
    _make_release_tree(tmp_path)
    manifest = build_release_manifest(
        tmp_path,
        artifact=tmp_path / "artifacts" / "verifier",
        calibrator=tmp_path / "artifacts" / "router.pkl",
        evaluation_ledger=tmp_path / "ledger.json",
    )
    path = write_release_manifest(tmp_path / "release.json", manifest)
    assert validate_release_manifest(path, tmp_path)["valid"]

    (tmp_path / "src" / "radtrace" / "__init__.py").write_text("x = 2\n", encoding="utf-8")
    result = validate_release_manifest(path, tmp_path)
    assert not result["valid"]
    assert "Asset engine does not match the sealed release." in result["errors"]


def test_repository_release_manifest_matches_the_code():
    """Fails when app.py or src/ changed without `python scripts/reseal_release.py`."""

    result = validate_release_manifest(ROOT / "artifacts" / "mirror-rad-release.json", ROOT)
    errors = list(result["errors"])
    if not checkpoint_available():
        # CI checks out without Git LFS, so only the weights may differ.
        errors = [error for error in errors if "nli_weights" not in error]
    assert errors == []
