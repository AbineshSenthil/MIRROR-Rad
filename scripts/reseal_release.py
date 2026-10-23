"""Regenerate the sealed release manifest after changing app.py or src/radtrace.

The deployed app refuses to run audits when any sealed file differs from
``artifacts/mirror-rad-release.json``.  Run this from the repository root, with
the real checkpoint present (``git lfs pull``), and commit the manifest together
with the code change:

    python scripts/reseal_release.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radtrace.release import (  # noqa: E402
    build_release_manifest,
    validate_release_manifest,
    write_release_manifest,
)

MANIFEST = ROOT / "artifacts" / "mirror-rad-release.json"
WEIGHTS = ROOT / "artifacts" / "mirror-rad-clinical-nli" / "best" / "model.safetensors"


def main() -> int:
    if WEIGHTS.stat().st_size < 10_000_000:
        raise SystemExit("The checkpoint is still a Git LFS pointer. Run `git lfs pull` first.")
    manifest = build_release_manifest(
        ROOT,
        artifact=ROOT / "artifacts" / "mirror-rad-system" / "tuned_verifier",
        calibrator=ROOT / "artifacts" / "mirror-rad-risk-calibrator.pkl",
        evaluation_ledger=ROOT / "reports" / "public" / "evaluation_summary.json",
    )
    write_release_manifest(MANIFEST, manifest)
    result = validate_release_manifest(MANIFEST, ROOT)
    if not result["valid"]:
        raise SystemExit("Resealed manifest failed validation: " + "; ".join(result["errors"]))
    print(f"Resealed release {result['release_id']} ({len(manifest['assets'])} assets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
