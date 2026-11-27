"""Publish an exported voice bundle to the Hugging Face Hub and pin it in the package.

    # dry run: checksums + model card only
    python scripts/release/publish_voice.py exports/sahithi_v1 --repo <org>/kural-tts-sahithi

    # upload (needs `huggingface-cli login` or HF_TOKEN with write access)
    python scripts/release/publish_voice.py exports/sahithi_v1 --repo <org>/kural-tts-sahithi \
        --upload

After uploading, the script prints the lines to paste into src/kural_tts/hub.py (repo id, commit
revision and sha256 of each file). Every package release then pins one exact voice revision.
Only release a voice once the consent agreement (recording/consent_template.md) is signed and
the evaluation report meets the release bar in docs/PLAN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = ("model.onnx", "config.json")
REPO = Path(__file__).resolve().parents[2]
CARD_TEMPLATE = REPO / "docs" / "MODEL_CARD.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def prepare(bundle: Path) -> dict[str, str]:
    missing = [f for f in FILES if not (bundle / f).exists()]
    if missing:
        raise SystemExit(f"{bundle} is missing {missing}; run training.export_onnx first")
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    card = CARD_TEMPLATE.read_text(encoding="utf-8")
    card = card.replace("{{training_step}}", str(cfg.get("training_step")))
    card = card.replace("{{sample_rate}}", str(cfg["sample_rate"]))
    (bundle / "README.md").write_text(card, encoding="utf-8")
    return {name: sha256(bundle / name) for name in FILES}


def upload(bundle: Path, repo_id: str, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(bundle), repo_id=repo_id, repo_type="model", commit_message=message,
        allow_patterns=[*FILES, "README.md"],
    )  # fmt: skip
    return info.oid


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("bundle", type=Path)
    p.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. org/kural-tts-sahithi")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--message", default="Release Sahithi voice")
    args = p.parse_args(argv)

    hashes = prepare(args.bundle)
    print("checksums:")
    for name, digest in hashes.items():
        print(f"  {name}: {digest}")
    if not args.upload:
        print("\ndry run: model card written to", args.bundle / "README.md")
        return
    revision = upload(args.bundle, args.repo, args.message)
    print("\nPaste into src/kural_tts/hub.py:\n")
    print(f'HF_REPO_ID = "{args.repo}"')
    print(f'HF_REVISION: str | None = "{revision}"')
    print("MODEL_FILES: dict[str, str] = {")
    for name, digest in hashes.items():
        print(f'    "{name}": "{digest}",')
    print("}")


if __name__ == "__main__":
    main()
