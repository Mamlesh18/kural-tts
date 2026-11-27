"""Command-line interface: ``kural-tts "text" -o out.wav``."""

from __future__ import annotations

import argparse
import sys

from kural_tts import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kural-tts", description="Tamil-English text-to-speech.")
    p.add_argument("text", nargs="?", help="Text: Tamil, English, code-mixed or Tanglish.")
    p.add_argument("-f", "--file", help="Read text from a UTF-8 file instead.")
    p.add_argument("-o", "--output", default="out.wav", help="Output WAV path (default: out.wav).")
    p.add_argument("--lang", choices=["auto", "ta", "en"], default="auto")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--version", action="version", version=f"kural-tts {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        build_parser().print_usage(sys.stderr)
        return 2

    from kural_tts.api import KuralTTS, ModelNotAvailableError

    try:
        path = KuralTTS().save(text, args.output, lang=args.lang, speed=args.speed)
    except ModelNotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
