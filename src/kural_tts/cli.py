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
    p.add_argument("--speed", type=float, default=1.0, help="> 1 speaks faster.")
    p.add_argument("--noise-scale", type=float, help="Expressiveness (0 = flattest, most stable).")
    p.add_argument("--model", help="Voice directory with model.onnx + config.json.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Only run the text frontend and print language-tagged spans (no audio).",
    )
    p.add_argument(
        "--phonemes",
        action="store_true",
        help="Print the phoneme sequence the model receives (no audio).",
    )
    p.add_argument("--version", action="version", version=f"kural-tts {__version__}")
    return p


def serve_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="kural-tts serve", description="Run the local TTS server.")
    p.add_argument("--model", help="Voice directory with model.onnx + config.json.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-concurrency", type=int, default=2)
    args = p.parse_args(argv)
    try:
        from kural_tts.server.app import serve
    except ImportError:
        print('error: install the server extra: pip install "kural-tts[server]"', file=sys.stderr)
        return 1
    serve(args.model, args.host, args.port, args.device, args.max_concurrency)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:])
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

    if args.normalize:
        _print_spans(KuralTTS().normalize(text, lang=args.lang))
        return 0
    if args.phonemes:
        from kural_tts.text.g2p import format_phones

        _utf8_stdout()
        print(format_phones(KuralTTS().phonemize(text, lang=args.lang)))
        return 0

    try:
        tts = KuralTTS(model_path=args.model, device=args.device)
        path = tts.save(
            text, args.output, lang=args.lang, speed=args.speed, noise_scale=args.noise_scale
        )
    except ModelNotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")


def _print_spans(spans) -> None:
    _utf8_stdout()
    for span in spans:
        print(f"[{span.lang}] {span.text}")


if __name__ == "__main__":
    raise SystemExit(main())
