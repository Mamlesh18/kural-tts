"""Baseline model adapters. Each one imports its heavy dependencies lazily inside ``load``.

Each model runs in its own virtualenv (see ``envs/``), because their dependency pins conflict.
Licenses are recorded *as stated on the model card*. Re-verify them on the run date.
"""

from __future__ import annotations

import os

import numpy as np
from common import Adapter

_state: dict = {}


def _torch_device(device: str) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


# ------------------------------------------------------------------------ MMS-TTS Tamil (VITS)
def _mms_load(device):
    from transformers import AutoTokenizer, VitsModel

    d = _torch_device(device)
    _state["mms"] = (
        VitsModel.from_pretrained("facebook/mms-tts-tam").to(d).eval(),
        AutoTokenizer.from_pretrained("facebook/mms-tts-tam"),
        d,
    )


def _mms_synth(text):
    import torch

    model, tok, d = _state["mms"]
    inputs = tok(text, return_tensors="pt").to(d)
    torch.manual_seed(0)
    with torch.no_grad():
        wav = model(**inputs).waveform[0].float().cpu().numpy()
    return wav, model.config.sampling_rate


MMS_TAM = Adapter(
    name="mms_tam",
    hf_id="facebook/mms-tts-tam",
    license="CC-BY-NC-4.0 (non-commercial; baseline only)",
    supports=("ta", "cs", "tanglish"),  # Latin-script words are dropped by its tokenizer
    load=_mms_load,
    synth=_mms_synth,
    notes="Single speaker, Tamil only. English words in CS input are silently dropped.",
)


# ------------------------------------------------------------------------ Indic Parler-TTS
PARLER_DESCRIPTION = os.environ.get(
    "PARLER_DESCRIPTION",
    "Jaya speaks with a clear, moderately paced and expressive voice in a close-sounding "
    "environment with very clear audio quality.",
)


def _parler_load(device):
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    d = _torch_device(device)
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(d)
    tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    desc = desc_tok(PARLER_DESCRIPTION, return_tensors="pt").to(d)
    _state["parler"] = (model.eval(), tok, desc, d)


def _parler_synth(text):
    import torch

    model, tok, desc, d = _state["parler"]
    prompt = tok(text, return_tensors="pt").to(d)
    torch.manual_seed(0)
    with torch.no_grad():
        gen = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
    return gen.float().cpu().numpy().squeeze(), model.config.sampling_rate


INDIC_PARLER = Adapter(
    name="indic_parler",
    hf_id="ai4bharat/indic-parler-tts",
    license="Apache-2.0 (gated repo: accept terms on HF)",
    supports=("en", "ta", "cs", "tanglish"),
    load=_parler_load,
    synth=_parler_synth,
    notes="Speaker 'Jaya' (card-recommended Tamil voice) used for ALL categories to test "
    "cross-lingual voice consistency.",
)


# ------------------------------------------------------------------------ IndicF5
def _f5_load(device):
    from transformers import AutoModel

    ref_audio = os.environ.get("F5_REF_AUDIO")
    ref_text = os.environ.get("F5_REF_TEXT")
    if not ref_audio or not ref_text:
        raise RuntimeError(
            "IndicF5 needs a Tamil reference clip: set F5_REF_AUDIO (wav path) and F5_REF_TEXT "
            "(its exact transcript). Use a clip whose license permits this use."
        )
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    if hasattr(model, "to"):
        model = model.to(_torch_device(device))
    _state["f5"] = (model, ref_audio, ref_text)


def _f5_synth(text):
    model, ref_audio, ref_text = _state["f5"]
    audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
    return np.asarray(audio), 24000


INDIC_F5 = Adapter(
    name="indic_f5",
    hf_id="ai4bharat/IndicF5",
    license="MIT tag on card; training-data lineage disputed (treat as reference only)",
    supports=("en", "ta", "cs", "tanglish"),
    load=_f5_load,
    synth=_f5_synth,
    notes="Zero-shot voice cloning from F5_REF_AUDIO. Voice identity follows the reference.",
)


# ------------------------------------------------------------------------ Coqui VITS English
def _coqui_load(device):
    from TTS.api import TTS

    _state["coqui"] = TTS("tts_models/en/ljspeech/vits").to(_torch_device(device))


def _coqui_synth(text):
    tts = _state["coqui"]
    wav = tts.tts(text=text)
    return np.asarray(wav, dtype=np.float32), tts.synthesizer.output_sample_rate


COQUI_VITS_EN = Adapter(
    name="coqui_vits_en",
    hf_id="coqui-tts: tts_models/en/ljspeech/vits",
    license="Code MPL-2.0; LJSpeech data public domain; CHECKPOINT LICENSE TO VERIFY",
    supports=("en",),
    load=_coqui_load,
    synth=_coqui_synth,
    notes="Candidate warm-start checkpoint for Kural-VITS.",
)


# ------------------------------------------------------------------------ svara-TTS (EXPERIMENTAL)
# The model card has no inference code yet. This follows the standard Orpheus token layout
# (start 128259, end 128009+128260, audio after 128257, 7 SNAC codes per frame, offset 128266).
# VERIFY against https://github.com/Kenpath/svara-tts-inference before trusting results.
SVARA_VOICE = os.environ.get("SVARA_VOICE", "Tamil (Female)")


def _svara_load(device):
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = _torch_device(device)
    dtype = torch.bfloat16 if d == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained("kenpath/svara-tts-v1")
    model = AutoModelForCausalLM.from_pretrained("kenpath/svara-tts-v1", torch_dtype=dtype).to(d)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(d).eval()
    _state["svara"] = (model.eval(), tok, snac, d)


def _svara_synth(text):
    import torch

    model, tok, snac, d = _state["svara"]
    ids = tok(f"{SVARA_VOICE}: {text}", return_tensors="pt").input_ids
    ids = torch.cat([torch.tensor([[128259]]), ids, torch.tensor([[128009, 128260]])], dim=1).to(d)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=2400,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128258,
        )[0].tolist()
    if 128257 in out:
        out = out[len(out) - out[::-1].index(128257) :]
    codes = [t - 128266 for t in out if t != 128258]
    codes = codes[: len(codes) // 7 * 7]
    if not codes:
        raise RuntimeError("model produced no audio tokens")
    l1, l2, l3 = [], [], []
    for i in range(0, len(codes), 7):
        c = codes[i : i + 7]
        l1.append(c[0])
        l2.append(c[1] - 4096)
        l3 += [c[2] - 2 * 4096, c[3] - 3 * 4096]
        l2.append(c[4] - 4 * 4096)
        l3 += [c[5] - 5 * 4096, c[6] - 6 * 4096]
    layers = [torch.tensor(x, device=d).unsqueeze(0) for x in (l1, l2, l3)]
    with torch.no_grad():
        audio = snac.decode(layers)
    return audio.float().cpu().numpy().squeeze(), 24000


SVARA = Adapter(
    name="svara",
    hf_id="kenpath/svara-tts-v1",
    license="Apache-2.0 (dataset terms apply separately)",
    supports=("en", "ta", "cs", "tanglish"),
    load=_svara_load,
    synth=_svara_synth,
    notes="EXPERIMENTAL decoding (Orpheus layout, unverified for svara). Voice from SVARA_VOICE.",
)


ADAPTERS = {a.name: a for a in (MMS_TAM, INDIC_PARLER, INDIC_F5, COQUI_VITS_EN, SVARA)}
