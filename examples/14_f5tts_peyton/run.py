#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


RESUMEBOT_DIR = Path("/data/resumebot")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Peyton-voice F5TTS smoke-test WAV.")
    parser.add_argument(
        "--text",
        default="This is a quick local test of my F5 TTS voice running from Agent Kernel Lite.",
        help="Text to synthesize.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory to copy the generated WAV into.")
    parser.add_argument("--model-dir", default=str(RESUMEBOT_DIR / "checkpoints"), help="F5TTS checkpoint directory.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint file. Defaults to <model-dir>/final_finetuned_model.pt.",
    )
    parser.add_argument("--voice-profile", default="Peyton", help="Voice profile name.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"),
        help="CUDA_VISIBLE_DEVICES value set before importing torch/F5TTS.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--nfe-step", type=int, default=16, help="F5TTS denoising steps for the smoke test.")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="Classifier-free guidance strength.")
    parser.add_argument("--speed", type=float, default=1.0, help="F5-TTS generation speed multiplier.")
    parser.add_argument(
        "--simulate-q4",
        action="store_true",
        help="Quantize/dequantize matrix weights to rowwise signed int4 before inference.",
    )
    parser.add_argument("--output-name", default="", help="Output WAV filename. Defaults based on quantization mode.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the original temp file under /data/resumebot/temp_audio in addition to the copied WAV.",
    )
    return parser.parse_args()


def should_q4_simulate(name: str, tensor) -> bool:
    if not name.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    if any(part in name for part in ("text_embed.text_embed", "mel_spec")):
        return False
    return True


def q4_dequantize_rowwise(tensor):
    import torch

    original_dtype = tensor.dtype
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
    dequantized = (quantized * scale[:, None]).reshape_as(tensor)
    return dequantized.to(original_dtype)


def apply_q4_simulation(state: dict[str, object]) -> tuple[int, int]:
    import torch

    tensors = 0
    params = 0
    for name, tensor in list(state.items()):
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and should_q4_simulate(name, tensor):
            state[name] = q4_dequantize_rowwise(tensor)
            tensors += 1
            params += int(tensor.numel())
    return tensors, params


def main() -> None:
    args = parse_args()
    if not RESUMEBOT_DIR.exists():
        raise SystemExit(f"missing resumebot directory: {RESUMEBOT_DIR}")

    # tts_service imports torch at module import time, so set this first.
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, str(RESUMEBOT_DIR))
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from config import TEMP_AUDIO_DIR  # noqa: PLC0415
    from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text  # noqa: PLC0415
    from f5_tts.model import CFM, DiT  # noqa: PLC0415

    def soundfile_torchaudio_load(path: str):
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        return torch.from_numpy(data.T.copy()), int(sample_rate)

    torchaudio.load = soundfile_torchaudio_load

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model_dir = Path(args.model_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else model_dir / "final_finetuned_model.pt"
    vocab_path = model_dir / "F5TTS_Base_vocab.txt"
    voice_dir = RESUMEBOT_DIR / "voice_profiles" / args.voice_profile
    samples_path = voice_dir / "samples.txt"

    vocab = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines()]
    vocab_char_map = {char: idx for idx, char in enumerate(vocab) if char}
    vocab_size = len(vocab_char_map) + 1

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    print(f"loading checkpoint on CPU: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    if args.simulate_q4:
        q4_tensors, q4_params = apply_q4_simulation(state)
        print(f"simulated_q4_tensors={q4_tensors}")
        print(f"simulated_q4_params={q4_params}")
    model.load_state_dict(state)
    del checkpoint
    model = model.to(device).eval()

    print(f"loading Vocos on {device}")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    sample_line = samples_path.read_text(encoding="utf-8").splitlines()[0]
    audio_file, ref_text = sample_line.split("|", 1)
    ref_audio_path = voice_dir / Path(audio_file).name
    ref_audio, ref_text = preprocess_ref_audio_text(str(ref_audio_path), ref_text)

    print(f"synthesizing on {device}: {args.text!r}")
    synth_start = time.perf_counter()
    with torch.inference_mode():
        audio, sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            str(args.text),
            model,
            vocoder,
            mel_spec_type="vocos",
            speed=float(args.speed),
            nfe_step=int(args.nfe_step),
            cfg_strength=float(args.cfg_strength),
            sway_sampling_coef=-1.0,
        )
    synth_elapsed = time.perf_counter() - synth_start
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))

    temp_dir = Path(TEMP_AUDIO_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    generated_path = temp_dir / "agent_kernel_lite_peyton_f5tts_smoke.wav"
    sf.write(generated_path, audio, sample_rate, format="WAV", subtype="PCM_16")

    output_name = args.output_name or ("peyton_f5tts_q4_sim_smoke.wav" if args.simulate_q4 else "peyton_f5tts_smoke.wav")
    target_path = out_dir / output_name
    shutil.copy2(generated_path, target_path)
    if not args.keep_temp:
        generated_path.unlink(missing_ok=True)

    print(f"generated_wav={target_path}")
    print(f"bytes={target_path.stat().st_size}")
    print(f"synth_elapsed_seconds={synth_elapsed:.4f}")


if __name__ == "__main__":
    main()
