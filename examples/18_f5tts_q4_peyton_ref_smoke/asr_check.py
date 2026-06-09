from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe(path: Path, model_id: str) -> str:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = max(abs(float(audio.min())), abs(float(audio.max()))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.8
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask, max_new_tokens=64)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ASR check for generated Peyton TTS WAVs.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--model-id", default="openai/whisper-tiny.en")
    args = parser.parse_args()
    print(transcribe(args.wav, args.model_id))


if __name__ == "__main__":
    main()
