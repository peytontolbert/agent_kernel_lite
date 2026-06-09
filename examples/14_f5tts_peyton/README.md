# F5TTS Peyton Voice Smoke Test

This example verifies the existing Peyton voice clone service from
`/data/resumebot` and writes a generated WAV into this example directory.

Run from `/data/agent_kernel_lite`:

```bash
/home/peyton/miniconda3/envs/ai/bin/python examples/14_f5tts_peyton/run.py \
  --cuda-visible-devices 2 \
  --text "This is a quick local test of my F5 TTS voice running from Agent Kernel Lite."
```

To hear a post-training int4 approximation before doing any int4-aware
training:

```bash
/home/peyton/miniconda3/envs/ai/bin/python examples/14_f5tts_peyton/run.py \
  --cuda-visible-devices 2 \
  --simulate-q4 \
  --nfe-step 8 \
  --text "This is a quick local test of my F5 TTS voice with simulated int4 weights."
```

To synthesize from a recovery-training checkpoint:

```bash
/home/peyton/miniconda3/envs/ai/bin/python examples/14_f5tts_peyton/run.py \
  --cuda-visible-devices 2 \
  --checkpoint /data/agent_kernel_lite/artifacts/f5tts_q4_stream_smoke/model_q4_last.pt \
  --output-name peyton_f5tts_q4_recovery_smoke.wav \
  --text "This is a quick local test after int4 recovery training."
```

Default assets:

- Checkpoint: `/data/resumebot/checkpoints/final_finetuned_model.pt`
- Voice profile: `/data/resumebot/voice_profiles/Peyton`
- Output dir: `/data/agent_kernel_lite/examples/14_f5tts_peyton/out`

The `--simulate-q4` mode uses the original PyTorch F5TTS graph and vocoder, but
rounds large matrix weights through rowwise signed int4 and dequantizes them
back into the model before synthesis. That gives an audio-quality preview of
post-training int4 before the complete Q4 WASM runtime is wired.

The script intentionally loads the checkpoint on CPU first and applies only
`model_state_dict`. That avoids the original resume bot service behavior of
loading the full checkpoint, including duplicate EMA state, straight onto CUDA.
