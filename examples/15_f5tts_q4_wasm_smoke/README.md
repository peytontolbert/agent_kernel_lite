# F5TTS Q4 WASM Smoke Test

This example verifies that the Q4 WASM primitive can run against a real tensor
from the exported F5TTS Peyton Q4 bundle.

Run from `/data/agent_kernel_lite`:

```bash
node examples/15_f5tts_q4_wasm_smoke/run.mjs
```

Default bundle:

`/data/resumebot/checkpoints/f5tts_peyton_q4_v0`

This does not synthesize audio yet. It validates the Q4 weight format, JS loader
shape handling, and WASM int4 matrix primitive on a real exported F5TTS tensor.
