# Toward Progressive Trajectory Distillation for Compact Diffusion Students

## Working Title

Progressive Trajectory Bridging for Distilling Large Rectified-Flow Image Models into Compact Students

## Abstract Draft

We study the problem of distilling a large FLUX-style rectified-flow image generator into a substantially smaller student model. Direct imitation of final images or one-step teacher deltas was insufficient: the student could match local losses while failing under fresh-noise rollout, producing texture fields rather than coherent objects. Through a sequence of cached trajectory takeovers, live-teacher on-policy training, low-frequency transport objectives, and direct timestep bridge initialization, we found that the critical bottleneck is not late image refinement alone but early trajectory reachability. A compact student must first learn to map fresh latent noise onto the teacher's intermediate denoising manifold before final-image closure becomes stable. Our latest bridge-initializer runs show student-only generations with emerging silhouettes and scene-level structure, suggesting that explicit fresh-latent-to-intermediate-timestep bridges can provide a practical path toward scalable small-model diffusion distillation.

## Core Claim

Large-to-small diffusion distillation should be treated as a staged dynamical-systems alignment problem, not a single endpoint regression problem.

The student needs to learn three separable maps:

1. Fresh-noise transport: map initial noise into the teacher's early/intermediate latent manifold.
2. Trajectory continuation: remain stable under repeated student denoising steps.
3. Endpoint closure: refine the reachable intermediate state into a target-quality image.

## Empirical Findings So Far

### 1. Cached takeovers work before fresh-noise generation works

Cached or teacher-assisted starts at later timesteps produced coherent images well before fresh T0 sampling did. This showed that the student had learned usable late trajectory behavior but not the full initial transport from noise.

Evidence:

- v311 cached t12 produced recognizable objects.
- v322 cached t8 retained recognizable silhouettes and scene structure.
- v323/v324 cached t4/t2 degraded into smeared partial structure.
- v325c-v331 fresh T0 improved contrast but remained mostly texture/noise.

### 2. One-step losses were not predictive of full-rollout quality

Several live-teacher runs reached very small one-step endpoint or latent losses while fresh student-only generations remained incoherent. This implies that local vector-field imitation is necessary but insufficient for a compact student.

Observed pattern:

- One-step endpoint losses were often near 1e-4 to 1e-3.
- Multistep terminal losses and decoded low-frequency losses remained much larger.
- Visual samples still lacked stable foreground/background and object identity.

Interpretation:

The student can approximate the teacher's local update when placed near the teacher trajectory, but errors compound when the student must generate its own trajectory from fresh noise.

### 3. Low-frequency transport is the right intermediate objective

Pushing final decoded image losses too early encouraged texture-like solutions. Runs that emphasized low-frequency shape, multistep transport, and reduced high-frequency pressure produced more useful silhouette formation.

Practical lesson:

Do not force final-image fidelity before the student reaches the teacher's intermediate manifold.

### 4. Bridge initialization is the strongest recent result

The t15 bridge-initializer family trains a direct map from initial latents to a target teacher timestep. The best recent bridge samples show visible student-only silhouettes and scene layouts, especially compared with direct fresh T0 no-refiner rollouts.

Important limitation:

The current strongest bridge samples use cached-seed bridge targets. These should be described as student-only bridge generations, not yet as fully general fresh-noise image generation.

## Proposed Method

### Stage A: Teacher Trajectory Corpus

Generate teacher trajectories for prompt/seed pairs:

- Store latent states at selected timesteps.
- Store prompt embeddings and metadata.
- Include both easy object prompts and harder compositional prompts.
- Preserve fixed eval prompts for visual regression tracking.

### Stage B: Cached Takeover Curriculum

Train the student from later cached teacher states backward:

- t15 -> final
- t12 -> final
- t8 -> final
- t6/t4/t2 -> final only when the prior stage is stable

Purpose:

This isolates late denoising and endpoint closure from the harder fresh-noise transport problem.

### Stage C: Fresh-Noise Transport

Train starts at t0 with multistep objectives toward intermediate teacher states:

- target t4/t6/t8 before final image
- emphasize low-frequency latent and decoded structure
- reduce edge/high-frequency losses
- monitor rollout divergence at every timestep

Suggested objective:

```text
L_transport =
  w_flow * L_teacher_flow_1step
  + w_latent * L_latent_1step
  + w_ms * L_student_rollout_to_teacher_tk
  + w_lowfreq * L_decoded_lowfreq_tk
  + w_prompt * L_prompt_counterfactual
```

### Stage D: Direct Timestep Bridge Initializer

Train a bridge model or bridge mode in the student:

```text
z_hat_k = B_student(z_0, t_0, prompt)
```

where `z_hat_k` approximates the teacher latent at an intermediate timestep k, such as t8, t12, or t15.

Bridge output modes tested:

- delta mode: predict `z_k - z_0`
- absolute mode: predict `z_k` directly

Current evidence favors absolute bridge prediction for producing visible silhouettes when paired with cached bridge targets.

### Stage E: Endpoint Closure

Once the bridge produces stable intermediate structure, train the normal student rollout from that intermediate state:

- t15/t12/t8 closure first
- decoded low-frequency before full decoded image
- endpoint/refiner losses later
- only then merge the bridge path with full T0 sampling

## Proposed Experiments For The Paper

### Experiment 1: One-step loss vs full-rollout quality

Compare:

- one-step endpoint loss
- horizon-4/6/15 terminal loss
- decoded low-frequency loss
- visual silhouette score

Claim to test:

One-step loss correlates poorly with fresh-noise student generation quality.

### Experiment 2: Cached takeover ladder

Evaluate the same prompts from:

- cached t15
- cached t12
- cached t8
- cached t6
- cached t4
- cached t2
- fresh t0

Claim to test:

The collapse point identifies where the student leaves the teacher manifold.

### Experiment 3: Low-frequency-first vs endpoint-first training

Compare:

- final endpoint-heavy training
- decoded edge/highfreq training
- low-frequency transport training
- bridge-initializer training

Claim to test:

Low-frequency trajectory transport improves silhouettes more reliably than direct final-image regression.

### Experiment 4: Bridge initializer ablation

Compare:

- no bridge
- delta bridge
- absolute bridge
- absolute bridge plus decoded low-frequency loss
- bridge target t8/t12/t15

Claim to test:

A direct fresh-latent-to-intermediate bridge improves reachability for compact students.

### Experiment 5: Prompt sensitivity

Use matched seeds with prompt swaps and negative prompts.

Measure:

- prompt counterfactual delta norm
- teacher/student prompt sensitivity ratio
- decoded low-frequency difference
- object silhouette diversity

Claim to test:

Early prompt sensitivity must be preserved during transport; otherwise the student forms generic texture fields.

## Metrics To Add

- Rollout divergence curve: `MSE(z_student_tk, z_teacher_tk)` for k in 1,2,4,6,8,12,15.
- Low-frequency decoded error at intermediate timesteps.
- Frequency energy ratio over rollout.
- Prompt sensitivity ratio.
- Silhouette/objectness score from segmentation or a lightweight detector.
- Cached-start success frontier: earliest timestep that still gives recognizable output.
- Bridge reachability score: distance from bridge output to teacher manifold.

## Paper Contribution Framing

The paper should not claim that we already have a 300M model matching FLUX-dev generally.

The honest contribution is stronger:

1. We identify why naive distillation fails for compact diffusion students.
2. We show that cached trajectory success does not imply fresh-noise success.
3. We introduce a staged trajectory-bridging curriculum.
4. We show early evidence that direct timestep bridge initialization produces silhouettes where direct T0 rollout produced texture fields.
5. We propose diagnostics that expose the gap between local vector-field imitation and global generative dynamics.

## Current Limitations

- Latest bridge samples are student-only, but many use cached-seed bridge targets.
- Silhouettes are visible, but object identity and image fidelity are still weak.
- Direct fresh T0 no-refiner samples remain noisy compared with bridge-initialized samples.
- The method needs controlled ablations before publication-quality claims.

## Next Paper-Quality Run

Use one fixed eval panel:

- 24 prompts
- fixed seeds
- teacher images
- direct T0 student
- cached t8/t12 student
- bridge t15 student
- bridge plus closure student

For every run, save:

- contact sheet
- per-sample metrics
- rollout divergence curves
- intermediate decoded states
- prompt sensitivity metrics

The key figure should be a row-wise progression:

```text
teacher final | direct T0 student | cached t8 student | bridge t15 student | bridge+closure student
```

This will make the research story clear: the bridge is not a cosmetic trick; it changes reachability.
