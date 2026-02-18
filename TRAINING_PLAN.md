# AASSP Neural Mesh — Training Plan

## 1. The Problem

We have a liquid neural network mesh with ~4,736 neurons, 13 circuits, 5 overlap zones, and 3 memory tiers — all wired up for inference but with no way to learn. Every neuron starts at `x=0`, every connection list is empty, every weight is zero. The mesh currently does nothing useful.

Training must:

1. Initialize and learn sparse connection weights between neurons
2. Learn overlap gate parameters so circuits communicate properly
3. Learn overlap masks to discover optimal overlap width
4. Learn write-gate weights so memory circuits know when to store
5. Produce a frozen universal core that generalizes across wells
6. Leave adapter layers plastic for online well-specific learning

---

## 2. Trainable Parameter Inventory

| Component | Parameters | Count (est.) |
|---|---|---|
| Neuron biases | 1 per neuron | 4,736 |
| Neuron connection weights | ~150-380 per neuron at target sparsity | ~500K-1M |
| Overlap gate weights | 74 per zone × 5 zones | 370 |
| Overlap masks | ~100 per zone × 5 zones | ~500 |
| Write-gate weights | ~340 per memory circuit × 3 | ~1,020 |
| Write-gate biases | 1 per memory circuit × 3 | 3 |
| Adapter weights (formation) | 64 × (input_dim + output_dim + 1) | ~8K-20K |
| Adapter weights (well) | 64 × (input_dim + output_dim + 1) | ~8K-20K |
| **Total** | | **~520K-1.05M** |

This is a small model. It fits comfortably in CPU cache. No GPU needed for training.

---

## 3. Decision: Manual BPTT in Rust

### Why not PyTorch?

- The model is tiny (~1M parameters). PyTorch's overhead buys us nothing.
- The architecture is custom (liquid neurons with asynchronous multi-timescale circuits). Mapping it to PyTorch tensors would be more work than writing the gradients by hand.
- Production inference is in Rust. A Python training loop means maintaining two codebases with subtle numerical divergence.
- Online learning (adapter training during drilling) must run in Rust anyway. Writing offline training in Rust means the online path is just a subset of the same code.

### Why manual BPTT is tractable

The gradient through one liquid neuron step is simple:

```
Forward:
  s_t     = Σ(w_i · x_i_t) + b
  x_{t+1} = x_t + (-x_t + tanh(s_t)) / tau · dt

Backward:
  ∂x_{t+1}/∂x_t  = 1 - dt/tau                          (leak factor)
  ∂x_{t+1}/∂s_t  = (1 - tanh²(s_t)) · dt / tau         (tanh derivative scaled)
  ∂s_t/∂w_i      = x_i_t                                (source neuron activation)
  ∂s_t/∂b        = 1
  ∂s_t/∂x_i_t    = w_i                                  (for upstream gradient)
```

Memory for BPTT: 4,736 neurons × 100 steps × 4 bytes = **1.8 MB per sample**. Trivial.

The MLP gates (6→8→2) and write gates (linear→sigmoid) are textbook backprop.

### What we need to implement

1. **Forward pass with recording** — store intermediate states (`x_t`, `s_t` pre-activation) at each step
2. **Loss computation** — compare circuit output neurons against labels
3. **Backward pass** — BPTT through 100 steps, accumulating gradients for weights, biases, masks, and gate parameters
4. **Optimizer** — Adam (per-parameter momentum and RMS tracking)
5. **Gradient clipping** — clip global norm to prevent exploding gradients through 100-step unrolling

---

## 4. Connection Initialization

**Current gap:** All neurons have `connections = []`. Before training can begin, neurons need sparse random connectivity.

### Strategy: Structured Random Initialization

```
For each circuit:
  For each neuron in circuit.neuron_indices:
    1. Determine target density based on circuit type:
       - Detection:  20% of circuit size
       - Causation:  30% of circuit size
       - Memory:     15% of circuit size
       - Prediction: 25% of circuit size

    2. Sample source indices:
       - 70% from same circuit (local connections)
       - 30% from any neuron in the mesh (global connections, enabling cross-circuit influence)
       - Exclude self-connections

    3. Initialize weights:
       - Xavier/Glorot: w ~ N(0, sqrt(2 / (fan_in + fan_out)))
       - fan_in = number of incoming connections
       - fan_out = average outgoing connections (estimated from density)
```

For detection circuits specifically: 100% of connections should be from within the circuit, since detection should be self-contained initially. Cross-circuit influence comes through the overlap zones.

### Adapter Initialization

The formation and well adapters need their `input_from` and `output_to` wired up:

```
Formation adapter (64 neurons):
  input_from:  Last 32 neurons of each causation circuit (96 total)
  output_to:   First 32 neurons of each prediction circuit (64 total)

Well adapter (64 neurons):
  input_from:  Last 32 neurons of each detection circuit (160 total)
  output_to:   First 32 neurons of each memory circuit (96 total)
```

Adapter weights initialized with Xavier, scaled by 0.1 (small initial influence).

---

## 5. Training Data

### 5.1 What We Need

| Data Type | Format | Purpose |
|---|---|---|
| Normal drilling | `Vec<WitsSnapshot>` time series | Baseline behaviour, negative examples |
| Kick events | `Vec<WitsSnapshot>` + label + onset index | Positive examples for kick detection |
| Loss events | Same | Positive examples for loss detection |
| Pack-off events | Same | Positive examples for pack-off detection |
| Stick-slip events | Same | Positive examples for stick-slip detection |
| Founder events | Same | Positive examples for founder detection |
| Causal annotations | `HashMap<CircuitId, Vec<f32>>` per sample | Which parameters caused each anomaly |
| Formation labels | `String` per well section | For formation adapter training |

A "training sequence" is a window of consecutive WITS snapshots (e.g., 300 samples = 5 minutes at 1 Hz) with optional anomaly labels at specific indices.

### 5.2 Synthetic Data Generator

For initial training before real field data is available, build a synthetic WITS generator:

```rust
pub struct SyntheticWellGenerator {
    pub base_params: WitsSnapshot,     // Typical operating point
    pub noise_scale: f32,              // Gaussian noise amplitude (0.01-0.05)
    pub trend_rate: f32,               // Slow parameter drift
    pub rng: StdRng,
}

impl SyntheticWellGenerator {
    /// Generate normal drilling sequence (no anomalies)
    pub fn normal_sequence(&mut self, length: usize) -> Vec<WitsSnapshot>;

    /// Inject a kick signature starting at onset_idx
    /// Pattern: flow_out rises 15-50 gpm over 10-30 samples,
    ///          pit_volume rises, total_gas may rise, spp may drop
    pub fn inject_kick(&self, sequence: &mut [WitsSnapshot], onset_idx: usize, severity: f32);

    /// Inject a loss signature
    /// Pattern: flow_out drops, pit_volume drops, spp may rise
    pub fn inject_loss(&self, sequence: &mut [WitsSnapshot], onset_idx: usize, severity: f32);

    /// Inject pack-off signature
    /// Pattern: torque rises sharply, spp rises, rop drops toward 0
    pub fn inject_packoff(&self, sequence: &mut [WitsSnapshot], onset_idx: usize, severity: f32);

    /// Inject stick-slip signature
    /// Pattern: torque oscillates (CV > 12%), rpm oscillates
    pub fn inject_stickslip(&self, sequence: &mut [WitsSnapshot], onset_idx: usize, severity: f32);

    /// Inject founder signature
    /// Pattern: wob increases but rop plateaus or decreases
    pub fn inject_founder(&self, sequence: &mut [WitsSnapshot], onset_idx: usize, severity: f32);
}
```

**Anomaly signatures** (physics-grounded):

| Anomaly | Primary Signal | Secondary Signals | Onset Speed |
|---|---|---|---|
| Kick | flow_out >> flow_in (+15-50 gpm) | pit_volume rising, total_gas rising, spp dropping | 10-60s |
| Loss | flow_out << flow_in (-10-40 gpm) | pit_volume dropping, spp rising | 5-30s |
| Pack-off | torque +30-100%, spp +20-50% | rop dropping toward 0, hook_load changing | 5-15s |
| Stick-slip | torque CV > 12%, rpm oscillation | irregular rop | Immediate |
| Founder | wob rising, rop flat/falling | mse rising sharply, d_exponent changing | 30-120s |

### 5.3 Labelling Format

```rust
pub struct TrainingSequence {
    pub wits_samples: Vec<WitsSnapshot>,
    pub labels: Vec<SequenceLabel>,
    pub formation: Option<String>,
    pub well_id: String,
}

pub struct SequenceLabel {
    pub onset_idx: usize,
    pub end_idx: usize,
    pub anomaly_type: AnomalyType,
    pub severity: f32,         // 0.0-1.0
    pub causal_params: Vec<CausalParam>,
}

pub enum AnomalyType {
    Kick, Loss, PackOff, StickSlip, Founder, Normal,
}

pub enum CausalParam {
    FlowBalance, PitVolume, Torque, SPP, WOB, ROP, TotalGas, RPM,
}
```

---

## 6. Loss Functions

### 6.1 Detection Loss (Binary Cross-Entropy per Circuit)

Each detection circuit produces a confidence value (mean |x| of its neurons). We compare this against a binary label.

```
For each detection circuit c:
  y_pred = circuit_confidence(c)          // mean(|x_i|) for i in c.neuron_indices
  y_true = 1.0 if anomaly active, 0.0 otherwise

  L_detection(c) = -[y_true · log(y_pred + ε) + (1 - y_true) · log(1 - y_pred + ε)]
```

**Problem:** `mean(|x_i|)` is not a great differentiable readout. The gradient of `|x|` has a discontinuity at 0.

**Solution:** Use a dedicated readout from the last N neurons of each detection circuit instead:

```
For each detection circuit c with neuron_indices [n_0, n_1, ..., n_k]:
  readout_neurons = [n_{k-8}, ..., n_k]   (last 8 neurons)
  y_pred = sigmoid(mean(readout_neurons.x))
```

This gives a smooth, differentiable readout that maps to (0, 1).

### 6.2 Causation Loss (Multi-Label BCE)

Each causation circuit should activate when its associated parameters are causal. We read the last 8 neurons of each causation circuit and compare against the causal parameter labels.

```
For causation circuit c with 8 readout neurons:
  For each readout neuron i (mapped to a parameter: torque, pressure, flow, etc.):
    y_pred_i = sigmoid(x_i)
    y_true_i = 1.0 if parameter i is causal, 0.0 otherwise
    L_causation(c, i) = BCE(y_pred_i, y_true_i)
```

### 6.3 Prediction Loss (MSE)

Prediction circuits should forecast future drilling state. We compare readout neurons against actual future WITS values.

```
For prediction circuit c:
  readout_neurons = last 14 neurons (one per encoded WITS parameter)
  y_pred = [x_i for i in readout_neurons]
  y_true = encoded WITS values at t + horizon

  L_prediction(c) = mean((y_pred - y_true)²)
```

Horizon: state_prediction uses t+5s, rop_prediction uses t+30s.

### 6.4 Physics Consistency Loss

The mesh must not contradict deterministic physics. Compare mesh-derived values against physics module outputs.

```
L_physics = |mesh_flow_balance_prediction - compute_flow_balance(wits)|²
          + |mesh_mse_trend - actual_mse_trend|²
```

This is a regularizer, not a primary loss. Weight: 0.05.

### 6.5 Overlap Sparsity Loss (Stage 1 & 3)

```
L_sparsity = λ · Σ|mask_i| for all overlap masks
```

λ = 0.01. Encourages masks to be sparse (most neurons not shared).

### 6.6 Overlap Stability Loss (Stage 3 only)

```
L_stability = λ_overlap · Σ(mask_current_i - mask_after_stage1_i)²
```

λ_overlap = 0.1. Prevents Stage 3 fine-tuning from destroying the overlap structure learned in Stage 1.

### 6.7 Mutual Information Loss (Stage 1)

For overlap pre-training, we want shared neurons to carry information useful to both circuits. Approximate via correlation maximization:

```
For overlap zone z between circuits a and b:
  output_a = mean activations of circuit a's non-shared neurons
  output_b = mean activations of circuit b's non-shared neurons
  shared   = activations of shared neurons

  L_MI(z) = -correlation(shared, output_a) - correlation(shared, output_b) + L_sparsity(z)
```

This encourages shared neurons to be predictive of both circuits' outputs while remaining sparse.

### 6.8 Combined Loss

```
Stage 1:  L = L_MI + 0.01 · L_sparsity
Stage 2:  L = L_detection + L_causation + L_prediction + 0.05 · L_physics
Stage 3:  L = L_detection + L_causation + L_prediction + 0.05 · L_physics
            + 0.1 · L_stability + 0.01 · L_sparsity
Online:   L = L_detection + L_causation  (adapters only, prediction is secondary)
```

---

## 7. Backpropagation Through the Mesh

### 7.1 Forward Pass with State Recording

For each training sample (a window of WITS snapshots):

```
record = []  // stores (step, neuron_states, pre_activations) for BPTT

for each wits_sample in window:
    encode_input(mesh, wits_sample, baselines)

    for step in 0..100:
        // Record current state before update
        record.push(snapshot of all neuron x values and pre-activation sums)

        // Standard mesh step (Phase 1 + Phase 2)
        mesh.step()

    // After 100 steps, read output neurons and compute per-sample loss
```

### 7.2 Backward Pass (BPTT)

After the forward pass, backpropagate through the recorded states in reverse:

```
// Initialize gradient accumulators
dL_dw[i][j] = 0  for all connection weights
dL_db[i] = 0     for all biases
dL_dx[i] = ∂L/∂x_i  for output neurons (from loss function)

// Walk backward through all recorded steps
for step in (0..total_steps).rev():
    for each neuron i that was updated at this step:
        // Which circuit updated this neuron?
        tau = circuit.tau

        // Retrieve recorded pre-activation
        s_i = recorded_pre_activation[step][i]
        tanh_deriv = 1.0 - tanh(s_i)²

        // Gradient through the liquid dynamics
        // x_{t+1} = x_t + (-x_t + tanh(s_t)) / tau * dt
        dx_ds = tanh_deriv * dt / tau
        dx_dx_prev = 1.0 - dt / tau

        // Accumulate weight gradients
        for (j, w_ij) in neuron_i.connections:
            dL_dw[i][j] += dL_dx[i] * dx_ds * x_j_at_step[step]
            dL_dx[j]    += dL_dx[i] * dx_ds * w_ij  // upstream gradient

        // Bias gradient
        dL_db[i] += dL_dx[i] * dx_ds

        // Propagate gradient to previous timestep
        dL_dx[i] = dL_dx[i] * dx_dx_prev

    // Handle overlap merge gradients
    for each overlap zone:
        // If safety override was active: gradient flows only to winning circuit
        // If gate merge was active: gradient flows through gate softmax
        backprop_through_gate(zone, recorded_gate_inputs[step])
```

### 7.3 Overlap Gate Backward

The gate is a 6→8→2 MLP with ReLU + softmax. Standard MLP backprop:

```
// Forward was: input → hidden (ReLU) → logits → softmax → priorities
// priorities used as: delta_merged = p_a * delta_a + p_b * delta_b

// Given dL/d(delta_merged):
dL_dp_a = dL_d_merged * delta_a
dL_dp_b = dL_d_merged * delta_b

// Softmax backward:
dL_dlogit_a = p_a * (dL_dp_a - (dL_dp_a * p_a + dL_dp_b * p_b))
dL_dlogit_b = p_b * (dL_dp_b - (dL_dp_a * p_a + dL_dp_b * p_b))

// Then standard dense layer backward through weights_ho, ReLU, weights_ih
```

### 7.4 Write-Gate Backward

```
// Forward: gate = sigmoid(dot(w, causation_output) + b)
// Used as: neuron_update *= gate

// Given dL/d(gated_update):
dL_d_gate = dL_d_gated_update * ungated_update
sigmoid_deriv = gate * (1.0 - gate)
dL_d_preact = dL_d_gate * sigmoid_deriv

dL_dw[k] += dL_d_preact * causation_output[k]
dL_db    += dL_d_preact
```

### 7.5 Mask Backward

```
// Masks scale the contribution of shared neurons
// During forward: effective_delta = mask * delta
// Plus L1 penalty: L_sparsity = λ * Σ|mask_i|

dL_d_mask[i] = dL_d_effective_delta * delta_i + λ * sign(mask_i)
```

---

## 8. Optimizer: Adam

Use Adam with per-parameter state. Small model, so memory is not a concern.

```rust
pub struct AdamOptimizer {
    pub lr: f32,
    pub beta1: f32,      // 0.9
    pub beta2: f32,      // 0.999
    pub epsilon: f32,    // 1e-8
    pub t: u64,          // step counter
    pub m: Vec<f32>,     // first moment (one per parameter)
    pub v: Vec<f32>,     // second moment (one per parameter)
}
```

Gradient clipping: clip global gradient norm to 1.0 before applying Adam update. This prevents exploding gradients from BPTT through 100+ steps.

---

## 9. The Three Training Stages

### Stage 1: Overlap Pre-Training

**Goal:** Discover how much overlap each zone needs. Learn gate weights so circuits can merge coherently.

**What trains:** Only overlap masks and gate weights (370 + 500 = 870 parameters).

**What's frozen:** All neuron weights, biases, write gates.

**Requires:** Neurons must have initialized connections first (see Section 4), but their weights are frozen. We're only learning the communication protocol, not the neurons themselves.

**Data:** Unlabelled normal drilling sequences. We're not detecting anomalies yet — just learning which neurons should be shared and how to merge updates.

**Procedure:**
```
1. Initialize connections for all neurons (Section 4)
2. Initialize all neuron weights with Xavier
3. For step in 0..5,000:
   a. Sample a batch of normal WITS sequences (batch_size=32, window=50 samples)
   b. Forward pass: encode + 100 mesh steps per WITS sample
   c. Compute L_MI for each overlap zone
   d. Backward pass: compute gradients for masks and gate weights only
   e. Adam update (lr=1e-3)
   f. Clamp masks to [0, 1]
4. Save mask values as mask_stage1 (for stability loss in Stage 3)
5. Prune neurons with mask < 0.1 (soft prune — set mask to 0, don't remove)
```

**Expected outcome:** Masks converge to bimodal distribution — most near 0 (not shared) or near 1 (actively shared). Gate weights learn to balance circuit priorities based on confidence.

### Stage 2: Circuit Training with Alternating Freezes

**Goal:** Train the actual neuron weights to detect anomalies and reason about causes.

**What trains:** Neuron weights and biases within unfrozen circuits. Write-gate weights for memory circuits.

**What's frozen:** Overlap masks and gates (learned in Stage 1). Whichever circuit group is frozen in the current round.

**Data:** Labelled drilling sequences with anomaly annotations.

**Procedure:**
```
For round in 0..5:
    // Round A: Train causation, freeze detection
    Freeze all detection circuits
    Unfreeze all causation circuits
    For step in 0..500:
        Sample batch with anomaly labels and causal annotations
        Forward pass through full mesh (detection fires but weights frozen)
        L = L_causation + 0.05 * L_physics
        Backward: accumulate gradients for causation neuron weights only
        Adam update (lr=1e-3)

    // Round B: Train detection, freeze causation
    Freeze all causation circuits
    Unfreeze all detection circuits
    For step in 0..500:
        Sample batch with anomaly labels
        Forward pass through full mesh
        L = L_detection + 0.05 * L_physics
        Backward: accumulate gradients for detection neuron weights only
        Adam update (lr=1e-3)

    // Round C: Train memory
    Freeze detection + causation
    Unfreeze memory circuits + write gates
    For step in 0..200:
        Sample batch with longer sequences (capture temporal patterns)
        Forward pass
        L = L_detection (memory should help detection accuracy)
        Backward: gradients for memory neuron weights + write-gate weights
        Adam update (lr=5e-4)

    // Round D: Train prediction
    Freeze everything except prediction circuits
    For step in 0..200:
        Sample batch with future WITS values as targets
        Forward pass
        L = L_prediction
        Backward: gradients for prediction neuron weights
        Adam update (lr=5e-4)
```

**Why alternating?** Detection and causation circuits share overlap neurons. Training both simultaneously creates a moving-target problem where overlap neurons receive conflicting gradients. Alternating gives each circuit group a stable target to learn against.

### Stage 3: Joint Fine-Tuning

**Goal:** Let everything adapt together with small learning rate. Recover any performance lost from the alternating freeze schedule.

**What trains:** Everything (neuron weights, biases, overlap masks, gate weights, write gates).

**Data:** Full labelled dataset.

**Procedure:**
```
For step in 0..2,000:
    Sample batch with full labels (anomaly + causal + future state)
    Forward pass through full mesh
    L = L_detection + L_causation + L_prediction
      + 0.05 * L_physics
      + 0.1  * L_stability      // keep masks near Stage 1 values
      + 0.01 * L_sparsity       // keep masks sparse
    Backward: gradients for all parameters
    Adam update (lr=1e-4)        // small LR to avoid catastrophic changes
```

### Post-Training

```
1. Hard prune: remove overlap neurons with mask < 0.3
   - Remove from shared_neuron_indices
   - Remove connections to/from pruned neurons
2. Freeze universal circuits:
   for circuit in [all detection, causation, base prediction]:
       circuit.frozen = true
3. Freeze overlap zones (masks and gates)
4. Validate topology (max 2 circuits per neuron still holds after pruning)
5. Run validation suite:
   - Detection accuracy > 95% on held-out test set
   - False positive rate < 5%
   - Causation accuracy > 85%
   - No NaN/Inf in any neuron state after 100,000 steps
6. Populate replay buffer with 500 representative samples from training data
7. Serialize frozen mesh to disk
```

---

## 10. Online Learning (During Drilling)

Only adapter layers train online. The universal core is frozen.

### When to Train

- Every WITS sample in Shadow mode (learning baselines + adapting)
- Every WITS sample in Advisory mode (refining)
- Every 10th sample in Production mode (maintenance learning, reduce compute)

### Online Training Step

```
fn online_train_step(mesh: &mut ProductionMesh, wits: &WitsSnapshot, label: Option<&SequenceLabel>):
    // 1. Forward pass through frozen universal mesh (already done in process())
    // 2. Forward pass through adapters
    formation_adapter.forward(universal_outputs)
    well_adapter.forward(universal_outputs)

    // 3. If we have a label (from driller confirmation or automated detection):
    if let Some(label) = label:
        // Compute adapter loss
        L = L_detection_adapter + L_causation_adapter

        // Backward through adapters only (universal mesh is frozen, no gradients needed)
        compute_adapter_gradients(L)

        // Mix with replay
        replay_batch = replay_buffer.sample(batch_size=1)
        for replay_sample in replay_batch:
            L_replay = forward_and_loss(replay_sample)
            accumulate_gradients(L_replay, scale=0.1)

        // Adam update
        adam_update(formation_adapter, lr=1e-4)
        adam_update(well_adapter, lr=1e-4)

    // 4. Maybe store in replay buffer
    if should_store(wits, label):
        replay_buffer.push(TrainingSample::from(wits, label))
```

### Replay Strategy

The replay buffer prevents catastrophic forgetting when adapting to a new formation:

- 90% of each mini-batch is the live sample
- 10% is randomly sampled from the replay buffer
- Buffer stores 500 samples maximum (FIFO eviction)
- Oversample anomaly samples 3x vs normal samples when adding to buffer

---

## 11. Training Data Volume Estimates

| Stage | Samples Needed | Rationale |
|---|---|---|
| Stage 1 (overlap) | 5,000 × 32 = 160K WITS samples | Unsupervised, just needs normal drilling data |
| Stage 2 (circuits) | 5 rounds × 1,400 steps × 32 = 224K samples | Needs labelled anomalies, ~10% should be anomalous |
| Stage 3 (fine-tune) | 2,000 × 32 = 64K samples | Full labels required |
| **Total** | **~450K WITS samples** | **~5 days of 1 Hz data, or synthetic** |

For synthetic data, generate 100 wells × 5,000 samples each = 500K samples. Inject anomalies at random intervals with random severity.

---

## 12. Validation Protocol

### Hold-Out Split

- Train: 70%
- Validation: 15% (for early stopping / hyperparameter tuning)
- Test: 15% (final evaluation only)

Split by well, not by sample. No well appears in both train and test.

### Metrics

| Metric | Target | Measured On |
|---|---|---|
| Detection accuracy | > 95% | Test set, per anomaly type |
| Detection precision | > 90% | Test set (few false positives) |
| Detection recall | > 95% | Test set (miss nothing) |
| Detection latency | < 5 samples (5s) from onset | Test set, median |
| Causation accuracy | > 85% | Test set, per causal parameter |
| Prediction MSE | < 0.1 (normalised) | Test set, state prediction at t+5s |
| False positive rate | < 5% over normal drilling | Normal-only test sequences |
| Inference time | < 1ms per mesh step | Benchmark, CPU |
| No NaN/Inf | 0 occurrences | 1M step stress test |

### Anomaly-Specific Detection Thresholds

After training, determine per-circuit confidence thresholds using the validation set:

```
For each detection circuit:
    Sweep threshold from 0.1 to 0.9 in steps of 0.01
    Find threshold that maximizes F1 score on validation set
    Store as circuit.detection_threshold
```

These thresholds are fixed after validation and used in production.

---

## 13. Implementation Order

### Phase 6a: Training Infrastructure

1. **`src/training/mod.rs`** — module declarations, `TrainingConfig`, `TrainingMetrics`
2. **`src/training/state.rs`** — `ForwardRecord` (stores intermediate states for BPTT), `GradientAccumulator` (per-parameter gradient storage)
3. **`src/training/backward.rs`** — `backprop_neuron_step()`, `backprop_gate()`, `backprop_write_gate()`, `backprop_mask()`
4. **`src/training/optimizer.rs`** — `AdamOptimizer` with per-parameter state, gradient clipping
5. **`src/training/loss.rs`** — all loss functions from Section 6

### Phase 6b: Data Pipeline

6. **`src/training/synthetic.rs`** — `SyntheticWellGenerator` with anomaly injection
7. **`src/training/data.rs`** — `TrainingSequence`, `SequenceLabel`, `DataLoader` (batching, shuffling)

### Phase 6c: Connection Initialization

8. **`src/training/init.rs`** — `initialize_connections()` (structured random connectivity per Section 4), `initialize_adapters()` (wire up adapter input/output indices)

### Phase 6d: Training Stages

9. **`src/training/overlap_pretrain.rs`** — Stage 1 loop
10. **`src/training/alternating.rs`** — Stage 2 loop with freeze/unfreeze
11. **`src/training/joint.rs`** — Stage 3 loop
12. **`src/training/freeze.rs`** — post-training freeze, prune, validate
13. **`src/training/online.rs`** — online adapter training step

### Phase 6e: Training Entrypoint

14. **`src/training/pipeline.rs`** — `run_full_training()` orchestrating Stages 1→2→3→freeze
15. **Binary: `src/bin/train.rs`** — CLI entrypoint for offline training

### Phase 6f: Tests

16. Unit tests for each backward function (gradient checking via finite differences)
17. Integration test: full Stage 1→2→3 on tiny synthetic data (32 neurons, 3 circuits)
18. Smoke test: loss decreases over 100 steps on synthetic data

---

## 14. Gradient Checking

Every backward function must be validated with numerical gradient checking before being trusted:

```rust
fn gradient_check(f: impl Fn(f32) -> f32, x: f32, analytic_grad: f32) -> bool {
    let eps = 1e-4;
    let numerical_grad = (f(x + eps) - f(x - eps)) / (2.0 * eps);
    let relative_error = (analytic_grad - numerical_grad).abs()
        / (analytic_grad.abs() + numerical_grad.abs() + 1e-8);
    relative_error < 1e-3  // must agree to 0.1%
}
```

Check gradients for:
- `compute_neuron_update` backward (weight, bias, upstream)
- Overlap gate backward (all 74 parameters)
- Write-gate backward
- Mask backward
- Full BPTT through 10 steps (small mesh)

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Exploding gradients through 100-step BPTT | Training diverges | Gradient clipping (norm=1.0), careful initialization |
| Vanishing gradients through 100-step BPTT | Slow/no learning for early steps | The leak factor `1 - dt/tau` is close to 1.0 for small dt, so gradients propagate well. Skip connections through overlap zones also help. |
| Overlap masks all go to 0 | No inter-circuit communication | L_MI loss in Stage 1 actively encourages non-zero masks. Monitor mask statistics during training. |
| Overlap masks all go to 1 | No specialization, circuits blur together | L_sparsity penalizes high masks. Target is 8-25% overlap per zone. |
| Detection circuits learn to always fire | 100% recall, 0% precision | Balance training data (50/50 normal/anomalous). Use precision-recall-based early stopping. |
| Memory circuits don't learn useful patterns | No benefit from history | Auxiliary loss: detection accuracy should improve when memory circuits are enabled vs disabled. If not, memory is dead weight. |
| Synthetic data doesn't transfer to real drilling | Model fails on deployment | Design synthetic generator from physics first principles. Validate on any available real data before deployment. Shadow mode provides a safety net. |
| Manual BPTT has bugs | Incorrect gradients, silent failures | Gradient checking (Section 14) for every backward function. |

---

## 16. Summary

```
┌─────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                       │
│                                                         │
│  ┌──────────┐    Initialize connections (Xavier)         │
│  │  INIT    │    Wire up adapter I/O                     │
│  └────┬─────┘    Generate synthetic training data        │
│       │                                                  │
│  ┌────▼─────┐    Train overlap masks + gates             │
│  │ STAGE 1  │    Loss: mutual information + sparsity     │
│  │ 5K steps │    Only 870 parameters train               │
│  └────┬─────┘    Save mask_stage1 snapshot               │
│       │                                                  │
│  ┌────▼─────┐    Alternating freeze:                     │
│  │ STAGE 2  │    Detection ↔ Causation (5 rounds)        │
│  │ 7K steps │    Then Memory, then Prediction            │
│  └────┬─────┘    Overlap zones frozen                    │
│       │                                                  │
│  ┌────▼─────┐    Joint fine-tuning                       │
│  │ STAGE 3  │    Everything unfrozen, small LR           │
│  │ 2K steps │    + stability + sparsity regularization   │
│  └────┬─────┘                                            │
│       │                                                  │
│  ┌────▼─────┐    Prune masks < 0.3                       │
│  │  FREEZE  │    Freeze universal circuits + overlaps    │
│  └────┬─────┘    Validate, populate replay buffer        │
│       │                                                  │
│  ┌────▼─────┐    Adapters only, 90/10 live/replay        │
│  │  ONLINE  │    lr=1e-4, runs during drilling           │
│  └──────────┘                                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Total offline training:** ~14,000 gradient steps. At ~10ms per step (small model, CPU), that's ~2-3 minutes wall clock. The bottleneck is data generation, not compute.

**Implementation effort:** ~2,500-3,500 lines of Rust across 15 files. The hardest part is getting BPTT right — gradient checking is essential.
