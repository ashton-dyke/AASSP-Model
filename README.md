# AASSP-Model

**Autonomous Anomaly Sensing & Safety Protocol — Neural Mesh**

An overlapping neural mesh built on Liquid Time-Constant (LTC) neurons with Closed-form Continuous-depth (CfC) evaluation. Multiple neural circuits share neurons through overlap zones and communicate through shared state. Designed for real-time drilling anomaly detection.

## What It Does

Ingests real-time WITS Level 0 drilling data and produces:

- **Anomaly detection** — kicks, losses, pack-off, stick-slip, founder conditions
- **Causal reasoning** about root causes (torque, pressure, flow)
- **Predictive state projection** — forward-looking drilling state and ROP trajectory
- **Memory-augmented pattern recognition** from historical events via episodic recall

Trained on synthetic drilling data (96% detection accuracy, zero false positives on normal data), then deployed for inference on real well data.

## Design Principles

1. **Physics is ground truth.** Every neural output is validated against deterministic physics calculations (MSE, ECD, d-exponent). The mesh augments physics, never overrides it.
2. **Synchronous CfC dynamics.** All neurons update simultaneously using the closed-form CfC solution. Input neurons are clamped (skip CfC) to preserve encoded WITS values.
3. **Max 2 circuits per neuron.** Hard architectural constraint. If three circuits need to communicate, chain the overlaps (A-B-C), never share a single neuron across 3+ circuits.
4. **Frozen universal core.** Detection and causation circuits are frozen after pre-training. Online adaptation happens only through thin adapter layers.
5. **Fail gracefully.** If the mesh fails, fall back to physics-only advisories. The system must never be worse than no system.

## Architecture

The mesh uses a synchronous two-phase CfC update:

1. **Phase 1 — Parallel CfC Forward:** All neurons compute new states from the current snapshot using the closed-form CfC solution (parallelized with rayon). Clamped input neurons retain their encoded values.
2. **Phase 2 — Atomic Write-back:** New states are applied atomically, then circuit confidences are updated.

### Default Layout (4,736 neurons, 13 circuits)

```
DETECTION BLOCK   (neurons 0..1023)     5 circuits, tau 0.10-0.15s, Safety
CAUSATION BLOCK   (neurons 1024..2047)  3 circuits, tau 0.15s,      Safety
MEMORY BLOCK      (neurons 2048..3583)  3 circuits, tau 0.2-30s,    Performance
PREDICTION BLOCK  (neurons 3584..4607)  2 circuits, tau 0.10-0.12s, Performance
ADAPTER NEURONS   (neurons 4608..4735)  128 neurons (64 formation + 64 well)
```

### Three-Tier Production Architecture

| Tier | What | Trainable? |
|---|---|---|
| Universal | Detection + causation + base prediction circuits | Frozen after pre-training |
| Adapters | 64-neuron formation adapter + 64-neuron well adapter | Plastic (online) |
| Adaptive | Well-specific circuits connected through adapters | Plastic (online) |

### Three-Stage Training Pipeline

1. **Stage 1 — Overlap pre-training:** Trains overlap masks and gate weights using mutual information loss with cosine LR scheduling.
2. **Stage 2 — Alternating freeze:** Cycles through circuit groups (detection, causation, memory, prediction), training each while freezing the others. Uses a shared replay buffer to mitigate catastrophic forgetting.
3. **Stage 3 — Joint fine-tuning:** All circuits unfrozen, mini-batch gradient accumulation with AdamW and stability/sparsity regularization.

### CfC Signal Propagation

Each neuron follows the closed-form dynamics:

```
f = sigmoid(gate_weights . x + gate_bias)       -- input-dependent gate
A = tanh(drive_weights . x + bias)               -- drive signal
alpha = 1/tau + f                                 -- effective decay rate
x(t+dt) = x(t) * exp(-alpha*dt) + (f*A/alpha) * (1 - exp(-alpha*dt))
```

Skip connections from input to readout neurons bypass multi-hop CfC attenuation. Readout gain of `2 * (1/tau + 0.5)` compensates for CfC steady-state amplitude reduction. Positive class weighting (3x) in detection loss prevents degenerate "suppress everything" solutions.

## Crate Structure

```
src/
├── lib.rs                    # Public API, re-exports
├── neuron.rs                 # LtcNeuron, CfC forward (full + inference)
├── circuit.rs                # Circuit, CircuitType, CircuitCriticality
├── overlap.rs                # OverlapZone, sparsity loss
├── topology.rs               # TopologyConstraint (max 2 circuits/neuron)
├── mesh.rs                   # OverlappingMesh, parallel step()
├── config.rs                 # MeshConfig, DegradationMode
├── layout.rs                 # Default 4,736-neuron layout
├── physics.rs                # MSE, d-exponent, ECD, flow balance
├── memory/
│   ├── hierarchical.rs       # HierarchicalMemory, WriteGate
│   └── episodic.rs           # EpisodicMemory, KNN recall, pruning
├── io/
│   ├── wits.rs               # WitsSnapshot (21 parameters)
│   ├── encoding.rs           # Baselines (Welford's), normalisation, input encoding
│   └── output.rs             # MeshOutput, Detection, CausalAnalysis, RiskLevel
├── training/
│   ├── pipeline.rs           # Three-stage training orchestrator
│   ├── init.rs               # Connection initialization, skip connections, gate bias
│   ├── loss.rs               # Detection/causation/prediction loss, readout gain
│   ├── backward.rs           # Backpropagation through time (BPTT)
│   ├── optimizer.rs          # AdamW optimizer, cosine LR with warmup
│   ├── state.rs              # ForwardRecord, GradientAccumulator
│   ├── data.rs               # Label generation, dataset shuffling
│   ├── synthetic.rs          # SyntheticWellGenerator (normal, kick, loss, packoff sequences)
│   └── online.rs             # Online adapter training
├── adaptation/
│   ├── adapter.rs            # AdapterLayer, AdapterNeuron
│   ├── replay.rs             # ReplayBuffer, TrainingSample
│   └── production.rs         # ProductionMesh (three-tier architecture)
└── bin/
    ├── diagnose.rs           # Post-training readout analysis
    └── infer_csv.rs          # Run inference on real WITS CSV data
```

## Quick Start

### Library Usage

```rust
use sairen_mesh::layout::build_default_mesh;
use sairen_mesh::adaptation::production::ProductionMesh;
use sairen_mesh::training::pipeline::{run_full_training, TrainingConfig};
use sairen_mesh::io::wits::WitsSnapshot;

// Build and train.
let mut mesh = build_default_mesh();
let metrics = run_full_training(&mut mesh, &TrainingConfig::default());

// Wrap in ProductionMesh for inference.
let mut prod = ProductionMesh::from_trained(mesh);

// Process WITS samples.
let wits = WitsSnapshot::zeros();
let output = prod.process(&wits);
// output.detections, output.risk_level, output.causal_analyses, ...
```

### Run Inference on Real CSV Data

```bash
cargo run --release --bin infer_csv -- dataset/F-5_witsml.csv
```

Parses metric-unit WITS CSV, converts to imperial, computes derived physics (MSE, d-exponent, ECD), trains on synthetic data, then runs inference over all samples. Prints detections with timestamps, rig mode, anomaly type, and confidence.

### Diagnostic Tool

```bash
cargo run --release --bin diagnose
```

Trains the model and prints per-circuit readout activations for normal and kick sequences.

## Building

```bash
cargo build --release
```

## Testing

```bash
# Unit tests (116 tests, ~3s release)
cargo test --release

# Full integration test (~32s release)
cargo test --release end_to_end -- --ignored
```

116 tests covering:

- **Neuron dynamics** — CfC convergence, tanh bounds, gate modulation, steady-state behavior
- **Circuit management** — creation, confidence update, type filtering
- **Topology enforcement** — rejects 3+ circuits per neuron, validates bounds
- **Overlap zones** — mask filtering, sparsity loss
- **Memory** — hierarchical decay rates, write-gate blocking, episodic KNN recall, pruning
- **Input encoding** — Welford's statistics, normalisation, clamping to [-1, 1]
- **Physics** — MSE, ECD, d-exponent, flow balance
- **Training** — loss decreases, gradient checks (weight, bias, gate, tau), BPTT, optimizer
- **Adaptation** — adapter forward pass, replay buffer proportions, production mesh pipeline
- **Layout** — 4,736 neurons, 13 circuits, correct circuit counts per type
- **Integration** — full pipeline smoke test, A/B training comparison

## Dependencies

| Crate | Purpose |
|---|---|
| `rayon` | Parallel CfC neuron updates in mesh step |
| `rand` | Synthetic data generation, training sampling |
| `serde` / `serde_json` | Serialisation |
| `log` | Logging |
| `thiserror` | Error types |

## Status

| Phase | Status |
|---|---|
| 1: Foundation (LTC/CfC neurons, circuits, topology) | Done |
| 2: Overlap system (zones, masks, sparsity) | Done |
| 3: Memory (hierarchical + episodic) | Done |
| 4: I/O (WITS encoding, output decoding) | Done |
| 5: Adaptation (adapters, replay, ProductionMesh) | Done |
| 6: Training pipeline (BPTT, 3-stage, AdamW) | Done |
| 7: Real-data inference (CSV parsing, unit conversion) | Done |

## License

SAIREN Ltd. All rights reserved.
