# AASSP-Model

**SAIREN-OS Asynchronous Overlapping Neural Mesh**

An overlapping liquid neural network mesh implementing Asymmetric Asynchronous Shared-State Parallelism (AASSP). Multiple neural circuits share neurons through overlap zones, process at different timescales, and communicate through shared state rather than message passing.

Built for real-time drilling anomaly detection on edge hardware (RTX 4060 Ti, 16 GB VRAM).

## What It Does

Ingests real-time WITS Level 0 drilling data and produces:

- **Sub-millisecond anomaly detection** — kicks, losses, pack-off, stick-slip, founder
- **Causal reasoning** about root causes (torque, pressure, flow)
- **Predictive state projection** — forward-looking drilling state and ROP trajectory
- **Memory-augmented pattern recognition** from historical events via episodic recall

## Design Principles

1. **Physics is ground truth.** Every neural output is validated against deterministic physics calculations (MSE, ECD, d-exponent). The mesh augments physics, never overrides it.
2. **Safety circuits are deterministic.** Detection and causation circuits use deterministic firing. Only non-critical circuits (memory, prediction) use stochastic firing.
3. **Max 2 circuits per neuron.** Hard architectural constraint. If three circuits need to communicate, chain the overlaps (A↔B↔C), never share a single neuron across 3+ circuits.
4. **Frozen universal core.** Physics-based circuits are frozen after pre-training. Online adaptation happens only through thin adapter layers and plastic circuits.
5. **Fail gracefully.** If the mesh fails, fall back to physics-only advisories. The system must never be worse than no system.

## Performance Targets

| Metric | Target | Hard Limit |
|---|---|---|
| Detection latency | < 500 us | < 1 ms |
| Full mesh step | < 1 ms | < 5 ms |
| Memory footprint | < 20 MB | < 50 MB |
| Neuron count | 8,192 | 16,384 max |
| Anomaly detection accuracy | > 95% | > 90% min |

## Architecture

The mesh uses a two-phase update cycle:

1. **Phase 1 — Parallel Computation:** All firing circuits compute proposed neuron updates independently (embarrassingly parallel, no shared mutable state).
2. **Phase 2 — Sequential Merge:** Overlap neurons receive competing updates which are resolved via safety override (detection circuits with high confidence win outright) or a learned gating network (small MLP producing softmax-weighted priorities).

### Default Layout (4,736 neurons, 13 circuits)

```
DETECTION BLOCK   (neurons 0..1023)     5 circuits, tau 10-20ms, Safety
CAUSATION BLOCK   (neurons 1024..2047)  3 circuits, tau 50ms,    Safety
MEMORY BLOCK      (neurons 2048..3583)  3 circuits, tau 0.2-30s, Performance
PREDICTION BLOCK  (neurons 3584..4607)  2 circuits, tau 20-30ms, Performance
ADAPTER NEURONS   (neurons 4608..4735)  128 neurons (64 formation + 64 well)
RESERVED          (neurons 4736..8191)  for future expansion
```

### Three-Tier Production Architecture

| Tier | What | Trainable? |
|---|---|---|
| Universal | Detection + causation + base prediction circuits | Frozen after pre-training |
| Adapters | 64-neuron formation adapter + 64-neuron well adapter | Plastic (online) |
| Adaptive | Well-specific circuits connected through adapters | Plastic (online) |

## Crate Structure

```
src/
├── lib.rs                    # Public API, re-exports
├── neuron.rs                 # LiquidNeuron, compute_neuron_update
├── circuit.rs                # Circuit, CircuitType, CircuitCriticality, should_fire
├── overlap.rs                # OverlapZone, OverlapGate (6->8->2 MLP)
├── topology.rs               # TopologyConstraint (max 2 circuits/neuron)
├── mesh.rs                   # OverlappingMesh, step(), merge_updates()
├── config.rs                 # MeshConfig with all hyperparameters
├── layout.rs                 # Default 4,736-neuron layout
├── physics.rs                # MSE, d-exponent, ECD, flow balance
├── memory/
│   ├── hierarchical.rs       # HierarchicalMemory, MemoryCircuit, WriteGate
│   └── episodic.rs           # EpisodicMemory, Episode, KNN recall
├── io/
│   ├── wits.rs               # WitsSnapshot (21 parameters)
│   ├── encoding.rs           # Baselines (Welford's), normalisation, input encoding
│   └── output.rs             # MeshOutput, Detection, CausalAnalysis, RiskLevel
└── adaptation/
    ├── adapter.rs            # AdapterLayer, AdapterNeuron
    ├── replay.rs             # ReplayBuffer, TrainingSample
    └── production.rs         # ProductionMesh (three-tier architecture)
```

## Quick Start

```rust
use sairen_mesh::{MeshConfig, WitsSnapshot};
use sairen_mesh::layout::build_default_mesh;
use sairen_mesh::io::encoding::{Baselines, encode_input};

// Build the default 4,736-neuron mesh.
let mut mesh = build_default_mesh();

// Create baselines and a WITS snapshot.
let mut baselines = Baselines::new();
let wits = WitsSnapshot::zeros();
baselines.update(&wits);

// Encode input and run mesh steps.
encode_input(&mut mesh, &wits, &baselines);
for _ in 0..100 {
    mesh.step();
}
```

Or use the full production pipeline:

```rust
use sairen_mesh::{MeshConfig, ProductionMesh, WitsSnapshot};

let mut mesh = ProductionMesh::new(MeshConfig::default());
let wits = WitsSnapshot::zeros();
let output = mesh.process(&wits);
```

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

67 tests covering:
- Neuron dynamics (convergence, tanh bounds, connection contributions)
- Circuit firing (deterministic Safety mode at exact tau intervals, stochastic Performance mode)
- Topology enforcement (rejects 3+ circuits per neuron, validates bounds)
- Overlap gates (softmax sums to 1.0, safety override, threshold adjustment)
- Memory (decay rates, write-gate blocking, short-term decays faster than long-term)
- Episodic memory (store/recall round-trips, KNN, pruning, sparse compression)
- Input encoding (Welford's statistics, normalisation, clamping)
- Adaptation (adapter forward pass, tanh bounds, replay buffer proportions)
- Layout (4,736 neurons, 13 circuits, all tau values match spec)
- Physics (MSE, ECD, flow balance)
- Integration (10,000 steps no panic, safety determinism, overlap merge)

## Dependencies

| Crate | Purpose |
|---|---|
| `rayon` | Parallel computation (Phase 1 of update cycle) |
| `rand` | Stochastic firing for Performance circuits |
| `serde` / `serde_json` | Serialisation |
| `log` | Logging |
| `thiserror` | Error types |

## Status

| Phase | Status |
|---|---|
| 1: Foundation (neurons, circuits, topology) | Done |
| 2: Overlap system (zones, gates, merge) | Done |
| 3: Memory (hierarchical + episodic) | Done |
| 4: I/O (WITS encoding, output decoding) | Done |
| 5: Adaptation (adapters, replay, ProductionMesh) | Done |
| 6: Training pipeline (backprop, 3-stage training) | Not started |
| 7: Benchmarks and polish | Partial |

## License

SAIREN Ltd. All rights reserved.
