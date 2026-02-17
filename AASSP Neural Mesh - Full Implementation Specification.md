
**System:** SAIREN-OS Asynchronous Overlapping Neural Mesh  
**Version:** 1.0  
**Date:** February 2026  
**Target:** Rust implementation, single-threaded with Rayon parallelism  
**Hardware:** RTX 4060 Ti (16 GB VRAM) edge box, CPU-first with future NPU path  
**Author:** SAIREN Ltd

-----

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Core Data Structures](#2-core-data-structures)
3. [Neuron Model](#3-neuron-model)
4. [Circuit System](#4-circuit-system)
5. [Overlap Zones & Topology Constraints](#5-overlap-zones--topology-constraints)
6. [Two-Phase Update Cycle](#6-two-phase-update-cycle)
7. [Adaptive Merge System](#7-adaptive-merge-system)
8. [Multi-Timescale Memory](#8-multi-timescale-memory)
9. [Episodic Memory Store](#9-episodic-memory-store)
10. [Modular Architecture & Adaptation](#10-modular-architecture--adaptation)
11. [Training Pipeline](#11-training-pipeline)
12. [Execution Modes](#12-execution-modes)
13. [WITS Integration Interface](#13-wits-integration-interface)
14. [Configuration & Hyperparameters](#14-configuration--hyperparameters)
15. [Testing Strategy](#15-testing-strategy)
16. [Build Order](#16-build-order)
17. [Crate Structure](#17-crate-structure)
18. [Appendix A: Physics Formulas](#appendix-a-physics-formulas)
19. [Appendix B: Circuit Catalogue](#appendix-b-circuit-catalogue)

-----

## 1. Overview & Goals

### What This Is

An overlapping liquid neural network mesh implementing **Asymmetric Asynchronous Shared-State Parallelism (AASSP)**. Multiple neural circuits share neurons through overlap zones, process at different timescales, and communicate through shared state rather than message passing.

### What It Does

Ingests real-time WITS Level 0 drilling data and produces:

- Sub-millisecond anomaly detection (kicks, losses, pack-off, stick-slip, founder)
- Causal reasoning about root causes
- Predictive state projection
- Memory-augmented pattern recognition from historical events

### Design Principles

1. **Physics is ground truth.** Every neural output is validated against deterministic physics calculations (MSE, ECD, d-exponent). The mesh augments physics, never overrides it.
2. **Safety circuits are deterministic.** Detection and causation circuits use deterministic firing. Only non-critical circuits (memory, prediction) use stochastic firing.
3. **Max 2 circuits per neuron.** Hard architectural constraint. If three circuits need to communicate, chain the overlaps (A↔B↔C), never share a single neuron across 3+ circuits.
4. **Frozen universal core.** Physics-based circuits are frozen after pre-training. Online adaptation happens only through thin adapter layers and plastic circuits.
5. **Fail gracefully.** If the mesh fails, fall back to template-based physics-only advisories. The system must never be worse than no system.

### Performance Targets

|Metric                    |Target  |Hard Limit   |
|--------------------------|--------|-------------|
|Detection latency         |< 500 μs|< 1 ms       |
|Causal reasoning latency  |< 2 ms  |< 5 ms       |
|Full mesh step            |< 1 ms  |< 5 ms       |
|Memory footprint (mesh)   |< 20 MB |< 50 MB      |
|Neuron count              |8,192   |16,384 max   |
|Circuit count             |64      |128 max      |
|Anomaly detection accuracy|> 95%   |> 90% minimum|
|Power draw (CPU inference)|< 15W   |< 30W        |

-----

## 2. Core Data Structures

### 2.1 Liquid Neuron

```rust
/// A single neuron in the liquid neural mesh.
/// Uses continuous-time dynamics: dx/dt = (-x + f(input)) / tau
pub struct LiquidNeuron {
    /// Current activation state (f32, range approximately -1.0 to 1.0)
    pub x: f32,
    
    /// Bias term
    pub bias: f32,
    
    /// Incoming connections: (source_neuron_index, weight)
    /// Sparse representation - only store non-zero connections
    pub connections: Vec<(usize, f32)>,
    
    /// Maximum number of circuits this neuron belongs to (enforced: max 2)
    pub circuit_membership_count: u8,
}
```

### 2.2 Circuit

```rust
/// A logical grouping of neurons that performs a specific function.
/// Circuits overlap by sharing neurons in their index sets.
pub struct Circuit {
    /// Unique identifier
    pub id: CircuitId,
    
    /// Human-readable name (e.g., "kick_detection", "torque_causation")
    pub name: String,
    
    /// Which neurons this circuit reads and writes
    pub neuron_indices: Vec<usize>,
    
    /// Time constant in seconds (determines update rate)
    /// Small tau = fast updates, large tau = slow integration
    pub tau: f32,
    
    /// Circuit function type
    pub circuit_type: CircuitType,
    
    /// Criticality level (determines firing mode)
    pub criticality: CircuitCriticality,
    
    /// Current confidence output (0.0 to 1.0)
    pub confidence: f32,
    
    /// Deterministic accumulator for firing (used when criticality = Safety)
    pub fire_accumulator: f32,
    
    /// Whether this circuit's weights are frozen (not trainable)
    pub frozen: bool,
}

pub type CircuitId = u32;

#[derive(Clone, Copy, PartialEq)]
pub enum CircuitType {
    Detection,   // Fast anomaly detection
    Causation,   // Root cause analysis
    Prediction,  // Forward state projection
    Memory,      // Working memory (short/medium/long)
}

#[derive(Clone, Copy, PartialEq)]
pub enum CircuitCriticality {
    /// Deterministic firing. Used for detection and causation circuits.
    /// Fires when accumulator exceeds tau. Fully reproducible.
    Safety,
    
    /// Stochastic firing. Used for memory and prediction circuits.
    /// Fires with probability dt/tau per timestep.
    Performance,
}
```

### 2.3 Overlap Zone

```rust
/// Defines the overlap between exactly two circuits.
/// Contains the merge strategy and learned gating parameters.
pub struct OverlapZone {
    /// The two circuits that share neurons in this zone
    pub circuit_a: CircuitId,
    pub circuit_b: CircuitId,
    
    /// Neuron indices that belong to BOTH circuits
    pub shared_neuron_indices: Vec<usize>,
    
    /// Learned soft masks per shared neuron (0.0 to 1.0)
    /// Used during training to discover optimal overlap width.
    /// After training, neurons with mask < PRUNE_THRESHOLD are removed.
    pub masks: Vec<f32>,
    
    /// Learned gating network for this overlap zone
    pub gate: OverlapGate,
    
    /// Base priority threshold for safety override
    pub safety_threshold: f32,
}

/// Small MLP that determines merge priority between two circuits
/// in an overlap zone based on current context.
pub struct OverlapGate {
    /// Input dimension: 6 (conf_a, tau_a, delta_a, conf_b, tau_b, delta_b)
    /// Hidden dimension: 8
    /// Output dimension: 2 (priority_a, priority_b after softmax)
    pub weights_ih: [[f32; 6]; 8],   // Input → Hidden (8 x 6)
    pub bias_h: [f32; 8],             // Hidden bias
    pub weights_ho: [[f32; 8]; 2],    // Hidden → Output (2 x 8)
    pub bias_o: [f32; 2],             // Output bias
}
```

### 2.4 The Mesh

```rust
/// The complete overlapping neural mesh.
/// This is the top-level structure that owns all neurons, circuits,
/// and overlap zones.
pub struct OverlappingMesh {
    /// All neurons in the mesh (shared substrate)
    pub neurons: Vec<LiquidNeuron>,
    
    /// All circuits operating on subsets of neurons
    pub circuits: Vec<Circuit>,
    
    /// Overlap zones between circuit pairs
    pub overlap_zones: Vec<OverlapZone>,
    
    /// Topology validator (enforces max-2-circuits-per-neuron)
    pub topology: TopologyConstraint,
    
    /// Multi-timescale memory system (inside the mesh)
    pub memory: HierarchicalMemory,
    
    /// Episodic memory store (outside the mesh, HNSW indexed)
    pub episodic_store: EpisodicMemory,
    
    /// Adapter layers for online adaptation
    pub formation_adapter: AdapterLayer,
    pub well_adapter: AdapterLayer,
    
    /// Replay buffer for catastrophic forgetting prevention
    pub replay_buffer: ReplayBuffer,
    
    /// Global mesh configuration
    pub config: MeshConfig,
    
    /// Current timestep (monotonically increasing)
    pub tick: u64,
}
```

-----

## 3. Neuron Model

### 3.1 Dynamics

Each neuron follows continuous-time liquid neural network dynamics:

```
dx/dt = (-x + tanh(Σ(w_i * x_i) + bias)) / tau
```

Where:

- `x` is the neuron’s current state
- `w_i` are incoming connection weights
- `x_i` are states of connected neurons
- `tau` is the circuit’s time constant (NOT per-neuron — the circuit provides tau)
- `tanh` is the activation function

### 3.2 Discrete Update

For a given timestep `dt`:

```rust
fn compute_neuron_update(
    neuron: &LiquidNeuron,
    all_neurons: &[LiquidNeuron],
    tau: f32,
    dt: f32,
) -> f32 {
    // Compute weighted input sum
    let input_sum: f32 = neuron.connections.iter()
        .map(|(src_idx, weight)| all_neurons[*src_idx].x * weight)
        .sum::<f32>() + neuron.bias;
    
    // Liquid neuron dynamics
    let dx = (-neuron.x + input_sum.tanh()) / tau;
    
    // Return the delta (not the new state — caller applies it)
    dx * dt
}
```

### 3.3 Connection Sparsity

Neurons use sparse connections. Target connection density:

- Detection circuits: 15-25% connectivity (fast, focused)
- Causation circuits: 25-40% connectivity (need broader context)
- Memory circuits: 10-20% connectivity (selective storage)
- Prediction circuits: 20-30% connectivity (moderate)

Connection weights are stored as `Vec<(usize, f32)>` where usize is the source neuron index. This avoids allocating full NxN weight matrices.

-----

## 4. Circuit System

### 4.1 Circuit Firing

Circuits do not update every timestep. Their firing rate is governed by their time constant `tau` and criticality level.

```rust
impl Circuit {
    /// Determine whether this circuit should fire (compute updates) this timestep.
    pub fn should_fire(&mut self, dt: f32) -> bool {
        match self.criticality {
            CircuitCriticality::Safety => {
                // Deterministic accumulator. Fully reproducible.
                self.fire_accumulator += dt;
                if self.fire_accumulator >= self.tau {
                    self.fire_accumulator -= self.tau;
                    true
                } else {
                    false
                }
            }
            CircuitCriticality::Performance => {
                // Stochastic firing. Probability proportional to dt/tau.
                // Use a seeded RNG for replay capability.
                rand::random::<f32>() < (dt / self.tau)
            }
        }
    }
}
```

### 4.2 Circuit Computation

When a circuit fires, it computes updates for all neurons in its index set:

```rust
impl Circuit {
    /// Compute proposed updates for all neurons in this circuit.
    /// Returns Vec of (neuron_index, proposed_delta).
    /// Does NOT modify neuron state — caller handles merge.
    pub fn compute_updates(
        &self,
        neurons: &[LiquidNeuron],
        dt: f32,
    ) -> Vec<(usize, f32)> {
        let mut updates = Vec::with_capacity(self.neuron_indices.len());
        
        for &neuron_idx in &self.neuron_indices {
            let delta = compute_neuron_update(
                &neurons[neuron_idx],
                neurons,
                self.tau,
                dt,
            );
            updates.push((neuron_idx, delta));
        }
        
        updates
    }
}
```

### 4.3 Circuit Tau Values

|Circuit Type        |Tau      |Effective Rate|Criticality|
|--------------------|---------|--------------|-----------|
|Kick Detection      |10 ms    |100 Hz        |Safety     |
|Loss Detection      |10 ms    |100 Hz        |Safety     |
|Pack-off Detection  |15 ms    |~67 Hz        |Safety     |
|Stick-slip Detection|10 ms    |100 Hz        |Safety     |
|Founder Detection   |20 ms    |50 Hz         |Safety     |
|Torque Causation    |50 ms    |20 Hz         |Safety     |
|Pressure Causation  |50 ms    |20 Hz         |Safety     |
|Flow Causation      |50 ms    |20 Hz         |Safety     |
|Short-term Memory   |200 ms   |5 Hz          |Performance|
|Medium-term Memory  |2,000 ms |0.5 Hz        |Performance|
|Long-term Memory    |30,000 ms|~0.03 Hz      |Performance|
|State Prediction    |20 ms    |50 Hz         |Performance|
|ROP Prediction      |30 ms    |~33 Hz        |Performance|

-----

## 5. Overlap Zones & Topology Constraints

### 5.1 Hard Constraint: Max 2 Circuits Per Neuron

This is enforced at mesh construction time and validated before every topology modification.

```rust
pub struct TopologyConstraint {
    /// Maximum circuits any single neuron may belong to
    pub max_circuits_per_neuron: u8,  // Always 2
}

impl TopologyConstraint {
    /// Validate the entire mesh topology.
    /// Returns Err with details if any neuron exceeds the limit.
    pub fn validate(&self, neurons: &[LiquidNeuron], circuits: &[Circuit]) -> Result<(), TopologyError> {
        let mut membership_count = vec![0u8; neurons.len()];
        
        for circuit in circuits {
            for &neuron_idx in &circuit.neuron_indices {
                membership_count[neuron_idx] += 1;
                if membership_count[neuron_idx] > self.max_circuits_per_neuron {
                    return Err(TopologyError::ExcessMembership {
                        neuron_idx,
                        count: membership_count[neuron_idx],
                        max: self.max_circuits_per_neuron,
                    });
                }
            }
        }
        
        Ok(())
    }
}
```

### 5.2 Chaining for Multi-Circuit Communication

If circuits A, B, and C all need to share information, DO NOT share neurons across all three. Instead, create chained overlaps:

```
Circuit A: neurons [0..128]
    Overlap A↔B: neurons [100..128]
Circuit B: neurons [100..256]
    Overlap B↔C: neurons [228..256]
Circuit C: neurons [228..384]
```

Information flows A → B → C through two sequential overlaps. This is slower than direct sharing but eliminates the 3+ contention problem entirely.

### 5.3 Overlap Zone Discovery (Differentiable Masking)

During training, overlap width is NOT fixed. Each potential overlap neuron has a learned soft mask:

```rust
impl OverlapZone {
    /// Apply masks to filter which neurons are actively shared.
    /// Masks are learned during training via gradient descent with L1 penalty.
    pub fn active_neuron_indices(&self, threshold: f32) -> Vec<usize> {
        self.shared_neuron_indices.iter()
            .zip(&self.masks)
            .filter(|(_, &mask)| mask > threshold)
            .map(|(&idx, _)| idx)
            .collect()
    }
    
    /// L1 regularization loss to encourage sparse overlaps
    pub fn sparsity_loss(&self) -> f32 {
        self.masks.iter().map(|m| m.abs()).sum::<f32>()
    }
}
```

**Initialisation:** All masks start at 0.5. L1 penalty coefficient λ = 0.01.  
**Pruning:** After training, remove neurons with mask < 0.3.  
**Expected result:** Network discovers 8-25% overlap per zone (not 30%).

### 5.4 Default Overlap Layout

Initial topology before training discovers optimal overlaps:

```
DETECTION BLOCK (neurons 0..1023, 5 circuits)
├── Kick Detection:       [0..204]
├── Loss Detection:       [205..409]
├── Pack-off Detection:   [410..614]
├── Stick-slip Detection: [615..819]
└── Founder Detection:    [820..1023]

    Overlap: Detection → Causation: neurons [920..1123]  (~10% initial)

CAUSATION BLOCK (neurons 1024..2047, 3 circuits)
├── Torque Causation:   [1024..1365]
├── Pressure Causation: [1366..1707]
└── Flow Causation:     [1708..2047]

    Overlap: Causation → Memory: neurons [1948..2147]  (~10% initial)

MEMORY BLOCK (neurons 2048..3583, 3 circuits)
├── Short-term Memory:  [2048..2559]
├── Medium-term Memory: [2560..3071]
└── Long-term Memory:   [3072..3583]

    Overlap: Short↔Medium: neurons [2460..2659]  (~10%)
    Overlap: Medium↔Long: neurons [2972..3171]   (~10%)

PREDICTION BLOCK (neurons 3584..4607, 2 circuits)
├── State Prediction: [3584..4095]
└── ROP Prediction:   [4096..4607]

    Overlap: Detection → Prediction: neurons [920..1023, 3484..3683]  (~5% weak coupling)

ADAPTER NEURONS (neurons 4608..4735, 128 neurons)
├── Formation Adapter: [4608..4671]  (64 neurons)
└── Well Adapter:      [4672..4735]  (64 neurons)

TOTAL: 4,736 neurons in initial layout (well under 8,192 budget)
RESERVED: neurons 4736..8191 for future expansion
```

-----

## 6. Two-Phase Update Cycle

This is the core execution loop. Every mesh step follows this exact sequence.

### 6.1 Phase 1: Parallel Computation

All circuits that fire this timestep compute their proposed updates independently. This phase is embarrassingly parallel — no shared mutable state.

```rust
impl OverlappingMesh {
    pub fn step(&mut self, dt: f32) {
        // ═══════════════════════════════════════════════════
        // PHASE 1: PARALLEL COMPUTATION (no race conditions)
        // ═══════════════════════════════════════════════════
        
        let updates: Vec<(CircuitId, Vec<(usize, f32)>)> = self.circuits
            .par_iter_mut()  // Rayon parallel iteration
            .filter_map(|circuit| {
                if circuit.should_fire(dt) {
                    let circuit_updates = circuit.compute_updates(&self.neurons, dt);
                    // Also update circuit confidence based on output neurons
                    Some((circuit.id, circuit_updates))
                } else {
                    None
                }
            })
            .collect();
        
        // ═══════════════════════════════════════════════════
        // PHASE 2: SEQUENTIAL MERGE (handles overlaps)
        // ═══════════════════════════════════════════════════
        
        self.merge_updates(&updates, dt);
        
        // ═══════════════════════════════════════════════════
        // PHASE 3: POST-STEP BOOKKEEPING
        // ═══════════════════════════════════════════════════
        
        self.tick += 1;
    }
}
```

### 6.2 Phase 2: Sequential Merge

For each neuron, collect all proposed updates and merge them. Non-overlapping neurons get direct application. Overlapping neurons go through the adaptive merge system.

```rust
impl OverlappingMesh {
    fn merge_updates(
        &mut self,
        updates: &[(CircuitId, Vec<(usize, f32)>)],
        dt: f32,
    ) {
        // Build a map: neuron_index → Vec<(circuit_id, delta)>
        let mut neuron_updates: Vec<Vec<(CircuitId, f32)>> = 
            vec![Vec::new(); self.neurons.len()];
        
        for (circuit_id, circuit_updates) in updates {
            for &(neuron_idx, delta) in circuit_updates {
                neuron_updates[neuron_idx].push((*circuit_id, delta));
            }
        }
        
        // Apply updates
        for (neuron_idx, pending) in neuron_updates.iter().enumerate() {
            match pending.len() {
                0 => {} // No update this timestep
                1 => {
                    // Single circuit owns this neuron — direct application
                    self.neurons[neuron_idx].x += pending[0].1;
                }
                2 => {
                    // Overlap zone — use adaptive merge
                    let delta = self.merge_overlap(neuron_idx, pending);
                    self.neurons[neuron_idx].x += delta;
                }
                _ => {
                    // Should never happen (topology constraint enforces max 2)
                    // If it does, log error and use highest-confidence circuit
                    log::error!(
                        "Neuron {} has {} pending updates (max 2 expected)",
                        neuron_idx, pending.len()
                    );
                    let best = pending.iter()
                        .max_by(|a, b| {
                            self.get_circuit_confidence(a.0)
                                .partial_cmp(&self.get_circuit_confidence(b.0))
                                .unwrap()
                        })
                        .unwrap();
                    self.neurons[neuron_idx].x += best.1;
                }
            }
        }
    }
}
```

-----

## 7. Adaptive Merge System

### 7.1 Strategy: Priority Hierarchy + Learned Gate (Hybrid)

When two circuits propose updates for the same neuron:

1. **Check safety override:** If either circuit is a Detection type with confidence above the (context-adjusted) threshold, that circuit wins outright.
2. **Otherwise, use learned gate:** The overlap zone’s gating network determines the priority weighting.

```rust
impl OverlappingMesh {
    fn merge_overlap(
        &self,
        neuron_idx: usize,
        pending: &[(CircuitId, f32)],
    ) -> f32 {
        debug_assert!(pending.len() == 2);
        
        let (circuit_a_id, delta_a) = pending[0];
        let (circuit_b_id, delta_b) = pending[1];
        
        let circuit_a = self.get_circuit(circuit_a_id);
        let circuit_b = self.get_circuit(circuit_b_id);
        
        // Find the overlap zone for these two circuits
        let zone = self.find_overlap_zone(circuit_a_id, circuit_b_id);
        
        // ─── Safety Override ───
        // If a detection circuit has high confidence, it wins immediately.
        // The threshold is adjusted by the learned gate based on context.
        let context_threshold = zone.gate.adjust_threshold(
            zone.safety_threshold,
            circuit_a,
            circuit_b,
        );
        
        if circuit_a.circuit_type == CircuitType::Detection 
            && circuit_a.confidence > context_threshold {
            return delta_a;
        }
        if circuit_b.circuit_type == CircuitType::Detection 
            && circuit_b.confidence > context_threshold {
            return delta_b;
        }
        
        // ─── Learned Gate Merge ───
        let (priority_a, priority_b) = zone.gate.compute_priorities(
            circuit_a, delta_a,
            circuit_b, delta_b,
        );
        
        delta_a * priority_a + delta_b * priority_b
    }
}
```

### 7.2 Overlap Gate Forward Pass

```rust
impl OverlapGate {
    /// Compute merge priorities for two circuits.
    /// Returns (priority_a, priority_b) where both sum to 1.0.
    pub fn compute_priorities(
        &self,
        circuit_a: &Circuit, delta_a: f32,
        circuit_b: &Circuit, delta_b: f32,
    ) -> (f32, f32) {
        // Input features: [conf_a, tau_a, delta_a, conf_b, tau_b, delta_b]
        let input = [
            circuit_a.confidence,
            circuit_a.tau,
            delta_a,
            circuit_b.confidence,
            circuit_b.tau,
            delta_b,
        ];
        
        // Hidden layer: ReLU activation
        let mut hidden = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = self.bias_h[i];
            for j in 0..6 {
                sum += self.weights_ih[i][j] * input[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }
        
        // Output layer: raw logits
        let mut logits = [0.0f32; 2];
        for i in 0..2 {
            let mut sum = self.bias_o[i];
            for j in 0..8 {
                sum += self.weights_ho[i][j] * hidden[j];
            }
            logits[i] = sum;
        }
        
        // Softmax
        let max_logit = logits[0].max(logits[1]);
        let exp_a = (logits[0] - max_logit).exp();
        let exp_b = (logits[1] - max_logit).exp();
        let sum = exp_a + exp_b;
        
        (exp_a / sum, exp_b / sum)
    }
    
    /// Adjust the safety override threshold based on context.
    /// Returns a threshold in [0.5, 0.95].
    pub fn adjust_threshold(
        &self,
        base_threshold: f32,
        circuit_a: &Circuit,
        circuit_b: &Circuit,
    ) -> f32 {
        // Simple heuristic: if both circuits are highly confident
        // but disagree, lower the threshold (be more cautious).
        let disagreement = (circuit_a.confidence - circuit_b.confidence).abs();
        let adjusted = base_threshold - (disagreement * 0.1);
        adjusted.clamp(0.5, 0.95)
    }
}
```

-----

## 8. Multi-Timescale Memory

### 8.1 Hierarchical Memory (Inside the Mesh)

Three memory circuits with different time constants provide working memory at different temporal scales.

```rust
pub struct HierarchicalMemory {
    /// Recent events, decays in ~1 second
    pub short_term: MemoryCircuit,   // tau = 200ms
    
    /// Current operation context, decays in ~10 seconds
    pub medium_term: MemoryCircuit,  // tau = 2,000ms
    
    /// Well behaviour patterns, decays in ~5 minutes
    pub long_term: MemoryCircuit,    // tau = 30,000ms
}

pub struct MemoryCircuit {
    /// Base circuit (contains neuron indices, tau, etc.)
    pub circuit: Circuit,
    
    /// Exponential decay rate (lambda)
    /// Applied every timestep: x *= exp(-lambda * dt)
    pub decay_rate: f32,
    
    /// Learned write gate weights
    /// Input: causation circuit output vector
    /// Output: scalar gate value (0.0 = ignore, 1.0 = write)
    pub write_gate: WriteGate,
}

pub struct WriteGate {
    /// Simple linear layer: input_dim → 1, followed by sigmoid
    pub weights: Vec<f32>,  // Length = number of causation output neurons
    pub bias: f32,
}
```

### 8.2 Memory Circuit Update

```rust
impl MemoryCircuit {
    pub fn update(&mut self, neurons: &mut [LiquidNeuron], causation_output: &[f32], dt: f32) {
        // Step 1: Exponential decay of all memory neurons
        for &neuron_idx in &self.circuit.neuron_indices {
            neurons[neuron_idx].x *= (-self.decay_rate * dt).exp();
        }
        
        // Step 2: Compute write gate
        let gate_value = self.write_gate.forward(causation_output);
        
        // Step 3: Gated write (standard liquid neuron update scaled by gate)
        if gate_value > 0.05 {  // Skip if gate is effectively closed
            let updates = self.circuit.compute_updates(neurons, dt);
            for (neuron_idx, delta) in updates {
                neurons[neuron_idx].x += gate_value * delta;
            }
        }
    }
}

impl WriteGate {
    pub fn forward(&self, causation_output: &[f32]) -> f32 {
        let sum: f32 = self.weights.iter()
            .zip(causation_output)
            .map(|(w, x)| w * x)
            .sum::<f32>() + self.bias;
        
        // Sigmoid activation
        1.0 / (1.0 + (-sum).exp())
    }
}
```

### 8.3 Memory Decay Rates

|Memory Level|Tau      |Decay Rate (λ)|Half-life  |Purpose                                |
|------------|---------|--------------|-----------|---------------------------------------|
|Short-term  |200 ms   |0.7           |~1 second  |“What just happened?”                  |
|Medium-term |2,000 ms |0.07          |~10 seconds|“What’s been happening this operation?”|
|Long-term   |30,000 ms|0.004         |~3 minutes |“How has this well been behaving?”     |

-----

## 9. Episodic Memory Store

### 9.1 Purpose

The episodic store lives OUTSIDE the neural mesh. It provides long-term memory across hours, days, and wells using an HNSW index for fast approximate nearest-neighbor retrieval.

### 9.2 Data Structure

```rust
pub struct EpisodicMemory {
    /// All stored episodes
    pub episodes: Vec<Episode>,
    
    /// HNSW index for fast similarity search
    /// Index vectors are mesh state snapshots (dimensionality = num_neurons)
    pub index: HnswIndex,
    
    /// Maximum number of stored episodes (prevent unbounded growth)
    pub max_episodes: usize,  // Default: 10,000
    
    /// Minimum confidence to trigger storage
    pub store_threshold: f32,  // Default: 0.85
}

pub struct Episode {
    /// When this episode was recorded
    pub timestamp: std::time::Instant,
    
    /// Monotonic tick when recorded
    pub tick: u64,
    
    /// Compressed mesh state snapshot
    /// Only stores neurons with |x| > 0.01 (sparse representation)
    pub state_vector: Vec<(usize, f32)>,
    
    /// What the causation circuits concluded
    pub diagnosis: DiagnosisType,
    
    /// Confidence at time of storage
    pub confidence: f32,
    
    /// Key drilling parameters at time of event
    pub drilling_context: DrillingContext,
    
    /// What happened after (outcome, filled in retrospectively)
    pub outcome: Option<Outcome>,
}

#[derive(Clone)]
pub struct DrillingContext {
    pub depth_m: f32,
    pub wob_klbs: f32,
    pub rop_ft_hr: f32,
    pub rpm: f32,
    pub torque_ft_lbs: f32,
    pub spp_psi: f32,
    pub flow_in_gpm: f32,
    pub flow_out_gpm: f32,
    pub mse_psi: f32,
    pub formation: Option<String>,
}

pub enum DiagnosisType {
    KickDetected,
    LossDetected,
    PackOff,
    StickSlip,
    FounderCondition,
    MSEInefficiency,
    FormationChange,
    EquipmentAnomaly(String),
    Unknown,
}

pub enum Outcome {
    Resolved { action_taken: String, time_to_resolve_s: f32 },
    FalseAlarm,
    Escalated,
    StillOngoing,
}
```

### 9.3 Store and Recall

```rust
impl EpisodicMemory {
    /// Store a new episode if confidence exceeds threshold.
    pub fn maybe_store(
        &mut self,
        mesh: &OverlappingMesh,
        diagnosis: DiagnosisType,
        confidence: f32,
        drilling_context: DrillingContext,
    ) {
        if confidence < self.store_threshold {
            return;
        }
        
        // Compress mesh state (only non-trivial neurons)
        let state_vector: Vec<(usize, f32)> = mesh.neurons.iter()
            .enumerate()
            .filter(|(_, n)| n.x.abs() > 0.01)
            .map(|(i, n)| (i, n.x))
            .collect();
        
        let episode = Episode {
            timestamp: std::time::Instant::now(),
            tick: mesh.tick,
            state_vector,
            diagnosis,
            confidence,
            drilling_context,
            outcome: None,
        };
        
        // Add to HNSW index
        let dense_vector = self.to_dense_vector(&episode.state_vector, mesh.neurons.len());
        self.index.insert(&dense_vector, self.episodes.len());
        self.episodes.push(episode);
        
        // Prune if over capacity (remove oldest low-confidence episodes)
        if self.episodes.len() > self.max_episodes {
            self.prune_oldest_low_confidence();
        }
    }
    
    /// Find k most similar past episodes to current mesh state.
    pub fn recall_similar(
        &self,
        mesh: &OverlappingMesh,
        k: usize,
    ) -> Vec<&Episode> {
        let current_state: Vec<f32> = mesh.neurons.iter().map(|n| n.x).collect();
        
        let indices = self.index.knn_search(&current_state, k);
        
        indices.iter()
            .filter_map(|&idx| self.episodes.get(idx))
            .collect()
    }
}
```

-----

## 10. Modular Architecture & Adaptation

### 10.1 Three-Tier Architecture

```rust
pub struct ProductionMesh {
    // ═══════════════════════════════════════
    // TIER 1: UNIVERSAL (Frozen after pre-training)
    // ═══════════════════════════════════════
    // Physics-based detection and causation circuits.
    // Trained on labelled drilling data. Never modified online.
    pub universal_mesh: OverlappingMesh,
    
    // ═══════════════════════════════════════
    // TIER 2: ADAPTERS (Plastic, thin layers)
    // ═══════════════════════════════════════
    // Small adapter layers (64 neurons each) that transform
    // universal circuit outputs into formation/well-specific representations.
    pub formation_adapter: AdapterLayer,  // 64 neurons
    pub well_adapter: AdapterLayer,       // 64 neurons
    
    // ═══════════════════════════════════════
    // TIER 3: ADAPTIVE (Plastic, online learning)
    // ═══════════════════════════════════════
    // Well-specific circuits that learn online during drilling.
    // Connected to universal circuits ONLY through adapters.
    pub adaptive_circuits: Vec<Circuit>,
    
    // ═══════════════════════════════════════
    // SUPPORT STRUCTURES
    // ═══════════════════════════════════════
    pub replay_buffer: ReplayBuffer,
    pub episodic_store: EpisodicMemory,
}
```

### 10.2 Adapter Layer

```rust
/// Thin transformation layer between frozen universal circuits
/// and plastic adaptive circuits. Only 64 neurons.
pub struct AdapterLayer {
    /// Adapter neurons (not part of the main mesh neuron array)
    pub neurons: Vec<AdapterNeuron>,
    
    /// Which universal circuit outputs feed into this adapter
    pub input_from: Vec<usize>,  // Neuron indices in universal mesh
    
    /// Which adaptive circuit inputs this adapter feeds
    pub output_to: Vec<usize>,   // Neuron indices in adaptive circuits
    
    /// Whether this adapter is currently learning
    pub trainable: bool,
}

pub struct AdapterNeuron {
    pub x: f32,
    pub weights_in: Vec<f32>,   // From universal circuit output neurons
    pub weights_out: Vec<f32>,  // To adaptive circuit input neurons  
    pub bias: f32,
}
```

### 10.3 Replay Buffer

```rust
/// Stores canonical drilling scenarios to prevent catastrophic forgetting
/// during online adapter training.
pub struct ReplayBuffer {
    /// Representative samples from pre-training data
    pub samples: Vec<TrainingSample>,
    
    /// Maximum buffer size
    pub max_size: usize,  // Default: 500
    
    /// Fraction of training batch that comes from replay (vs live data)
    pub replay_fraction: f32,  // Default: 0.1 (10%)
}

pub struct TrainingSample {
    /// Input WITS data snapshot
    pub input: WitsSnapshot,
    
    /// Expected circuit outputs (labels)
    pub expected_outputs: HashMap<CircuitId, Vec<f32>>,
    
    /// Formation type (for formation-specific replay)
    pub formation: Option<String>,
    
    /// Sample importance weight
    pub weight: f32,
}

impl ReplayBuffer {
    /// Sample a mixed batch: (1 - replay_fraction) live + replay_fraction replay
    pub fn mixed_batch(
        &self,
        live_samples: &[TrainingSample],
        batch_size: usize,
    ) -> Vec<&TrainingSample> {
        let n_replay = (batch_size as f32 * self.replay_fraction) as usize;
        let n_live = batch_size - n_replay;
        
        let mut batch: Vec<&TrainingSample> = Vec::with_capacity(batch_size);
        
        // Live samples
        for sample in live_samples.iter().take(n_live) {
            batch.push(sample);
        }
        
        // Replay samples (random selection)
        let mut rng = rand::thread_rng();
        for _ in 0..n_replay {
            let idx = rng.gen_range(0..self.samples.len());
            batch.push(&self.samples[idx]);
        }
        
        batch
    }
}
```

-----

## 11. Training Pipeline

### 11.1 Three-Stage Training

Training follows a strict three-stage pipeline. Do not skip or reorder stages.

**Stage 1: Overlap Zone Pre-Training**

Train overlap zones as communication protocols before training the circuits that use them.

```
Objective: Maximise mutual information between circuit-pair outputs
Method: Unsupervised, trains only overlap zone masks and gate weights
Duration: 5,000 steps
Learning rate: 1e-3
Loss: mutual_information_loss + λ_sparsity * sparsity_loss
```

**Stage 2: Circuit Training with Alternating Freezes**

Train circuits in alternating rounds with frozen overlap zones.

```
For round in 0..5:
    1. Freeze detection circuits
    2. Train causation circuits (500 steps, lr=1e-3)
    3. Freeze causation circuits  
    4. Train detection circuits (500 steps, lr=1e-3)
    5. Train memory circuits (200 steps, lr=5e-4)
    6. Train prediction circuits (200 steps, lr=5e-4)

Overlap zones remain frozen throughout Stage 2.
```

**Stage 3: Joint Fine-Tuning**

Unfreeze everything and fine-tune end-to-end with overlap stability regularisation.

```
Duration: 2,000 steps
Learning rate: 1e-4 (small, to avoid catastrophic changes)
Loss: task_loss + λ_overlap * overlap_stability_loss + λ_sparsity * sparsity_loss

overlap_stability_loss = Σ (mask_current - mask_stage1)² for all overlap masks
λ_overlap = 0.1
λ_sparsity = 0.01
```

### 11.2 Post-Training Freeze

After Stage 3:

1. Prune overlap neurons with mask < 0.3
2. Freeze all universal circuits (detection, causation, base prediction)
3. Freeze all overlap zones
4. Only adapter layers and adaptive circuits remain trainable
5. Validate topology constraint (max 2 circuits per neuron)
6. Run validation suite (see Section 15)

### 11.3 Online Learning (During Drilling)

Only adapter layers and adaptive circuits train online.

```rust
impl ProductionMesh {
    pub fn online_train_step(&mut self, live_sample: &TrainingSample) {
        // Only train adapters and adaptive circuits
        // Universal circuits and overlaps are frozen
        
        // Get mixed batch (90% live, 10% replay)
        let batch = self.replay_buffer.mixed_batch(
            &[live_sample.clone()],
            10,  // batch size
        );
        
        for sample in batch {
            // Forward pass through frozen universal mesh
            let universal_output = self.universal_mesh.forward(&sample.input);
            
            // Train formation adapter
            if self.formation_adapter.trainable {
                self.formation_adapter.train_step(
                    &universal_output,
                    &sample.expected_outputs,
                    /* lr */ 1e-4,
                );
            }
            
            // Train well adapter
            if self.well_adapter.trainable {
                self.well_adapter.train_step(
                    &universal_output,
                    &sample.expected_outputs,
                    /* lr */ 1e-4,
                );
            }
        }
    }
}
```

-----

## 12. Execution Modes

### 12.1 Shadow Mode (Days 1-7 of deployment)

```rust
pub enum ExecutionMode {
    /// No outputs. Silently learns baselines. Zero operational impact.
    Shadow,
    
    /// Generates advisories but flags them as "unvalidated".
    Advisory,
    
    /// Full production. Advisories are confident and logged.
    Production,
}
```

In Shadow mode:

- All circuits run normally
- No advisories are generated or displayed
- The system is learning this rig’s baseline parameters
- Memory circuits accumulate normal operating patterns
- Adaptive circuits begin online learning

### 12.2 Degradation Modes

```rust
pub enum DegradationMode {
    /// Full mesh operational
    Full,
    
    /// Mesh failed, using physics-only calculations
    PhysicsOnly,
    
    /// Mesh partially failed, running detection circuits only
    DetectionOnly,
    
    /// Memory/episodic store unavailable, mesh runs without history
    NoMemory,
}

impl ProductionMesh {
    pub fn health_check(&self) -> DegradationMode {
        // Check if mesh can step without errors
        if !self.universal_mesh.is_healthy() {
            return DegradationMode::PhysicsOnly;
        }
        
        // Check if memory is operational
        if !self.episodic_store.is_healthy() {
            return DegradationMode::NoMemory;
        }
        
        // Check if all circuit types are firing
        if !self.all_circuit_types_active() {
            return DegradationMode::DetectionOnly;
        }
        
        DegradationMode::Full
    }
}
```

-----

## 13. WITS Integration Interface

### 13.1 Input Format

The mesh receives pre-parsed WITS data as a struct:

```rust
/// A single WITS data snapshot, parsed from the raw TCP stream.
/// This is the input to the mesh at each timestep.
pub struct WitsSnapshot {
    pub timestamp: std::time::Instant,
    
    // Primary drilling parameters
    pub wob_klbs: f32,           // Weight on Bit (thousands of pounds)
    pub rop_ft_hr: f32,          // Rate of Penetration (feet/hour)
    pub rpm: f32,                // Rotary Speed
    pub torque_ft_lbs: f32,      // Torque at motor (ft-lbs)
    pub hook_load_klbs: f32,     // Hook load (thousands of pounds)
    pub spp_psi: f32,            // Standpipe Pressure (psi)
    pub flow_in_gpm: f32,        // Mud pump flow in (gallons/min)
    pub flow_out_gpm: f32,       // Return flow out (gallons/min)
    pub pit_volume_bbl: f32,     // Active pit volume (barrels)
    pub bit_depth_ft: f32,       // Current bit depth (feet)
    
    // Mud properties
    pub mud_weight_in_ppg: f32,  // Mud weight in (ppg)
    pub mud_weight_out_ppg: f32, // Mud weight out (ppg)
    pub mud_temp_in_f: f32,      // Mud temperature in (°F)
    pub mud_temp_out_f: f32,     // Mud temperature out (°F)
    
    // Gas monitoring
    pub total_gas_pct: f32,      // Total gas (%)
    pub h2s_ppm: f32,            // H2S (ppm)
    
    // Derived physics (computed by the physics engine BEFORE mesh input)
    pub mse_psi: f32,            // Mechanical Specific Energy
    pub d_exponent: f32,         // D-exponent
    pub ecd_ppg: f32,            // Equivalent Circulating Density
    pub flow_balance_gpm: f32,   // flow_out - flow_in
    pub pit_rate_bbl_hr: f32,    // Rate of change of pit volume
}
```

### 13.2 Input Encoding

The WITS snapshot must be encoded into neuron activations for the mesh input layer. Use normalisation based on learned baselines:

```rust
impl OverlappingMesh {
    /// Encode a WITS snapshot into the mesh's input neurons.
    /// Input neurons are the first N neurons of each detection circuit.
    pub fn encode_input(&mut self, wits: &WitsSnapshot) {
        // Normalise each parameter to approximately [-1, 1] range
        // using running mean and standard deviation from baseline period
        let encoded = [
            self.normalise(wits.wob_klbs, &self.baselines.wob),
            self.normalise(wits.rop_ft_hr, &self.baselines.rop),
            self.normalise(wits.rpm, &self.baselines.rpm),
            self.normalise(wits.torque_ft_lbs, &self.baselines.torque),
            self.normalise(wits.spp_psi, &self.baselines.spp),
            self.normalise(wits.flow_in_gpm, &self.baselines.flow_in),
            self.normalise(wits.flow_out_gpm, &self.baselines.flow_out),
            self.normalise(wits.pit_volume_bbl, &self.baselines.pit_volume),
            self.normalise(wits.mse_psi, &self.baselines.mse),
            self.normalise(wits.d_exponent, &self.baselines.d_exponent),
            self.normalise(wits.ecd_ppg, &self.baselines.ecd),
            self.normalise(wits.flow_balance_gpm, &self.baselines.flow_balance),
            self.normalise(wits.total_gas_pct, &self.baselines.total_gas),
            self.normalise(wits.h2s_ppm, &self.baselines.h2s),
        ];
        
        // Write to input neurons (first 14 neurons of each detection circuit)
        for circuit in &self.circuits {
            if circuit.circuit_type == CircuitType::Detection {
                for (i, &value) in encoded.iter().enumerate() {
                    if i < circuit.neuron_indices.len() {
                        self.neurons[circuit.neuron_indices[i]].x = value;
                    }
                }
            }
        }
    }
    
    fn normalise(&self, value: f32, baseline: &BaselineStats) -> f32 {
        if baseline.std_dev < 1e-6 {
            return 0.0;  // No variance, treat as zero
        }
        ((value - baseline.mean) / baseline.std_dev).clamp(-3.0, 3.0) / 3.0
    }
}

pub struct BaselineStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sample_count: u64,
}
```

### 13.3 Output Decoding

```rust
/// The mesh's output after processing one WITS snapshot.
pub struct MeshOutput {
    /// Per-detection-circuit results
    pub detections: Vec<Detection>,
    
    /// Per-causation-circuit results
    pub causal_analyses: Vec<CausalAnalysis>,
    
    /// Predictions for next N timesteps
    pub predictions: Vec<Prediction>,
    
    /// Similar past episodes (if any)
    pub episodic_matches: Vec<EpisodicMatch>,
    
    /// Overall risk level (aggregated from all circuits)
    pub risk_level: RiskLevel,
    
    /// Current degradation mode
    pub mode: DegradationMode,
}

pub struct Detection {
    pub circuit_name: String,
    pub anomaly_type: DiagnosisType,
    pub confidence: f32,
    pub severity: Severity,
}

pub struct CausalAnalysis {
    pub circuit_name: String,
    pub root_cause: String,
    pub contributing_factors: Vec<String>,
    pub confidence: f32,
    pub recommended_action: String,
}

pub enum RiskLevel {
    Green,   // >85% efficiency, no anomalies
    Yellow,  // Minor anomaly, log it
    Amber,   // Sustained issue, needs attention
    Red,     // Hard limit breach, immediate alert
}

pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}
```

-----

## 14. Configuration & Hyperparameters

### 14.1 Mesh Configuration

```rust
pub struct MeshConfig {
    // ─── Neuron Parameters ───
    pub total_neurons: usize,              // Default: 4,736 (expandable to 8,192)
    pub reserved_neurons: usize,           // Default: 3,456 (8192 - 4736)
    
    // ─── Timestep ───
    pub dt: f32,                           // Default: 0.001 (1 ms)
    pub wits_sample_rate_hz: f32,          // Default: 1.0 (1 Hz from WITS)
    pub mesh_steps_per_wits_sample: u32,   // Default: 100 (100 mesh steps per WITS sample)
    
    // ─── Overlap ───
    pub initial_overlap_fraction: f32,     // Default: 0.10 (10%)
    pub mask_prune_threshold: f32,         // Default: 0.3
    pub sparsity_lambda: f32,              // Default: 0.01
    pub overlap_stability_lambda: f32,     // Default: 0.1
    
    // ─── Safety ───
    pub safety_override_threshold: f32,    // Default: 0.75
    pub max_circuits_per_neuron: u8,       // Default: 2 (HARD LIMIT)
    
    // ─── Memory ───
    pub short_term_tau: f32,               // Default: 0.2 (200 ms)
    pub short_term_decay: f32,             // Default: 0.7
    pub medium_term_tau: f32,              // Default: 2.0 (2 s)
    pub medium_term_decay: f32,            // Default: 0.07
    pub long_term_tau: f32,                // Default: 30.0 (30 s)
    pub long_term_decay: f32,              // Default: 0.004
    
    // ─── Episodic Store ───
    pub episodic_store_threshold: f32,     // Default: 0.85
    pub episodic_max_episodes: usize,      // Default: 10,000
    pub episodic_knn_k: usize,            // Default: 5
    
    // ─── Replay Buffer ───
    pub replay_buffer_size: usize,         // Default: 500
    pub replay_fraction: f32,              // Default: 0.1
    
    // ─── Adapter ───
    pub formation_adapter_size: usize,     // Default: 64
    pub well_adapter_size: usize,          // Default: 64
    
    // ─── Training ───
    pub stage1_steps: u32,                 // Default: 5,000
    pub stage1_lr: f32,                    // Default: 1e-3
    pub stage2_rounds: u32,               // Default: 5
    pub stage2_steps_per_round: u32,      // Default: 500
    pub stage2_lr: f32,                    // Default: 1e-3
    pub stage3_steps: u32,                 // Default: 2,000
    pub stage3_lr: f32,                    // Default: 1e-4
    pub online_lr: f32,                    // Default: 1e-4
    
    // ─── Baseline Learning ───
    pub baseline_min_samples: u64,         // Default: 100
    pub baseline_period_s: f32,            // Default: 604,800 (7 days)
    
    // ─── Execution ───
    pub initial_mode: ExecutionMode,       // Default: Shadow
    pub enable_episodic_store: bool,       // Default: true
    pub enable_online_learning: bool,      // Default: true
}
```

-----

## 15. Testing Strategy

### 15.1 Unit Tests

Each component must have unit tests:

|Component              |Tests Required                                                                                    |
|-----------------------|--------------------------------------------------------------------------------------------------|
|`LiquidNeuron`         |Update dynamics converge; tanh bounds output; zero input → decay to zero                          |
|`Circuit::should_fire` |Safety mode: deterministic at exact tau intervals; Performance mode: fires ~N/tau times in N steps|
|`TopologyConstraint`   |Rejects 3+ circuits per neuron; accepts valid topologies                                          |
|`OverlapGate`          |Softmax outputs sum to 1.0; safety override triggers above threshold                              |
|`OverlapZone`          |Masking prunes correctly; sparsity loss is non-negative                                           |
|`MemoryCircuit`        |Decay reduces activation over time; write gate blocks low-confidence events                       |
|`EpisodicMemory`       |Store/recall round-trips; KNN returns correct neighbours; pruning removes oldest                  |
|`WitsSnapshot encoding`|All parameters normalised to [-1, 1]; zero-variance handled                                       |
|`ReplayBuffer`         |Mixed batch has correct proportions; buffer respects max size                                     |
|`AdapterLayer`         |Forward pass produces output; frozen adapters don’t update                                        |

### 15.2 Integration Tests

|Test                                |Description                                                                                      |
|------------------------------------|-------------------------------------------------------------------------------------------------|
|`test_full_step_no_panic`           |Run 10,000 mesh steps with random WITS input, no panics                                          |
|`test_safety_determinism`           |Run same input twice, safety circuit outputs are bit-identical                                   |
|`test_detection_kick`               |Inject kick signature (flow_out >> flow_in), verify detection circuit fires with confidence > 0.8|
|`test_detection_packoff`            |Inject pack-off signature (torque + SPP rising), verify detection                                |
|`test_overlap_information_flow`     |Verify that detection circuit output affects causation circuit state through overlap             |
|`test_memory_decay`                 |Store event, advance 10 seconds, verify short-term memory decayed but long-term retained         |
|`test_episodic_recall`              |Store 100 episodes, query with similar state, verify correct matches returned                    |
|`test_topology_enforcement`         |Attempt to create mesh with 3-circuit neuron, verify rejection                                   |
|`test_degradation_fallback`         |Corrupt mesh state, verify system falls back to PhysicsOnly mode                                 |
|`test_online_learning_no_forgetting`|Train adapter for 1000 steps, verify universal circuit outputs unchanged                         |

### 15.3 Benchmark Tests

|Benchmark               |Target                                                  |
|------------------------|--------------------------------------------------------|
|`bench_single_step`     |< 500 μs for full mesh step (4,736 neurons, 13 circuits)|
|`bench_detection_only`  |< 100 μs for detection circuits only                    |
|`bench_episodic_query`  |< 200 μs for KNN-5 on 10,000 episodes                   |
|`bench_input_encoding`  |< 10 μs for WITS → neuron encoding                      |
|`bench_memory_footprint`|< 20 MB total mesh state                                |

-----

## 16. Build Order

Implement in this exact order. Each phase builds on the previous. Do not skip ahead.

### Phase 1: Foundation (Core neuron and circuit types)

1. `LiquidNeuron` struct and `compute_neuron_update` function
2. `Circuit` struct with `should_fire` (both modes) and `compute_updates`
3. `TopologyConstraint` with validation
4. Basic `OverlappingMesh` struct (neurons + circuits, no overlaps yet)
5. Two-phase update cycle (`step` method) with direct application (no merge yet)
6. Unit tests for all above

### Phase 2: Overlap System

1. `OverlapZone` struct with mask storage
2. `OverlapGate` struct with forward pass
3. Adaptive merge in the `merge_updates` method (safety override + learned gate)
4. Topology validation including overlaps
5. Default overlap layout from Section 5.4
6. Unit tests for overlap zones and merge logic

### Phase 3: Memory

1. `MemoryCircuit` with decay and write-gating
2. `HierarchicalMemory` (short/medium/long-term)
3. `EpisodicMemory` with HNSW index, store, and recall
4. Integration with mesh step cycle
5. Unit and integration tests for memory

### Phase 4: I/O

1. `WitsSnapshot` struct
2. Input encoding (normalisation, baseline stats)
3. Output decoding (`MeshOutput`, `Detection`, `CausalAnalysis`)
4. `BaselineStats` accumulation during shadow mode
5. Tests for encoding/decoding round-trips

### Phase 5: Adaptation

1. `AdapterLayer` struct and forward pass
2. `ReplayBuffer` struct and mixed batch sampling
3. `ProductionMesh` (three-tier architecture)
4. Online learning step
5. Tests for adapter training and replay

### Phase 6: Training Pipeline

1. Stage 1: Overlap pre-training (mutual information objective)
2. Stage 2: Alternating freeze training
3. Stage 3: Joint fine-tuning with regularisation
4. Post-training freeze and prune
5. Validation suite

### Phase 7: Execution Modes and Polish

1. `ExecutionMode` (Shadow, Advisory, Production)
2. `DegradationMode` and health checks
3. `MeshConfig` with all hyperparameters
4. Benchmark suite
5. Integration test suite
6. Documentation

-----

## 17. Crate Structure

```
sairen-mesh/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API, re-exports
│   ├── neuron.rs               # LiquidNeuron, compute_neuron_update
│   ├── circuit.rs              # Circuit, CircuitType, CircuitCriticality
│   ├── overlap.rs              # OverlapZone, OverlapGate, differentiable masks
│   ├── topology.rs             # TopologyConstraint, validation
│   ├── mesh.rs                 # OverlappingMesh, step(), merge_updates()
│   ├── merge.rs                # Adaptive merge logic (safety override + gate)
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── hierarchical.rs     # HierarchicalMemory, MemoryCircuit, WriteGate
│   │   └── episodic.rs         # EpisodicMemory, Episode, HNSW integration
│   ├── adaptation/
│   │   ├── mod.rs
│   │   ├── adapter.rs          # AdapterLayer, AdapterNeuron
│   │   ├── replay.rs           # ReplayBuffer, TrainingSample
│   │   └── production.rs       # ProductionMesh (three-tier)
│   ├── training/
│   │   ├── mod.rs
│   │   ├── overlap_pretrain.rs # Stage 1: Mutual information objective
│   │   ├── alternating.rs      # Stage 2: Alternating freeze
│   │   ├── joint.rs            # Stage 3: Joint fine-tuning
│   │   └── online.rs           # Online learning during drilling
│   ├── io/
│   │   ├── mod.rs
│   │   ├── wits.rs             # WitsSnapshot, parsing
│   │   ├── encoding.rs         # Input encoding, BaselineStats
│   │   └── output.rs           # MeshOutput, Detection, CausalAnalysis
│   ├── config.rs               # MeshConfig, defaults
│   ├── execution.rs            # ExecutionMode, DegradationMode
│   └── physics.rs              # MSE, ECD, d-exponent (ground truth layer)
├── tests/
│   ├── unit/
│   │   ├── neuron_tests.rs
│   │   ├── circuit_tests.rs
│   │   ├── overlap_tests.rs
│   │   ├── topology_tests.rs
│   │   ├── memory_tests.rs
│   │   ├── episodic_tests.rs
│   │   ├── adapter_tests.rs
│   │   └── encoding_tests.rs
│   ├── integration/
│   │   ├── full_step_test.rs
│   │   ├── detection_tests.rs
│   │   ├── information_flow_test.rs
│   │   ├── degradation_test.rs
│   │   └── online_learning_test.rs
│   └── benchmarks/
│       ├── step_bench.rs
│       ├── detection_bench.rs
│       ├── episodic_bench.rs
│       └── memory_bench.rs
├── examples/
│   ├── basic_mesh.rs           # Create and step a minimal mesh
│   ├── wits_simulation.rs      # Feed simulated WITS data through mesh
│   └── kick_detection.rs       # Demonstrate kick detection scenario
└── README.md
```

### Dependencies (Cargo.toml)

```toml
[package]
name = "sairen-mesh"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.8"              # Parallel computation (Phase 1 of update cycle)
rand = "0.8"               # Stochastic firing for Performance circuits
serde = { version = "1", features = ["derive"] }  # Serialisation
serde_json = "1"           # Config loading
log = "0.4"                # Logging
thiserror = "1"            # Error types
instant-distance = "0.6"   # HNSW approximate nearest neighbour (episodic memory)

[dev-dependencies]
criterion = "0.5"          # Benchmarks
approx = "0.5"             # Float comparison in tests
```

-----

## Appendix A: Physics Formulas

These are computed BEFORE the mesh receives data. The mesh operates on already-computed physics values. These formulas are provided for reference and for the `physics.rs` module.

### Mechanical Specific Energy (MSE)

```
MSE = (WOB / A_bit) + (120 * π * RPM * Torque) / (A_bit * ROP)

Where:
  A_bit = π * (bit_diameter / 2)² (bit cross-sectional area, in²)
  WOB in lbs
  RPM in rev/min
  Torque in ft-lbs
  ROP in ft/hr
  MSE in psi

Target: 35,000 - 50,000 psi (formation dependent)
```

### D-Exponent

```
d_exp = log10(ROP / (60 * RPM)) / log10(12 * WOB / (1000 * bit_diameter))

Normalised: d_exp_corrected = d_exp * (mud_weight_normal / mud_weight_actual)

Rising d_exp → harder formation or increasing pore pressure
Falling d_exp → softer formation or decreasing pore pressure
>15% shift → formation boundary
```

### Equivalent Circulating Density (ECD)

```
ECD = mud_weight + (annular_pressure_loss / (0.052 * TVD))

Where:
  mud_weight in ppg
  annular_pressure_loss in psi
  TVD in feet
  ECD in ppg

Warning: ECD margin < 0.3 ppg below fracture gradient
Critical: ECD margin < 0.1 ppg
```

### Flow Balance

```
flow_balance = flow_out - flow_in (gpm)

Positive → potential kick (formation fluid influx)
Negative → potential loss (mud into formation)

Warning: |flow_balance| > 10 gpm
Critical: |flow_balance| > 20 gpm
```

-----

## Appendix B: Circuit Catalogue

### Detection Circuits (Safety, τ = 10-20 ms)

|Circuit             |Neurons|τ    |Detects                |Key Inputs                                         |
|--------------------|-------|-----|-----------------------|---------------------------------------------------|
|Kick Detection      |204    |10 ms|Formation fluid influx |flow_balance, pit_volume, pit_rate, total_gas      |
|Loss Detection      |204    |10 ms|Mud losses to formation|flow_balance (negative), pit_volume (dropping), spp|
|Pack-off Detection  |204    |15 ms|Annular blockage       |torque (rising) + spp (rising) + rop (falling)     |
|Stick-slip Detection|204    |10 ms|Torsional oscillation  |torque CV > 12%, rpm instability                   |
|Founder Detection   |204    |20 ms|WOB/ROP decoupling     |wob (rising) + rop (flat/falling)                  |

### Causation Circuits (Safety, τ = 50 ms)

|Circuit           |Neurons|τ    |Analyses                |Reads From                   |
|------------------|-------|-----|------------------------|-----------------------------|
|Torque Causation  |341    |50 ms|Why torque is changing  |Pack-off, Stick-slip overlaps|
|Pressure Causation|341    |50 ms|Why pressure is changing|Kick, Loss overlaps          |
|Flow Causation    |341    |50 ms|Why flow balance shifted|Kick, Loss overlaps          |

### Memory Circuits (Performance, τ = 200 ms - 30 s)

|Circuit    |Neurons|τ     |Decay  |Stores                    |
|-----------|-------|------|-------|--------------------------|
|Short-term |512    |200 ms|λ=0.7  |Last few seconds of events|
|Medium-term|512    |2 s   |λ=0.07 |Current operation context |
|Long-term  |512    |30 s  |λ=0.004|Well behaviour patterns   |

### Prediction Circuits (Performance, τ = 20-30 ms)

|Circuit         |Neurons|τ    |Predicts           |Horizon       |
|----------------|-------|-----|-------------------|--------------|
|State Prediction|512    |20 ms|Next drilling state|5-30 seconds  |
|ROP Prediction  |512    |30 ms|ROP trajectory     |30-120 seconds|

-----

## End of Specification

**This document contains everything needed to implement the AASSP Neural Mesh from scratch in Rust.** Follow the build order in Section 16. Implement each phase completely with tests before moving to the next. When in doubt, refer to the design principles in Section 1.

The mesh is one component of the broader SAIREN-OS pipeline. It replaces the existing Tactical/Strategic agent LLM-based reasoning with a neural approach that is faster (sub-millisecond vs seconds), more deterministic (physics-grounded), and capable of fleet-wide learning through the episodic memory system.