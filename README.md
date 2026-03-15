# 3D Cancer Clonal Evolution Simulation Framework

A modular Python framework for stochastic simulation and agent-based modeling (ABM) of cancer evolution in a 3D tumor microenvironment.

## Core Architecture

### 1) Agent Class: `CancerCell`
- **Genome**: fixed-length mutation bit vector (`np.ndarray[int8]`)
- **Fitness score**: mutation-burden aware fitness function
- **Age**: timestep age counter
- **CellState**: `Proliferating`, `Quiescent`, `Hypoxic`, `Apoptotic`
- **Methods**:
  - `mutate(μ)`: stochastic Bernoulli bit-flip mutation
  - `divide()`: asymmetric division with daughter-specific mutation updates
  - `die()`: apoptosis state transition

### 2) Environment Class: `TumorMicroenvironment`
- 3D NumPy grids for **Oxygen (O2)** and **Nutrient (N)**
- Finite Difference Method (FDM) diffusion updates accelerated with **Numba JIT**
- Local carrying capacity / contact inhibition (`can_divide`) to block proliferation in crowded voxels

### 3) Evolutionary Logic
- **Clonal competition**:
  - Driver mutation bits increase division probability by 5–10%
  - Driver-positive clones gain increased tolerance under low oxygen
- **Bottleneck treatment event**:
  - `apply_treatment_bottleneck()` kills ~99% of non-resistant cells
  - Resistant clones (specific resistance mutation bit) survive selective pressure

### 4) ML Inference Layer
- **Liquid biopsy export** every 100 steps as synthetic VAF distributions
- **Clonal diversity** quantified via Shannon entropy
- **Random Forest classifier** predicts a proxy `Time to Treatment Failure` from:
  - Shannon entropy
  - average mutation burden
  - resistant clone fraction

## Outputs
Running `src/main.py` generates:
- `artifacts/mutation_tracking.csv` (pandas mutation log)
- `artifacts/liquid_biopsy_latest.csv` (VAF)
- `artifacts/multi_region_biopsy.csv` (regional VAF)
- `artifacts/tumor_3d.png` (3D cell map)
- `artifacts/clonal_expansion.png` (clonal expansion trend)

## The "Privacy Math" of Synthetic Data
This framework generates **Synthetic Patient Data** during simulation.

**The pitch**: Because we generate synthetic Variant Allele Frequency (VAF) distributions from in-silico clones, the prototype phase bypasses sensitive NHS patient-level data. This supports privacy-preserving development and aligns with an **ϵ-Differential Privacy-by-design mindset** when model iteration occurs before real-world data onboarding.

## Multi-Region Sequencing (MRS) Simulation
The model supports **multi-region biopsy sampling** across different 3D tumor coordinates, not only whole-tumor aggregate sampling.

This explicitly models **Intra-Tumor Heterogeneity (ITH)** by producing region-specific VAF signatures, which is crucial for understanding clonal coexistence and treatment escape pathways.

## Real-World Benchmarking Positioning
This simulator is inspired by **TumE** and **evoCancerGPT-style evolutionary modeling principles** for clonal dynamics.

A practical benchmarking workflow is to compare simulated **clonal expansion** trajectories against empirical lung tumor cohorts, including cases that show **late-stage clonal expansions**, then calibrate model priors (driver benefits, bottleneck intensity, oxygen penalties) for improved realism.

## Run
```bash
python -m src.main
```
