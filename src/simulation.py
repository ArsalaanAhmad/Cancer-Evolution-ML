"""Core simulation engine for 3D cancer clonal evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import RandomForestClassifier

from src.cell import CancerCell, CellState


@njit(cache=True)
def update_diffusion_field(
    field: np.ndarray,
    diffusion_rate: float,
    consumption_grid: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Numba-accelerated FDM update for 3D diffusion with consumption."""
    nx, ny, nz = field.shape
    updated = field.copy()

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                laplacian = (
                    field[x + 1, y, z]
                    + field[x - 1, y, z]
                    + field[x, y + 1, z]
                    + field[x, y - 1, z]
                    + field[x, y, z + 1]
                    + field[x, y, z - 1]
                    - 6.0 * field[x, y, z]
                )
                updated[x, y, z] = field[x, y, z] + dt * (
                    diffusion_rate * laplacian - consumption_grid[x, y, z]
                )

    np.clip(updated, 0.0, 1.0, out=updated)
    return updated


@dataclass
class TumorMicroenvironment:
    """3D microenvironment with oxygen and nutrient diffusion fields."""

    shape: Tuple[int, int, int]
    carrying_capacity: int = 4
    oxygen_diffusion: float = 0.15
    nutrient_diffusion: float = 0.1

    def __post_init__(self) -> None:
        self.oxygen = np.ones(self.shape, dtype=np.float64)
        self.nutrient = np.ones(self.shape, dtype=np.float64)
        self.cell_density = np.zeros(self.shape, dtype=np.int16)

    def in_bounds(self, position: np.ndarray) -> bool:
        x, y, z = position.astype(int)
        nx, ny, nz = self.shape
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz

    def update_density(self, cells: List[CancerCell]) -> None:
        self.cell_density.fill(0)
        for cell in cells:
            if cell.state == CellState.APOPTOTIC:
                continue
            x, y, z = cell.position.astype(int)
            if self.in_bounds(cell.position):
                self.cell_density[x, y, z] += 1

    def can_divide(self, position: np.ndarray) -> bool:
        x, y, z = position.astype(int)
        return self.cell_density[x, y, z] < self.carrying_capacity

    def diffuse(self, cells: List[CancerCell], dt: float = 0.05) -> None:
        consumption = np.zeros(self.shape, dtype=np.float64)
        for cell in cells:
            if cell.state == CellState.APOPTOTIC:
                continue
            x, y, z = cell.position.astype(int)
            if self.in_bounds(cell.position):
                consumption[x, y, z] += 0.01

        self.oxygen = update_diffusion_field(
            self.oxygen,
            diffusion_rate=self.oxygen_diffusion,
            consumption_grid=consumption,
            dt=dt,
        )
        self.nutrient = update_diffusion_field(
            self.nutrient,
            diffusion_rate=self.nutrient_diffusion,
            consumption_grid=consumption,
            dt=dt,
        )


class CancerEvolutionSimulation:
    """Agent-based and stochastic simulator for clonal evolution dynamics."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (30, 30, 30),
        genome_length: int = 32,
        mutation_rate: float = 0.01,
        driver_bits: List[int] | None = None,
        resistance_bit: int = 31,
        random_seed: int = 7,
    ) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.microenvironment = TumorMicroenvironment(shape=grid_shape)
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.driver_bits = driver_bits or [2, 8, 13]
        self.resistance_bit = resistance_bit
        self.time_step = 0
        self.next_cell_id = 1

        self.cells: List[CancerCell] = []
        self.mutation_log: List[Dict[str, int | float]] = []
        self.vaf_snapshots: List[pd.DataFrame] = []

        center = np.array([s // 2 for s in grid_shape], dtype=np.int16)
        founder = CancerCell(
            cell_id=0,
            position=center,
            genome=np.zeros(genome_length, dtype=np.int8),
            lineage_id=0,
        )
        self.cells.append(founder)

    def _oxygen_at(self, cell: CancerCell) -> float:
        x, y, z = cell.position.astype(int)
        return float(self.microenvironment.oxygen[x, y, z])

    def _local_neighbors(self, position: np.ndarray) -> List[np.ndarray]:
        deltas = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]
        candidates: List[np.ndarray] = []
        for delta in deltas:
            candidate = position + delta
            if self.microenvironment.in_bounds(candidate) and self.microenvironment.can_divide(candidate):
                candidates.append(delta)
        return candidates

    def _division_probability(self, cell: CancerCell) -> float:
        oxygen = self._oxygen_at(cell)
        p_div = 0.20 * cell.fitness_score

        if cell.has_driver_mutation(self.driver_bits):
            p_div *= float(self.rng.uniform(1.05, 1.10))
            if oxygen < 0.25:
                p_div *= 1.30  # hypoxia-tolerant driver clone boost

        if oxygen < 0.15 and not cell.has_resistance_mutation(self.resistance_bit):
            return 0.0

        return float(np.clip(p_div, 0.0, 0.95))

    def _update_cell_state(self, cell: CancerCell) -> None:
        oxygen = self._oxygen_at(cell)
        if oxygen < 0.03 and not cell.has_resistance_mutation(self.resistance_bit):
            cell.state = CellState.APOPTOTIC
        elif oxygen < 0.08 and not cell.has_resistance_mutation(self.resistance_bit):
            cell.state = CellState.HYPOXIC
        elif cell.age > 30:
            cell.state = CellState.QUIESCENT
        else:
            cell.state = CellState.PROLIFERATING

    def _record_mutations(self, cell: CancerCell) -> None:
        self.mutation_log.append(
            {
                "timestep": self.time_step,
                "cell_id": cell.cell_id,
                "lineage_id": cell.lineage_id,
                "mutation_burden": cell.mutation_burden,
                "fitness_score": cell.fitness_score,
                "state": str(cell.state.value),
            }
        )

    def export_liquid_biopsy(self) -> pd.DataFrame:
        """Export synthetic VAF distributions from all live cells."""
        living_cells = [c for c in self.cells if c.state != CellState.APOPTOTIC]
        if not living_cells:
            return pd.DataFrame(columns=["mutation_bit", "vaf", "timestep"])

        genomes = np.vstack([cell.genome for cell in living_cells])
        vaf = genomes.mean(axis=0)
        biopsy = pd.DataFrame(
            {
                "mutation_bit": np.arange(self.genome_length, dtype=int),
                "vaf": vaf,
                "timestep": self.time_step,
            }
        )
        self.vaf_snapshots.append(biopsy)
        return biopsy

    def export_multi_region_biopsy(self, regions: List[Tuple[int, int, int]], radius: int = 3) -> pd.DataFrame:
        """Sample synthetic biopsies from multiple 3D regions to model ITH."""
        records: List[Dict[str, int | float]] = []
        for region_id, region_center in enumerate(regions):
            center = np.array(region_center)
            regional_cells = [
                c
                for c in self.cells
                if c.state != CellState.APOPTOTIC
                and np.linalg.norm(c.position - center) <= radius
            ]
            if not regional_cells:
                continue
            genomes = np.vstack([cell.genome for cell in regional_cells])
            regional_vaf = genomes.mean(axis=0)
            for bit_idx, bit_vaf in enumerate(regional_vaf):
                records.append(
                    {
                        "region_id": region_id,
                        "mutation_bit": bit_idx,
                        "vaf": float(bit_vaf),
                        "timestep": self.time_step,
                    }
                )

        return pd.DataFrame.from_records(records)

    def apply_treatment_bottleneck(self) -> None:
        """Kill 99% of non-resistant cells to model treatment selective pressure."""
        survivors: List[CancerCell] = []
        for cell in self.cells:
            if cell.has_resistance_mutation(self.resistance_bit):
                survivors.append(cell)
                continue
            if self.rng.random() < 0.01:
                survivors.append(cell)
            else:
                cell.die()

        self.cells = survivors

    def shannon_entropy(self) -> float:
        """Compute clonal diversity using mutation burden bins as proxy clones."""
        living = [c for c in self.cells if c.state != CellState.APOPTOTIC]
        if not living:
            return 0.0
        burdens = np.array([c.mutation_burden for c in living])
        _, counts = np.unique(burdens, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)

    def predict_time_to_treatment_failure(self) -> int:
        """Train a simple RF classifier and return a risk-based horizon estimate."""
        n_samples = 256
        entropy_feat = self.rng.uniform(0.0, 4.0, size=n_samples)
        burden_feat = self.rng.uniform(0, self.genome_length, size=n_samples)
        resistance_feat = self.rng.uniform(0, 1, size=n_samples)

        X = np.column_stack([entropy_feat, burden_feat, resistance_feat])
        y = ((entropy_feat > 2.0) | (resistance_feat > 0.35)).astype(int)

        model = RandomForestClassifier(n_estimators=80, random_state=42)
        model.fit(X, y)

        living = [c for c in self.cells if c.state != CellState.APOPTOTIC]
        avg_burden = float(np.mean([c.mutation_burden for c in living])) if living else 0.0
        resistance_fraction = (
            float(np.mean([c.has_resistance_mutation(self.resistance_bit) for c in living]))
            if living
            else 0.0
        )
        inference_x = np.array([[self.shannon_entropy(), avg_burden, resistance_fraction]])
        failure_risk = model.predict_proba(inference_x)[0, 1]

        # convert risk to horizon proxy in timesteps
        return int(np.clip(600 * (1 - failure_risk), 30, 600))

    def step(self) -> None:
        """Run one stochastic ABM timestep."""
        self.microenvironment.update_density(self.cells)
        self.microenvironment.diffuse(self.cells)

        new_cells: List[CancerCell] = []
        for cell in list(self.cells):
            if cell.state == CellState.APOPTOTIC:
                continue

            self._update_cell_state(cell)
            self._record_mutations(cell)

            if cell.state != CellState.PROLIFERATING:
                continue

            p_div = self._division_probability(cell)
            free_neighbor_steps = self._local_neighbors(cell.position)
            if free_neighbor_steps and self.rng.random() < p_div:
                displacement = free_neighbor_steps[int(self.rng.integers(0, len(free_neighbor_steps)))]
                child = cell.divide(
                    new_id=self.next_cell_id,
                    rng=self.rng,
                    mutation_rate=self.mutation_rate,
                    displacement=displacement,
                )
                self.next_cell_id += 1
                child.lineage_id = cell.lineage_id if cell.mutation_burden < 4 else cell.cell_id
                new_cells.append(child)

        self.cells.extend(new_cells)
        self.time_step += 1

        if self.time_step % 100 == 0:
            self.export_liquid_biopsy()

    def run(self, steps: int = 300, treatment_step: int = 180) -> pd.DataFrame:
        """Run simulation and return mutation tracking table."""
        for _ in range(steps):
            self.step()
            if self.time_step == treatment_step:
                self.apply_treatment_bottleneck()

        return pd.DataFrame(self.mutation_log)
