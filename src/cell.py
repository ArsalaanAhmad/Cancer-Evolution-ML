"""Cell agent definitions for the cancer clonal evolution simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class CellState(str, Enum):
    """Discrete cell states used by the agent-based model."""

    PROLIFERATING = "Proliferating"
    QUIESCENT = "Quiescent"
    HYPOXIC = "Hypoxic"
    APOPTOTIC = "Apoptotic"


@dataclass
class CancerCell:
    """Cancer cell agent with genome and stochastic evolution behaviour."""

    cell_id: int
    position: np.ndarray
    genome: np.ndarray
    age: int = 0
    state: CellState = CellState.PROLIFERATING
    fitness_score: float = 1.0
    lineage_id: int = 0

    def __post_init__(self) -> None:
        self.fitness_score = self.compute_fitness()

    @property
    def mutation_burden(self) -> int:
        return int(self.genome.sum())

    def has_driver_mutation(self, driver_bits: List[int]) -> bool:
        return any(self.genome[idx] == 1 for idx in driver_bits)

    def has_resistance_mutation(self, resistance_bit: int) -> bool:
        return bool(self.genome[resistance_bit] == 1)

    def compute_fitness(self) -> float:
        """Compute fitness as a function of selected mutation effects."""
        baseline = 1.0
        deleterious_penalty = 0.01 * max(self.mutation_burden - 2, 0)
        return max(0.2, baseline - deleterious_penalty)

    def mutate(self, mutation_rate: float, rng: np.random.Generator) -> None:
        """Stochastic bit-flip mutation using Bernoulli sampling."""
        flips = rng.random(self.genome.shape[0]) < mutation_rate
        self.genome = np.bitwise_xor(self.genome, flips.astype(np.int8))
        self.fitness_score = self.compute_fitness()

    def divide(
        self,
        new_id: int,
        rng: np.random.Generator,
        mutation_rate: float,
        displacement: np.ndarray,
    ) -> "CancerCell":
        """Asymmetric division: parent retains majority genotype, child mutates independently."""
        child_genome = self.genome.copy()

        # Asymmetric inheritance: one random bit is reset in one daughter branch.
        asym_idx = int(rng.integers(0, len(child_genome)))
        child_genome[asym_idx] = 0

        child = CancerCell(
            cell_id=new_id,
            position=self.position + displacement,
            genome=child_genome,
            age=0,
            state=CellState.PROLIFERATING,
            lineage_id=self.lineage_id,
        )
        child.mutate(mutation_rate=mutation_rate, rng=rng)

        self.age += 1
        self.mutate(mutation_rate=mutation_rate, rng=rng)

        return child

    def die(self) -> None:
        """Mark cell as apoptotic."""
        self.state = CellState.APOPTOTIC
