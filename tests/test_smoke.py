"""Smoke tests for the 3D cancer clonal evolution framework."""

from __future__ import annotations

import unittest


class SimulationSmokeTest(unittest.TestCase):
    """Minimal runtime checks to validate core behavior."""

    def setUp(self) -> None:
        try:
            from src.simulation import CancerEvolutionSimulation
        except ModuleNotFoundError as exc:
            self.skipTest(f"Dependency missing in environment: {exc}")

        self.simulation_cls = CancerEvolutionSimulation

    def test_short_run_produces_mutation_tracking(self) -> None:
        simulation = self.simulation_cls(
            grid_shape=(12, 12, 12),
            genome_length=16,
            mutation_rate=0.02,
            resistance_bit=15,
            random_seed=42,
        )
        mutation_df = simulation.run(steps=20, treatment_step=10)

        self.assertFalse(mutation_df.empty)
        self.assertIn("timestep", mutation_df.columns)
        self.assertIn("mutation_burden", mutation_df.columns)

    def test_liquid_biopsy_schema(self) -> None:
        simulation = self.simulation_cls(
            grid_shape=(10, 10, 10),
            genome_length=12,
            resistance_bit=11,
            random_seed=9,
        )
        simulation.run(steps=5, treatment_step=4)
        biopsy_df = simulation.export_liquid_biopsy()

        self.assertListEqual(
            ["mutation_bit", "vaf", "timestep"],
            list(biopsy_df.columns),
        )
        self.assertEqual(len(biopsy_df), 12)


if __name__ == "__main__":
    unittest.main()
