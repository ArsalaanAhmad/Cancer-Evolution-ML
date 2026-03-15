"""Executable entrypoint for the 3D cancer clonal evolution framework."""

from __future__ import annotations

from pathlib import Path

from src.simulation import CancerEvolutionSimulation
from src.visualizer import plot_clonal_expansion, plot_tumor_3d


def main() -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    simulation = CancerEvolutionSimulation(
        grid_shape=(24, 24, 24),
        genome_length=40,
        mutation_rate=0.015,
        driver_bits=[4, 10, 15],
        resistance_bit=39,
        random_seed=12,
    )

    mutation_tracking = simulation.run(steps=320, treatment_step=180)
    biopsy_df = simulation.export_liquid_biopsy()
    mrs_df = simulation.export_multi_region_biopsy(regions=[(8, 8, 8), (12, 12, 12), (16, 16, 16)])

    mutation_tracking.to_csv(artifacts_dir / "mutation_tracking.csv", index=False)
    biopsy_df.to_csv(artifacts_dir / "liquid_biopsy_latest.csv", index=False)
    mrs_df.to_csv(artifacts_dir / "multi_region_biopsy.csv", index=False)

    plot_tumor_3d(simulation.cells, output_path=str(artifacts_dir / "tumor_3d.png"))
    plot_clonal_expansion(mutation_tracking, output_path=str(artifacts_dir / "clonal_expansion.png"))

    ttf_estimate = simulation.predict_time_to_treatment_failure()
    print(f"Live cells: {sum(c.state.value != 'Apoptotic' for c in simulation.cells)}")
    print(f"Shannon entropy: {simulation.shannon_entropy():.3f}")
    print(f"Estimated time to treatment failure (timesteps): {ttf_estimate}")


if __name__ == "__main__":
    main()
