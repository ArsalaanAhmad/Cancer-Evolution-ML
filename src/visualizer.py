"""Visualization helpers for clonal evolution outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cell import CancerCell, CellState


def plot_tumor_3d(cells: list[CancerCell], output_path: str = "artifacts/tumor_3d.png") -> str:
    """Create a 3D scatter plot of live cells colored by mutation burden."""
    live_cells = [c for c in cells if c.state != CellState.APOPTOTIC]
    if not live_cells:
        raise ValueError("No living cells to visualize.")

    coords = np.array([c.position for c in live_cells])
    burdens = np.array([c.mutation_burden for c in live_cells])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=burdens,
        cmap="viridis",
        s=24,
        alpha=0.85,
    )
    fig.colorbar(scatter, ax=ax, label="Mutation burden")
    ax.set_title("3D Cancer Clonal Evolution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_clonal_expansion(
    mutation_df: pd.DataFrame,
    output_path: str = "artifacts/clonal_expansion.png",
) -> str:
    """Plot clonal expansion trend as mean mutation burden over time."""
    if mutation_df.empty:
        raise ValueError("Mutation table is empty.")

    trend = mutation_df.groupby("timestep", as_index=False)["mutation_burden"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend["timestep"], trend["mutation_burden"], linewidth=2)
    ax.set_title("Simulated Clonal Expansion")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean mutation burden")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
