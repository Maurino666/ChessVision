from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────
def write_metric_plots(
    stats_per_epoch: Sequence[Dict[str, float]],
    output_dir: str | Path,
) -> None:
    """
    Draw one PNG per metric.  Each element in `stats_per_epoch`
    **must** contain an 'epoch' key (int).
    """
    output_dir = Path(output_dir) / "metric_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not stats_per_epoch:
        print("[Plot] nothing to plot.")
        return

    # --- sort by epoch so X-axis is monotonic
    stats_per_epoch = sorted(stats_per_epoch, key=lambda d: d["epoch"])

    metric_keys = [k for k in stats_per_epoch[0] if k != "epoch"]

    for key in metric_keys:
        epochs = [d["epoch"] for d in stats_per_epoch]
        values = [d[key] for d in stats_per_epoch]

        plt.figure()
        plt.plot(epochs, values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"{key} vs. epoch")
        plt.grid(True)

        filename = output_dir / f"{key}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"[Plot] saved {filename}")