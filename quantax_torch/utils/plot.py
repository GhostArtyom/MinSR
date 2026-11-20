"""Shared font settings and training plots for exported figures."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

FULL_LINE_WIDTH = 0.6  # Points
ROLLING_MARKER_SIZE = 4.0
ROLLING_LINE_WIDTH = 1.0


def setup_plot_font() -> None:
    """Require installed Arial, embed PDF fonts, and keep SVG text editable."""
    try:
        path = font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        # Newly installed faces may be missing from Matplotlib's font cache.
        for candidate in font_manager.findSystemFonts():
            try:
                if font_manager.get_font(candidate).family_name == "Arial":
                    font_manager.fontManager.addfont(candidate)
            except (OSError, RuntimeError):
                continue
        try:
            path = font_manager.findfont("Arial", fallback_to_default=False)
        except ValueError as error:
            raise RuntimeError("Install Arial before generating figures.") from error
    if font_manager.get_font(path).family_name != "Arial":
        raise RuntimeError("Install Arial before generating figures.")
    mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})


def save_training_plots(energy, variance, run_dir, title, step, baseline=None, suffix=""):
    """Save full and rolling views of the energy and variance traces."""
    run_dir = Path(run_dir)
    batch = min(10, step)
    series = (
        (energy, "energy", "Energy", baseline),
        (variance, "VarE", "Energy variance", None),
    )
    views = (
        (None, "", FULL_LINE_WIDTH),
        (-1000, "_rolling", ROLLING_LINE_WIDTH),
    )
    for start, view_suffix, line_width in views:
        for tracer, prefix, ylabel, reference in series:
            fig, ax = plt.subplots()
            try:
                with mpl.rc_context({"lines.linewidth": line_width}):
                    tracer.plot(batch=batch, start=start, baseline=reference)
                if start is None:
                    ax.lines[-1].set_marker("None")
                    ax.set_xlim(left=0)
                else:
                    ax.lines[-1].set(marker="o", markersize=ROLLING_MARKER_SIZE, markeredgewidth=0)
                    ax.set_xlim(left=tracer.time[start:][0])
                ax.set(title=title, xlabel="Update index", ylabel=ylabel)
                fig.tight_layout()
                fig.savefig(run_dir / f"{prefix}{suffix}{view_suffix}.pdf")
            finally:
                plt.close(fig)
