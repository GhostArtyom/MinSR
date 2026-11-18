"""Shared font settings and training plots for exported figures."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


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


def save_training_plots(energy, variance, run_dir, system_name, title, step, baseline=None, suffix=""):
    """Save the recent energy and variance traces as system-named PDFs."""
    for tracer, prefix, ylabel, reference in (
        (energy, "energy", "Energy", baseline),
        (variance, "VarE", "Energy variance", None),
    ):
        fig = plt.figure()
        try:
            tracer.plot(batch=min(10, step), start=-1000, baseline=reference)
            plt.gca().lines[-1].set_marker("o")
            plt.title(title)
            plt.xlabel("Update index")
            plt.ylabel(ylabel)
            plt.tight_layout()
            fig.savefig(Path(run_dir) / f"{prefix}_{system_name}{suffix}.pdf")
        finally:
            plt.close(fig)
