"""Plot all material/temperature spectra at each depth and animate them.

Run with: python sigma-piles/sigma_plot.py
Flux is per source neutron, per cm^2, per eV (no source rate assumed).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmc
from PIL import Image

from sigma import BATCHES

SCRIPT_DIR = Path(__file__).resolve().parent
LABELS = {"flibe": "FLiBe", "cllif": "ClLiF", "teflon": "PTFE", "bpe": "Borated PE (5 wt%)", "hdpe": "HDPE", "grphC": "Graphite"}


def read_case(path):
    with openmc.StatePoint(path) as sp:
        tally = sp.get_tally(name="flux_vs_depth_and_energy")
        mesh_filter = tally.find_filter(openmc.MeshFilter)
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        mesh = mesh_filter.mesh
        nx, ny, nz = mesh.dimension
        if (ny, nz) != (1, 1):
            raise ValueError(f"{path}: expected a one-dimensional depth mesh")
        if len(tally.filters) != 2:
            raise ValueError(f"{path}: expected mesh and energy filters only")
        values = tally.get_reshaped_data(value="mean")[:, :, 0, 0]
        if isinstance(tally.filters[0], openmc.EnergyFilter):
            values = values.T
        depth_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
        if not np.allclose(np.diff(depth_edges), 1.0):
            raise ValueError(f"{path}: expected 1-cm depth bins")
        energies = np.asarray(energy_filter.bins)
        cell_volume = np.prod((mesh.upper_right - mesh.lower_left) / mesh.dimension)
        spectra = values / cell_volume / (energies[:, 1] - energies[:, 0])[None, :]
    case_name = path.parent.name.removeprefix("sigma_")
    material, temperature = case_name.rsplit("_", 1)
    label = f"{LABELS.get(material, material)} — {temperature[:-1]} K"
    return label, depth_edges, energies, spectra


def export_video(output_dir, frame_count, frame_ms):
    """Export MP4 using OpenCV (pip install opencv-python)."""
    import cv2

    if frame_count <= 0 or frame_ms <= 0:
        raise ValueError("Frame count and duration must be positive")
    first = cv2.imread(str(output_dir / "flux_spectrum_depth_000.png"))
    if first is None:
        raise ValueError("Cannot read first video frame")
    height, width = first.shape[:2]
    video = output_dir / "flux_spectra_vs_depth.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             1000.0 / frame_ms, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV could not open the MP4 writer")
    try:
        for i in range(frame_count):
            path = output_dir / f"flux_spectrum_depth_{i:03d}.png"
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Unreadable or inconsistent frame: {path}")
            writer.write(frame)
    finally:
        writer.release()
    print(f"Saved {video}")
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_runs = SCRIPT_DIR / "OpenMC"
    if not default_runs.is_dir():
        default_runs = SCRIPT_DIR.parent / "OpenMC"
    parser.add_argument("--runs-dir", type=Path, default=default_runs)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "figures")
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--frame-ms", type=int, default=150)
    args = parser.parse_args()
    if args.frame_ms <= 0:
        parser.error("--frame-ms must be positive")

    paths = sorted(args.runs_dir.glob(f"sigma_*/statepoint.{args.batches}.h5"))
    if not paths:
        parser.error(f"No sigma_*/statepoint.{args.batches}.h5 files in {args.runs_dir}")
    cases = [read_case(path) for path in paths]
    _, depths, energies, _ = cases[0]
    for label, case_depths, case_energies, _ in cases[1:]:
        if not np.array_equal(depths, case_depths) or not np.array_equal(energies, case_energies):
            raise ValueError(f"Mesh or energy bins for {label} do not match the other cases")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    positive = np.concatenate([s[s > 0] for _, _, _, s in cases])
    if positive.size == 0:
        raise ValueError("No positive flux values to plot on logarithmic axes")
    ylim = (10 ** np.floor(np.log10(positive.min())),
            10 ** np.ceil(np.log10(positive.max())))
    if ylim[0] == ylim[1]:
        ylim = (ylim[0] / 10, ylim[1] * 10)
    energy_edges = np.concatenate((energies[:, 0], energies[-1:, 1]))
    frames = []
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    try:
        for i, (left, right) in enumerate(zip(depths[:-1], depths[1:])):
            ax.clear()
            for label, _, _, spectra in cases:
                # Zero-score bins are gaps, since zero cannot be shown on a log axis.
                ax.stairs(np.where(spectra[i] > 0, spectra[i], np.nan),
                          energy_edges, baseline=None, label=label, linewidth=1.2)
            ax.set(xscale="log", yscale="log", xlim=(energy_edges[0], energy_edges[-1]),
                   ylim=ylim, xlabel="Neutron energy [eV]",
                   ylabel=r"Differential flux [cm$^{-2}$ eV$^{-1}$ source neutron$^{-1}$]",
                   title=f"Flux energy spectra | depth {left:g}–{right:g} cm")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(loc="best")
            png = args.output_dir / f"flux_spectrum_depth_{i:03d}.png"
            fig.savefig(png, dpi=120)
            with Image.open(png) as frame:
                frames.append(frame.convert("RGB").quantize(colors=256))
            if (i + 1) % 10 == 0:
                print(f"Saved {i + 1}/{len(depths) - 1} depth plots", flush=True)
        gif = args.output_dir / "flux_spectra_vs_depth.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=args.frame_ms, loop=0, disposal=2, optimize=False)
        print(f"Saved {len(frames)} plots and {gif}")
        export_video(args.output_dir, len(frames), args.frame_ms)
    finally:
        plt.close(fig)
        for frame in frames:
            frame.close()


if __name__ == "__main__":
    main()
