"""Tally and plot neutron leakage on the six vacuum boundary faces."""

import hashlib
import json
from pathlib import Path

import numpy as np
import openmc
import pandas as pd

from python.parameters import (
    EXIT_ENERGY_MIN_EV, EXIT_ENERGY_MAX_EV, EXIT_MODE_MAX_REL_ERROR,
)


FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
# Use a finite sentinel: some OpenMC XML readers do not retain an 'inf' edge.
# This is above all neutron transport energies in evaluated nuclear libraries.
ENERGY_CEILING_EV = 1e30


def exit_grid_id(pixel_cm, energy_bins):
    """Keep cached statepoints separate when the energy grid changes."""
    grid = np.asarray([pixel_cm, energy_bins, EXIT_ENERGY_MIN_EV,
                       EXIT_ENERGY_MAX_EV, ENERGY_CEILING_EV], dtype="<f8")
    return hashlib.sha256(grid.tobytes()).hexdigest()[:8]


def build_exit_tallies(boundary, pixel_cm, energy_bins):
    """Use a surface filter for leakage and a mesh filter for face pixels.

    One mesh dimension spans the normal direction. This avoids storing unused
    interior mesh surfaces. Signed currents are converted to outward currents
    during export. All six physical surfaces must have vacuum boundaries.
    """
    lower, upper = (-boundary).bounding_box
    lower, upper = np.asarray(lower), np.asarray(upper)
    dimensions = np.maximum(1, np.ceil((upper - lower) / pixel_cm).astype(int))
    energy = openmc.EnergyFilter(np.r_[
        0.0, np.geomspace(EXIT_ENERGY_MIN_EV, EXIT_ENERGY_MAX_EV, energy_bins + 1),
        ENERGY_CEILING_EV,
    ])
    particle = openmc.ParticleFilter(["neutron"])
    tallies = []
    for axis in range(3):
        mesh = openmc.RegularMesh(name=f"neutron exit {'xyz'[axis]} pixels")
        shape = dimensions.copy()
        shape[axis] = 1
        mesh.dimension = shape
        mesh_lower, mesh_upper = lower.copy(), upper.copy()
        # Both physical faces must be strictly inside the normal mesh extent.
        # The surface filter still selects only the physical boundary itself.
        padding = (upper[axis] - lower[axis]) * 1e-9
        mesh_lower[axis] -= padding
        mesh_upper[axis] += padding
        mesh.lower_left, mesh.upper_right = mesh_lower, mesh_upper
        pixels = openmc.MeshFilter(mesh)
        for side in ("min", "max"):
            face = f"{'xyz'[axis]}_{side}"
            surface = getattr(boundary, face.replace("_", ""))
            if surface.boundary_type != "vacuum":
                raise ValueError("Neutron exit maps require vacuum pile boundaries.")
            filters = [openmc.SurfaceFilter(surface), pixels, particle]
            # The separate energy-integrated tally retains the correct variance.
            # Energy-bin errors cannot be summed without their covariances.
            for suffix, selected_filters in (
                    ("current", filters), ("spectrum", filters + [energy])):
                tally = openmc.Tally(name=f"neutron exit {face} {suffix}")
                tally.filters = selected_filters
                tally.scores = ["current"]
                tally.estimator = "analog"
                tallies.append(tally)
    return tallies


def _face_array(tally, value, axis):
    """Return [vertical pixel, horizontal pixel, energy bin] in global axes."""
    pixels = tally.find_filter(openmc.MeshFilter)
    axes = [i for i in range(3) if i != axis]
    indices = np.asarray(pixels.bins, dtype=int) - 1
    data = tally.get_reshaped_data(value=value).reshape(len(indices), -1)
    shape = (pixels.mesh.dimension[axes[1]], pixels.mesh.dimension[axes[0]],
             data.shape[1])
    result = np.empty(shape)
    result[indices[:, axes[1]], indices[:, axes[0]]] = data
    return result


def _read_face(statepoint, face, source_rate):
    axis = "xyz".index(face[0])
    sign = -1 if face.endswith("min") else 1
    current = statepoint.get_tally(name=f"neutron exit {face} current")
    spectrum = statepoint.get_tally(name=f"neutron exit {face} spectrum")
    mesh = current.find_filter(openmc.MeshFilter).mesh
    axes = [i for i in range(3) if i != axis]
    edges = [np.linspace(mesh.lower_left[i], mesh.upper_right[i],
                         mesh.dimension[i] + 1) for i in axes]
    energy_edges = np.asarray(spectrum.find_filter(openmc.EnergyFilter).values)
    if energy_edges[0] != 0. or energy_edges[-1] != ENERGY_CEILING_EV:
        raise ValueError("Exit energy guard bins are missing. Run neutron transport again.")
    mean = sign * _face_array(spectrum, "mean", axis)
    std_dev = _face_array(spectrum, "std_dev", axis)
    current_mean = sign * _face_array(current, "mean", axis)[..., 0]
    current_std_dev = _face_array(current, "std_dev", axis)[..., 0]
    if np.any(mean < -1e-12) or np.any(current_mean < -1e-12):
        raise ValueError(f"Negative outward neutron current on {face}.")
    if not np.allclose(mean.sum(axis=-1), current_mean, rtol=1e-7, atol=1e-12):
        raise ValueError(f"The energy spectrum does not sum to the current on {face}.")

    # Exclude the zero-to-minimum and maximum-to-ceiling guard bins from
    # dJ/d(log E). If a guard bin dominates, do not assign it a finite mode.
    regular = mean[..., 1:-1] / np.diff(np.log(energy_edges[1:-1]))
    winner = np.argmax(regular, axis=-1)
    winner_mean = np.take_along_axis(mean[..., 1:-1], winner[..., None], -1)[..., 0]
    winner_std = np.take_along_axis(std_dev[..., 1:-1], winner[..., None], -1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mode_error = winner_std / winner_mean
        relative_error = current_std_dev / current_mean
    mode_lower = energy_edges[1:-2][winner]
    mode_upper = energy_edges[2:-1][winner]
    mode = np.sqrt(mode_lower * mode_upper)
    guard_dominant = np.maximum(mean[..., 0], mean[..., -1]) >= winner_mean
    valid = (winner_mean > 0) & ~guard_dominant
    mode = np.where(valid, mode, np.nan)
    mode_lower = np.where(valid, mode_lower, np.nan)
    mode_upper = np.where(valid, mode_upper, np.nan)
    reliable = valid & np.isfinite(mode_error) & (mode_error <= EXIT_MODE_MAX_REL_ERROR)
    area = np.multiply.outer(np.diff(edges[1]), np.diff(edges[0]))
    scale = source_rate if source_rate is not None else 1.0
    return {
        "face": np.asarray(face),
        "u_axis": np.asarray("xyz"[axes[0]]),
        "v_axis": np.asarray("xyz"[axes[1]]),
        "u_edges_cm": edges[0], "v_edges_cm": edges[1],
        "energy_edges_eV": energy_edges,
        "pixel_area_cm2": area,
        "spectrum_mean": mean, "spectrum_std_dev": std_dev,
        "current_mean": current_mean, "current_std_dev": current_std_dev,
        "current_density": current_mean * scale / area,
        "current_density_std_dev": current_std_dev * scale / area,
        "current_relative_error": relative_error,
        "modal_energy_eV": mode,
        "modal_energy_lower_eV": mode_lower, "modal_energy_upper_eV": mode_upper,
        "mode_relative_error": mode_error, "mode_reliable": reliable,
        "spectrum_units": np.asarray("neutrons / pixel / source neutron / energy bin"),
        "current_density_units": np.asarray(
            "neutrons / cm2 / s" if source_rate is not None
            else "neutrons / cm2 / source neutron"),
    }


def _plot_faces(faces, output_dir):
    # Import only during export, and use a backend that does not need a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    for key, filename, label in (
            ("current_density", "exit_current.png", str(faces[0]["current_density_units"])),
            ("modal_energy_eV", "modal_energy.png", "Modal energy [eV]; mode of dJ/d(ln E)")):
        arrays = [np.ma.masked_invalid(face[key]) for face in faces]
        if key == "modal_energy_eV":
            arrays = [np.ma.masked_where(~face["mode_reliable"], array)
                      for face, array in zip(faces, arrays)]
        arrays = [np.ma.masked_less_equal(array, 0) for array in arrays]
        positive = np.concatenate([array.compressed() for array in arrays])
        vmin, vmax = (float(positive.min()), float(positive.max())) if positive.size else (1.0, 10.0)
        if vmin == vmax:
            vmin, vmax = vmin / 2, vmax * 2
        fig, panels = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        cmap = plt.get_cmap("viridis").with_extremes(bad="#dddddd")
        for panel, face, array in zip(panels.flat, faces, arrays):
            image = panel.pcolormesh(face["u_edges_cm"], face["v_edges_cm"], array,
                                    cmap=cmap, norm=LogNorm(vmin, vmax), shading="flat")
            panel.set(title=str(face["face"]), xlabel=f"{face['u_axis']} [cm]",
                      ylabel=f"{face['v_axis']} [cm]", aspect="equal")
        fig.colorbar(image, ax=list(panels.flat), label=label, shrink=0.85)
        if key == "modal_energy_eV":
            fig.suptitle(f"Gray: no finite mode or mode-bin relative error exceeds {EXIT_MODE_MAX_REL_ERROR:.0%}")
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def export_exit_maps(statepoint, output_dir, source_rate=None):
    """Save six spectra, pixel tables, two six-face maps, and a manifest.

    Spectrum arrays always retain per-source values. Only current_density is
    scaled by source_rate. The two guard bins retain all out-of-range leakage.
    """
    if source_rate is not None and (not np.isfinite(source_rate) or source_rate <= 0):
        raise ValueError("Source rate must be positive and finite.")
    output_dir = Path(output_dir)
    faces = [_read_face(statepoint, face, source_rate) for face in FACES]
    summary = {
        "source_rate_neutrons_per_s": source_rate,
        "spectrum_array_order": ["v_pixel", "u_pixel", "energy_bin"],
        "pixel_index_base": 0,
        "spectrum_units": "neutrons / pixel / source neutron / energy bin",
        "current_units": "neutrons / pixel / source neutron",
        "current_density_units": str(faces[0]["current_density_units"]),
        "statistical_error": "One standard error of the estimated mean",
        "batches": int(statepoint.n_realizations),
        "map_axes": "Global coordinates; no face is mirrored.",
        "mode_definition": "Maximum dJ/d(ln E) over finite positive energy bins",
        "mode_bin_max_relative_error": EXIT_MODE_MAX_REL_ERROR,
        "mode_reliable_note": "This tests precision in the selected bin, not separation from competing peaks.",
        "guard_bins": "First: [0, minimum). Last: [maximum, 1e30 eV]. Included in spectra and currents; excluded from log-energy mode.",
        "current_uncertainty": "Taken from an independent energy-integrated tally; energy-bin covariance is not discarded.",
        "faces": {},
    }
    for face in faces:
        name = str(face["face"])
        total = float(face["current_mean"].sum())
        # The original boundary tally has no spatial or energy restrictions.
        # It provides an independent check for missed surface pixels.
        try:
            reference = statepoint.get_tally(name="current")
        except LookupError:
            reference = None
        if reference is not None:
            current = statepoint.get_tally(name=f"neutron exit {name} current")
            surface_id = current.find_filter(openmc.SurfaceFilter).bins[0]
            expected = float(reference.get_values(
                filters=[openmc.SurfaceFilter], filter_bins=[(surface_id,)]).item())
            expected *= -1 if name.endswith("min") else 1
            if not np.isclose(total, expected, rtol=1e-7, atol=1e-12):
                raise ValueError(f"Exit pixels do not sum to the boundary current on {name}.")
        summary["faces"][name] = {
            "shape_v_u_energy": list(face["spectrum_mean"].shape),
            "total_neutrons_per_source": total,
            "guard_neutrons_per_source": float(face["spectrum_mean"][..., [0, -1]].sum()),
            "reliable_mode_pixels": int(face["mode_reliable"].sum()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for face in faces:
        name = str(face["face"])
        np.savez_compressed(output_dir / f"{name}_spectra.npz", **face)
        v, u = np.indices(face["current_mean"].shape)
        table = {"u_pixel": u.ravel(), "v_pixel": v.ravel()}
        for axis in ("u", "v"):
            edges = face[f"{axis}_edges_cm"]
            index = u if axis == "u" else v
            table[f"{axis}_lower_cm"] = edges[index].ravel()
            table[f"{axis}_upper_cm"] = edges[index + 1].ravel()
        for key in ("pixel_area_cm2", "current_mean", "current_std_dev",
                    "current_density", "current_density_std_dev", "current_relative_error",
                    "modal_energy_eV", "modal_energy_lower_eV", "modal_energy_upper_eV",
                    "mode_relative_error", "mode_reliable"):
            table[key] = face[key].ravel()
        pd.DataFrame(table).to_csv(output_dir / f"{name}_pixels.csv", index=False)
    _plot_faces(faces, output_dir)
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Neutron exit maps and spectra saved to: {output_dir}")
