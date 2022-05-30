import h5py
import numpy as np
from lmfit.models import Gaussian2dModel, GaussianModel

META_MATRIX = ("UB", )
META_CELL = ("cell", )
META_STR = ("name", )

def read_h5meta(filepath):
    """Open and parse content of a h5meta file.

    Args:
        filepath (str): File path of a h5meta file.

    Returns:
        dict: A dictionary with section names and their content.
    """
    with open(filepath) as file:
        content = parse_h5meta(file)

    return content


def parse_h5meta(file):
    content = dict()
    section = None
    for line in file:
        line = line.strip()
        if line.startswith("#begin "):
            section = line[len("#begin ") :]
            if section in ("detector parameters", "crystal"):
                content[section] = {}
            else:
                content[section] = []

        elif line.startswith("#end"):
            section = None

        elif section:
            if section in ("detector parameters", "crystal"):
                if "=" in line:
                    variable, value = line.split("=", 1)
                    variable = variable.strip()
                    value = value.strip()

                    if variable in META_STR:
                        pass
                    elif variable in META_CELL:
                        value = np.array(value.split(",")[:6], dtype=np.float)
                    elif variable in META_MATRIX:
                        value = np.array(value.split(",")[:9], dtype=np.float).reshape(3, 3)
                    else:  # default is a single float number
                        value = float(value)
                    content[section][variable] = value
            else:
                content[section].append(line)

    return content


def read_detector_data(filepath, cami_meta=None):
    """Read detector data and angles from an h5 file.

    Args:
        filepath (str): File path of an h5 file.

    Returns:
        ndarray: A 3D array of data, omega, gamma, nu.
    """
    with h5py.File(filepath, "r") as h5f:
        counts = h5f["/entry1/area_detector2/data"][:].astype(np.float64)

        n, cols, rows = counts.shape
        if "/entry1/experiment_identifier" in h5f:  # old format
            # reshape images (counts) to a correct shape (2006 issue)
            counts = counts.reshape(n, rows, cols)
        else:
            counts = counts.swapaxes(1, 2)

        scan = {"counts": counts, "counts_err": np.sqrt(np.maximum(counts, 1))}
        scan["original_filename"] = filepath
        scan["export"] = True

        if "/entry1/zebra_mode" in h5f:
            scan["zebra_mode"] = h5f["/entry1/zebra_mode"][0].decode()
        else:
            scan["zebra_mode"] = "nb"

        # overwrite zebra_mode from cami
        if cami_meta is not None:
            if "zebra_mode" in cami_meta:
                scan["zebra_mode"] = cami_meta["zebra_mode"][0]

        if "/entry1/control/Monitor" in h5f:
            scan["monitor"] = h5f["/entry1/control/Monitor"][0]
        else:  # old path
            scan["monitor"] = h5f["/entry1/control/data"][0]

        scan["idx"] = 1

        if "/entry1/sample/rotation_angle" in h5f:
            scan["omega"] = h5f["/entry1/sample/rotation_angle"][:]
        else:
            scan["omega"] = h5f["/entry1/area_detector2/rotation_angle"][:]
        if len(scan["omega"]) == 1:
            scan["omega"] = np.ones(n) * scan["omega"]

        scan["gamma"] = h5f["/entry1/ZEBRA/area_detector2/polar_angle"][:]
        scan["twotheta"] = h5f["/entry1/ZEBRA/area_detector2/polar_angle"][:]
        if len(scan["gamma"]) == 1:
            scan["gamma"] = np.ones(n) * scan["gamma"]
            scan["twotheta"] = np.ones(n) * scan["twotheta"]
        scan["nu"] = h5f["/entry1/ZEBRA/area_detector2/tilt_angle"][:1]
        scan["ddist"] = h5f["/entry1/ZEBRA/area_detector2/distance"][:1]
        scan["wave"] = h5f["/entry1/ZEBRA/monochromator/wavelength"][:1]
        scan["chi"] = h5f["/entry1/sample/chi"][:]
        if len(scan["chi"]) == 1:
            scan["chi"] = np.ones(n) * scan["chi"]
        scan["phi"] = h5f["/entry1/sample/phi"][:]
        if len(scan["phi"]) == 1:
            scan["phi"] = np.ones(n) * scan["phi"]
        if h5f["/entry1/sample/UB"].size == 0:
            scan["ub"] = np.eye(3) * 0.177
        else:
            scan["ub"] = h5f["/entry1/sample/UB"][:].reshape(3, 3)
        scan["name"] = h5f["/entry1/sample/name"][0].decode()
        scan["cell"] = h5f["/entry1/sample/cell"][:]

        if n == 1:
            # a default motor for a single frame file
            scan["scan_motor"] = "omega"
        else:
            for var in ("omega", "gamma", "nu", "chi", "phi"):
                if abs(scan[var][0] - scan[var][-1]) > 0.1:
                    scan["scan_motor"] = var
                    break
            else:
                raise ValueError("No angles that vary")

        scan["scan_motors"] = [scan["scan_motor"], ]

        # optional parameters
        if "/entry1/sample/magnetic_field" in h5f:
            scan["mf"] = h5f["/entry1/sample/magnetic_field"][:]

        if "/entry1/sample/temperature" in h5f:
            scan["temp"] = h5f["/entry1/sample/temperature"][:]
        elif "/entry1/sample/Ts/value" in h5f:
            scan["temp"] = h5f["/entry1/sample/Ts/value"][:]

        # overwrite metadata from .cami
        if cami_meta is not None:
            if "crystal" in cami_meta:
                cami_meta_crystal = cami_meta["crystal"]
                if "name" in cami_meta_crystal:
                    scan["name"] = cami_meta_crystal["name"]
                if "UB" in cami_meta_crystal:
                    scan["ub"] = cami_meta_crystal["UB"]
                if "cell" in cami_meta_crystal:
                    scan["cell"] = cami_meta_crystal["cell"]
                if "lambda" in cami_meta_crystal:
                    scan["wave"] = cami_meta_crystal["lambda"]

            if "detector parameters" in cami_meta:
                cami_meta_detparam = cami_meta["detector parameters"]
                if "dist2" in cami_meta_detparam:
                    scan["ddist"] = cami_meta_detparam["dist2"]

    return scan


def fit_event(scan, fr_from, fr_to, y_from, y_to, x_from, x_to):
    data_roi = scan["counts"][fr_from:fr_to, y_from:y_to, x_from:x_to]

    model = GaussianModel()
    fr = np.arange(fr_from, fr_to)
    counts_per_fr = np.sum(data_roi, axis=(1, 2))
    params = model.guess(counts_per_fr, fr)
    result = model.fit(counts_per_fr, x=fr, params=params)
    frC = result.params["center"].value
    intensity = result.params["height"].value

    counts_std = counts_per_fr.std()
    counts_mean = counts_per_fr.mean()
    snr = 0 if counts_std == 0 else counts_mean / counts_std

    model = Gaussian2dModel()
    xs, ys = np.meshgrid(np.arange(x_from, x_to), np.arange(y_from, y_to))
    xs = xs.flatten()
    ys = ys.flatten()
    counts = np.sum(data_roi, axis=0).flatten()
    params = model.guess(counts, xs, ys)
    result = model.fit(counts, x=xs, y=ys, params=params)
    xC = result.params["centerx"].value
    yC = result.params["centery"].value

    scan["fit"] = {"frame": frC, "x_pos": xC, "y_pos": yC, "intensity": intensity, "snr": snr}
