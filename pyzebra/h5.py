import h5py
import numpy as np


META_MATRIX = ("UB")
META_CELL = ("cell")
META_STR = ("name")

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
        data = h5f["/entry1/area_detector2/data"][:]

        # reshape data to a correct shape (2006 issue)
        n, cols, rows = data.shape
        data = data.reshape(n, rows, cols)

        det_data = {"data": data}
        det_data["original_filename"] = filepath

        if "/entry1/zebra_mode" in h5f:
            det_data["zebra_mode"] = h5f["/entry1/zebra_mode"][0].decode()
        else:
            det_data["zebra_mode"] = "nb"

        # overwrite zebra_mode from cami
        if cami_meta is not None:
            if "zebra_mode" in cami_meta:
                det_data["zebra_mode"] = cami_meta["zebra_mode"][0]

        # om, sometimes ph
        if det_data["zebra_mode"] == "nb":
            det_data["omega"] = h5f["/entry1/area_detector2/rotation_angle"][:]
        else:  # bi
            det_data["omega"] = h5f["/entry1/sample/rotation_angle"][:]

        det_data["gamma"] = h5f["/entry1/ZEBRA/area_detector2/polar_angle"][:]  # gammad
        det_data["nu"] = h5f["/entry1/ZEBRA/area_detector2/tilt_angle"][:]  # nud
        det_data["ddist"] = h5f["/entry1/ZEBRA/area_detector2/distance"][:]
        det_data["wave"] = h5f["/entry1/ZEBRA/monochromator/wavelength"][:]
        det_data["chi"] = h5f["/entry1/sample/chi"][:]  # ch
        det_data["phi"] = h5f["/entry1/sample/phi"][:]  # ph
        det_data["ub"] = h5f["/entry1/sample/UB"][:].reshape(3, 3)
        det_data["name"] = h5f["/entry1/sample/name"][0].decode()
        det_data["cell"] = h5f["/entry1/sample/cell"][:]

        if n == 1:
            # a default motor for a single frame file
            det_data["scan_motor"] = "omega"
        else:
            for var in ("omega", "gamma", "nu", "chi", "phi"):
                if abs(det_data[var][0] - det_data[var][-1]) > 0.1:
                    det_data["scan_motor"] = var
                    break
            else:
                raise ValueError("No angles that vary")

        # optional parameters
        if "/entry1/sample/magnetic_field" in h5f:
            det_data["mf"] = h5f["/entry1/sample/magnetic_field"][:]

        if "/entry1/sample/temperature" in h5f:
            det_data["temp"] = h5f["/entry1/sample/temperature"][:]

        # overwrite metadata from .cami
        if cami_meta is not None:
            if "crystal" in cami_meta:
                cami_meta_crystal = cami_meta["crystal"]
                if "name" in cami_meta_crystal:
                    det_data["name"] = cami_meta_crystal["name"]
                if "UB" in cami_meta_crystal:
                    det_data["ub"] = cami_meta_crystal["UB"]
                if "cell" in cami_meta_crystal:
                    det_data["cell"] = cami_meta_crystal["cell"]
                if "lambda" in cami_meta_crystal:
                    det_data["wave"] = cami_meta_crystal["lambda"]

            if "detector parameters" in cami_meta:
                cami_meta_detparam = cami_meta["detector parameters"]
                if "dist2" in cami_meta_detparam:
                    det_data["ddist"] = cami_meta_detparam["dist2"]

    return det_data
