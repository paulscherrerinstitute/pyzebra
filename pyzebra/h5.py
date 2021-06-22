import h5py
import numpy as np


META_MATRIX = ("UB")
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
                        content[section][variable] = value
                    elif variable in META_MATRIX:
                        ub_matrix = np.array(value.split(",")[:9], dtype=np.float).reshape(3, 3)
                        content[section][variable] = ub_matrix
                    else:  # default is a single float number
                        content[section][variable] = float(value)
            else:
                content[section].append(line)

    return content


def read_detector_data(filepath):
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

        if "/entry1/zebra_mode" in h5f:
            det_data["zebra_mode"] = h5f["/entry1/zebra_mode"][0].decode()
        else:
            det_data["zebra_mode"] = "nb"

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

    return det_data
