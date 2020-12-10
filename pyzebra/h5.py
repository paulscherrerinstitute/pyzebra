import h5py


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
            content[section] = []

        elif line.startswith("#end"):
            section = None

        elif section:
            content[section].append(line)

    return content


def read_detector_data(filepath):
    """Read detector data and angles from an h5 file.

    Args:
        filepath (str): File path of an h5 file.

    Returns:
        ndarray: A 3D array of data, rot_angle, pol_angle, tilt_angle.
    """
    with h5py.File(filepath, "r") as h5f:
        data = h5f["/entry1/area_detector2/data"][:]

        # reshape data to a correct shape (2006 issue)
        n, cols, rows = data.shape
        data = data.reshape(n, rows, cols)

        det_data = {"data": data}

        det_data["zebra_mode"] = h5f.get("/entry1/zebra_mode", "nb")

        # om, sometimes ph
        if det_data["zebra_mode"] == "nb":
            det_data["rot_angle"] = h5f["/entry1/area_detector2/rotation_angle"][:]
        else: # bi
            det_data["rot_angle"] = h5f["/entry1/sample/rotation_angle"][:]

        det_data["pol_angle"] = h5f["/entry1/ZEBRA/area_detector2/polar_angle"][:]  # gammad
        det_data["tlt_angle"] = h5f["/entry1/ZEBRA/area_detector2/tilt_angle"][:]  # nud
        det_data["ddist"] = h5f["/entry1/ZEBRA/area_detector2/distance"][:]
        det_data["wave"] = h5f["/entry1/ZEBRA/monochromator/wavelength"][:]
        det_data["chi_angle"] = h5f["/entry1/sample/chi"][:]  # ch
        det_data["phi_angle"] = h5f["/entry1/sample/phi"][:]  # ph
        det_data["UB"] = h5f["/entry1/sample/UB"][:].reshape(3, 3)

        for var in ("rot_angle", "pol_angle", "tlt_angle", "chi_angle", "phi_angle"):
            if abs(det_data[var][0] - det_data[var][-1]) > 0.1:
                det_data["variable"] = det_data[var]
                det_data["variable_name"] = var
                break
        else:
            raise ValueError("No angles that vary")

        # optional parameters
        if "/entry1/sample/magnetic_field" in h5f:
            det_data["magnetic_field"] = h5f["/entry1/sample/magnetic_field"][:]

        if "/entry1/sample/temperature" in h5f:
            det_data["temperature"] = h5f["/entry1/sample/temperature"][:]

    return det_data
