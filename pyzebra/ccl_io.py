import os
from collections import defaultdict

import numpy as np

META_VARS_STR = (
    "instrument",
    "title",
    "sample",
    "user",
    "ProposalID",
    "original_filename",
    "date",
    "zebra_mode",
    "proposal",
    "proposal_user",
    "proposal_title",
    "proposal_email",
    "detectorDistance",
)

META_VARS_FLOAT = (
    "omega",
    "mf",
    "2-theta",
    "chi",
    "phi",
    "nu",
    "temp",
    "wavelenght",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "cex1",
    "cex2",
    "mexz",
    "moml",
    "mcvl",
    "momu",
    "mcvu",
    "snv",
    "snh",
    "snvm",
    "snhm",
    "s1vt",
    "s1vb",
    "s1hr",
    "s1hl",
    "s2vt",
    "s2vb",
    "s2hr",
    "s2hl",
)

META_UB_MATRIX = ("ub1j", "ub2j", "ub3j")

CCL_FIRST_LINE = (("idx", int), ("h", float), ("k", float), ("l", float))

CCL_ANGLES = {
    "bi": (("twotheta", float), ("omega", float), ("chi", float), ("phi", float)),
    "nb": (("gamma", float), ("omega", float), ("nu", float)),
}

CCL_SECOND_LINE = (
    ("n_points", int),
    ("angle_step", float),
    ("monitor", float),
    ("temp", float),
    ("mf", float),
    ("date", str),
    ("time", str),
    ("scan_type", str),
)

AREA_METHODS = ("fit_area", "int_area")


def load_1D(filepath):
    """
    Loads *.ccl or *.dat file (Distinguishes them based on last 3 chars in string of filepath
    to add more variables to read, extend the elif list
    the file must include '#data' and number of points in right place to work properly

    :arg filepath
    :returns det_variables
    - dictionary of all detector/scan variables and dictinionary for every scan.
    Names of these dictionaries are M + scan number. They include HKL indeces, angles,
    monitors, stepsize and array of counts
    """
    with open(filepath, "r") as infile:
        _, ext = os.path.splitext(filepath)
        det_variables = parse_1D(infile, data_type=ext)

    return det_variables


def parse_1D(fileobj, data_type):
    metadata = {"data_type": data_type}

    # read metadata
    for line in fileobj:
        if "=" in line:
            variable, value = line.split("=")
            variable = variable.strip()
            value = value.strip()

            if variable in META_VARS_STR:
                metadata[variable] = value

            elif variable in META_VARS_FLOAT:
                if variable == "2-theta":  # fix that angle name not to be an expression
                    variable = "twotheta"
                metadata[variable] = float(value)

            elif variable in META_UB_MATRIX:
                if "ub" not in metadata:
                    metadata["ub"] = np.zeros((3, 3))
                row = int(variable[-2]) - 1
                metadata["ub"][row, :] = list(map(float, value.split()))

        if "#data" in line:
            # this is the end of metadata and the start of data section
            break

    # read data
    scan = []
    if data_type == ".ccl":
        ccl_first_line = (*CCL_FIRST_LINE, *CCL_ANGLES[metadata["zebra_mode"]])
        ccl_second_line = CCL_SECOND_LINE

        for line in fileobj:
            s = {}

            # first line
            for param, (param_name, param_type) in zip(line.split(), ccl_first_line):
                s[param_name] = param_type(param)

            # second line
            next_line = next(fileobj)
            for param, (param_name, param_type) in zip(next_line.split(), ccl_second_line):
                s[param_name] = param_type(param)

            s["om"] = np.linspace(
                s["omega"] - (s["n_points"] / 2) * s["angle_step"],
                s["omega"] + (s["n_points"] / 2) * s["angle_step"],
                s["n_points"],
            )

            # subsequent lines with counts
            counts = []
            while len(counts) < s["n_points"]:
                counts.extend(map(float, next(fileobj).split()))
            s["Counts"] = np.array(counts)

            scan.append({**metadata, **s})

    elif data_type == ".dat":
        # skip the first 2 rows, the third row contans the column names
        next(fileobj)
        next(fileobj)
        col_names = next(fileobj).split()
        s = defaultdict(list)

        for line in fileobj:
            if "END-OF-DATA" in line:
                # this is the end of data
                break

            for name, val in zip(col_names, line.split()):
                s[name].append(float(val))

        for name in col_names:
            s[name] = np.array(s[name])

        s["h"] = s["k"] = s["l"] = float("nan")

        s["om"] = np.array(s["om"])

        for param in ("mf", "temp"):
            if param not in metadata:
                s[param] = 0

        s["n_points"] = len(s["om"])
        s["monitor"] = s["Monitor1"][0]

        s["idx"] = 1

        scan.append({**metadata, **s})

    else:
        print("Unknown file extention")

    for s in scan:
        if s["h"].is_integer() and s["k"].is_integer() and s["l"].is_integer():
            s["h"], s["k"], s["l"] = map(int, (s["h"], s["k"], s["l"]))
            s["indices"] = "hkl"
        else:
            s["indices"] = "real"

    return scan


def export_1D(data, path, area_method=AREA_METHODS[0], lorentz=False, hkl_precision=2):
    """Exports data in the .comm/.incomm format

    Scans with integer/real hkl values are saved in .comm/.incomm files correspondingly. If no scans
    are present for a particular output format, that file won't be created.
    """
    zebra_mode = data[0]["zebra_mode"]
    file_content = {".comm": [], ".incomm": []}

    for scan in data:
        if "fit" not in scan:
            continue

        idx_str = f"{scan['idx']:6}"

        h, k, l = scan["h"], scan["k"], scan["l"]
        if scan["indices"] == "hkl":
            hkl_str = f"{h:6}{k:6}{l:6}"
        else:  # scan["indices"] == "real"
            hkl_str = f"{h:8.{hkl_precision}f}{k:8.{hkl_precision}f}{l:8.{hkl_precision}f}"

        area_n = scan["fit"][area_method].n
        area_s = scan["fit"][area_method].s

        # apply lorentz correction to area
        if lorentz:
            if zebra_mode == "bi":
                twotheta = np.deg2rad(scan["twotheta"])
                corr_factor = np.sin(twotheta)
            else:  # zebra_mode == "nb":
                gamma = np.deg2rad(scan["gamma"])
                nu = np.deg2rad(scan["nu"])
                corr_factor = np.sin(gamma) * np.cos(nu)

            area_n = np.abs(area_n * corr_factor)
            area_s = np.abs(area_s * corr_factor)

        area_str = f"{area_n:10.2f}{area_s:10.2f}"

        ang_str = ""
        for angle, _ in CCL_ANGLES[zebra_mode]:
            ang_str = ang_str + f"{scan[angle]:8}"

        ref = file_content[".comm"] if scan["indices"] == "hkl" else file_content[".incomm"]
        ref.append(idx_str + hkl_str + area_str + ang_str + "\n")

    for ext, content in file_content.items():
        if content:
            with open(path + ext, "w") as out_file:
                out_file.writelines(content)
