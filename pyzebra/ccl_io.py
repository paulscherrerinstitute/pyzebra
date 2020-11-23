import os
import re
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

CCL_FIRST_LINE = (
    ("scan_number", int),
    ("h_index", float),
    ("k_index", float),
    ("l_index", float),
)

CCL_ANGLES = {
    "bi": (
        ("twotheta_angle", float),
        ("omega_angle", float),
        ("chi_angle", float),
        ("phi_angle", float),
    ),
    "nb": (
        ("gamma_angle", float),
        ("omega_angle", float),
        ("nu_angle", float),
        ("unkwn_angle", float),
    ),
}

CCL_SECOND_LINE = (
    ("n_points", int),
    ("angle_step", float),
    ("monitor", float),
    ("temperature", float),
    ("mag_field", float),
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
    # read metadata
    metadata = {}
    for line in fileobj:
        if "=" in line:
            variable, value = line.split("=")
            variable = variable.strip()
            if variable in META_VARS_FLOAT:
                metadata[variable] = float(value)
            elif variable in META_VARS_STR:
                metadata[variable] = str(value)[:-1].strip()
            elif variable in META_UB_MATRIX:
                metadata[variable] = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))

        if "#data" in line:
            # this is the end of metadata and the start of data section
            break

    # read data
    scan = {}
    if data_type == ".ccl":
        ccl_first_line = (*CCL_FIRST_LINE, *CCL_ANGLES[metadata["zebra_mode"]])
        ccl_second_line = CCL_SECOND_LINE

        for line in fileobj:
            d = {}

            # first line
            for param, (param_name, param_type) in zip(line.split(), ccl_first_line):
                d[param_name] = param_type(param)

            # second line
            next_line = next(fileobj)
            for param, (param_name, param_type) in zip(next_line.split(), ccl_second_line):
                d[param_name] = param_type(param)

            d["om"] = np.linspace(
                d["omega_angle"] - (d["n_points"] / 2) * d["angle_step"],
                d["omega_angle"] + (d["n_points"] / 2) * d["angle_step"],
                d["n_points"],
            )

            # subsequent lines with counts
            counts = []
            while len(counts) < d["n_points"]:
                counts.extend(map(int, next(fileobj).split()))
            d["Counts"] = counts

            scan[d["scan_number"]] = d

    elif data_type == ".dat":
        # skip the first 2 rows, the third row contans the column names
        next(fileobj)
        next(fileobj)
        col_names = next(fileobj).split()
        data_cols = defaultdict(list)

        for line in fileobj:
            if "END-OF-DATA" in line:
                # this is the end of data
                break

            for name, val in zip(col_names, line.split()):
                data_cols[name].append(float(val))

        try:
            data_cols["h_index"] = float(metadata["title"].split()[-3])
            data_cols["k_index"] = float(metadata["title"].split()[-2])
            data_cols["l_index"] = float(metadata["title"].split()[-1])
        except (ValueError, IndexError):
            print("seems hkl is not in title")

        data_cols["om"] = np.array(data_cols["om"])

        data_cols["temperature"] = metadata["temp"]
        try:
            data_cols["mag_field"] = metadata["mf"]
        except KeyError:
            print("Mag_field not present in dat file")

        data_cols["omega_angle"] = metadata["omega"]
        data_cols["n_points"] = len(data_cols["om"])
        data_cols["monitor"] = data_cols["Monitor1"][0]
        data_cols["twotheta_angle"] = metadata["2-theta"]
        data_cols["chi_angle"] = metadata["chi"]
        data_cols["phi_angle"] = metadata["phi"]
        data_cols["nu_angle"] = metadata["nu"]

        data_cols["scan_number"] = 1
        scan[data_cols["scan_number"]] = dict(data_cols)

    else:
        print("Unknown file extention")

    # utility information
    if all(
        s["h_index"].is_integer() and s["k_index"].is_integer() and s["l_index"].is_integer()
        for s in scan.values()
    ):
        metadata["indices"] = "hkl"
    else:
        metadata["indices"] = "real"

    metadata["data_type"] = data_type
    metadata["area_method"] = AREA_METHODS[0]

    return {"meta": metadata, "scan": scan}


def export_comm(data, path, lorentz=False):
    """exports data in the *.comm format
    :param lorentz: perform Lorentz correction
    :param path: path to file + name
    :arg data - data to export, is dict after peak fitting

    """
    zebra_mode = data["meta"]["zebra_mode"]
    if data["meta"]["indices"] == "hkl":
        extension = ".comm"
    else:  # data["meta"]["indices"] == "real":
        extension = ".incomm"

    with open(str(path + extension), "w") as out_file:
        for key, scan in data["scan"].items():
            if "fit" not in scan:
                print("Scan skipped - no fit value for:", key)
                continue

            scan_str = f"{key:6}"

            h, k, l = scan["h_index"], scan["k_index"], scan["l_index"]
            if data["meta"]["indices"] == "hkl":
                hkl_str = f"{int(h):6}{int(k):6}{int(l):6}"
            else:  # data["meta"]["indices"] == "real"
                hkl_str = f"{h:6.2f}{k:6.2f}{l:6.2f}"

            area_method = data["meta"]["area_method"]
            area_n = scan["fit"][area_method].n
            area_s = scan["fit"][area_method].s

            # apply lorentz correction to area
            if lorentz:
                if zebra_mode == "bi":
                    twotheta_angle = np.deg2rad(scan["twotheta_angle"])
                    corr_factor = np.sin(twotheta_angle)
                else:  # zebra_mode == "nb":
                    gamma_angle = np.deg2rad(scan["gamma_angle"])
                    nu_angle = np.deg2rad(scan["nu_angle"])
                    corr_factor = np.sin(gamma_angle) * np.cos(nu_angle)

                area_n = np.abs(area_n * corr_factor)
                area_s = np.abs(area_s * corr_factor)

            area_str = f"{area_n:10.2f}{area_s:10.2f}"

            ang_str = ""
            for angle, _ in CCL_ANGLES[zebra_mode]:
                ang_str = ang_str + f"{scan[angle]:8}"

            out_file.write(scan_str + hkl_str + area_str + ang_str + "\n")
