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
            s = {}

            # first line
            for param, (param_name, param_type) in zip(line.split(), ccl_first_line):
                s[param_name] = param_type(param)

            # second line
            next_line = next(fileobj)
            for param, (param_name, param_type) in zip(next_line.split(), ccl_second_line):
                s[param_name] = param_type(param)

            s["om"] = np.linspace(
                s["omega_angle"] - (s["n_points"] / 2) * s["angle_step"],
                s["omega_angle"] + (s["n_points"] / 2) * s["angle_step"],
                s["n_points"],
            )

            # subsequent lines with counts
            counts = []
            while len(counts) < s["n_points"]:
                counts.extend(map(int, next(fileobj).split()))
            s["Counts"] = counts

            scan[s["scan_number"]] = s

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

        try:
            s["h_index"] = float(metadata["title"].split()[-3])
            s["k_index"] = float(metadata["title"].split()[-2])
            s["l_index"] = float(metadata["title"].split()[-1])
        except (ValueError, IndexError):
            print("seems hkl is not in title")

        s["om"] = np.array(s["om"])

        s["temperature"] = metadata["temp"]
        try:
            s["mag_field"] = metadata["mf"]
        except KeyError:
            print("Mag_field not present in dat file")

        s["omega_angle"] = metadata["omega"]
        s["n_points"] = len(s["om"])
        s["monitor"] = s["Monitor1"][0]
        s["twotheta_angle"] = metadata["2-theta"]
        s["chi_angle"] = metadata["chi"]
        s["phi_angle"] = metadata["phi"]
        s["nu_angle"] = metadata["nu"]

        s["scan_number"] = 1
        scan[s["scan_number"]] = dict(s)

    else:
        print("Unknown file extention")

    # utility information
    metadata["indices"] = []
    for s in scan.values():
        if s["h_index"].is_integer() and s["k_index"].is_integer() and s["l_index"].is_integer():
            s["h_index"] = int(s["h_index"])
            s["k_index"] = int(s["k_index"])
            s["l_index"] = int(s["l_index"])
            metadata["indices"].append("hkl")
        else:
            metadata["indices"].append("real")

    metadata["data_type"] = data_type
    metadata["area_method"] = AREA_METHODS[0]

    return {"meta": metadata, "scan": scan}


def export_1D(data, path, lorentz=False, hkl_precision=2):
    """Exports data in the .comm/.incomm format

    Scans with integer/real hkl values are saved in .comm/.incomm files correspondingly. If no scans
    are present for a particular output format, that file won't be created.
    """
    zebra_mode = data["meta"]["zebra_mode"]
    file_content = {".comm": [], ".incomm": []}

    for (key, scan), indices in zip(data["scan"].items(), data["meta"]["indices"]):
        if "fit" not in scan:
            print("Scan skipped - no fit value for:", key)
            continue

        scan_str = f"{key:6}"

        h, k, l = scan["h_index"], scan["k_index"], scan["l_index"]
        if indices == "hkl":
            hkl_str = f"{h:6}{k:6}{l:6}"
        else:  # indices == "real"
            hkl_str = f"{h:8.{hkl_precision}f}{k:8.{hkl_precision}f}{l:8.{hkl_precision}f}"

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

        file_content_ref = file_content[".comm"] if indices == "hkl" else file_content[".incomm"]
        file_content_ref.append(scan_str + hkl_str + area_str + ang_str + "\n")

    for ext, content in file_content.items():
        if content:
            with open(path + ext, "w") as out_file:
                out_file.writelines(content)
