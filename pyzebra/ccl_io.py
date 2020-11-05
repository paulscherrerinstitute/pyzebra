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

CCL_FIRST_LINE_BI = (
    *CCL_FIRST_LINE,
    ("twotheta_angle", float),
    ("omega_angle", float),
    ("chi_angle", float),
    ("phi_angle", float),
)

CCL_FIRST_LINE_NB = (
    *CCL_FIRST_LINE,
    ("gamma_angle", float),
    ("omega_angle", float),
    ("nu_angle", float),
    ("unkwn_angle", float),
)

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
        if metadata["zebra_mode"] == "bi":
            ccl_first_line = CCL_FIRST_LINE_BI
        elif metadata["zebra_mode"] == "nb":
            ccl_first_line = CCL_FIRST_LINE_NB
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

        scan[1] = dict(data_cols)

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
    metadata["area_method"] = "fit"

    return {"meta": metadata, "scan": scan}


def export_comm(data, path, lorentz=False):
    """exports data in the *.comm format
    :param lorentz: perform Lorentz correction
    :param path: path to file + name
    :arg data - data to export, is dict after peak fitting

    """
    zebra_mode = data["meta"]["zebra_mode"]
    align = ">"
    if data["meta"]["indices"] == "hkl":
        extension = ".comm"
        padding = [6, 4, 10, 8]
    elif data["meta"]["indices"] == "real":
        extension = ".incomm"
        padding = [4, 6, 10, 8]

    with open(str(path + extension), "w") as out_file:
        for key, scan in data["scan"].items():
            if "fit" not in scan:
                print("Scan skipped - no fit value for:", key)
                continue

            scan_number_str = f"{key:{align}{padding[0]}}"
            h_str = f'{int(scan["h_index"]):{padding[1]}}'
            k_str = f'{int(scan["k_index"]):{padding[1]}}'
            l_str = f'{int(scan["l_index"]):{padding[1]}}'
            if data["meta"]["area_method"] == "fit":
                area = float(scan["fit"]["fit_area"].n)
                sigma_str = (
                    f'{"{:8.2f}".format(float(scan["fit"]["fit_area"].s)):{align}{padding[2]}}'
                )
            elif data["meta"]["area_method"] == "integ":
                area = float(scan["fit"]["int_area"].n)
                sigma_str = (
                    f'{"{:8.2f}".format(float(scan["fit"]["int_area"].s)):{align}{padding[2]}}'
                )

            # apply lorentz correction to area
            if lorentz:
                if zebra_mode == "bi":
                    twotheta_angle = np.deg2rad(scan["twotheta_angle"])
                    corr_factor = np.sin(twotheta_angle)
                elif zebra_mode == "nb":
                    gamma_angle = np.deg2rad(scan["gamma_angle"])
                    nu_angle = np.deg2rad(scan["nu_angle"])
                    corr_factor = np.sin(gamma_angle) * np.cos(nu_angle)

                area = np.abs(area * corr_factor)

            int_str = f'{"{:8.2f}".format(area):{align}{padding[2]}}'

            if zebra_mode == "bi":
                angle_str1 = f'{scan["twotheta_angle"]:{padding[3]}}'
                angle_str2 = f'{scan["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{scan["chi_angle"]:{padding[3]}}'
                angle_str4 = f'{scan["phi_angle"]:{padding[3]}}'
            elif zebra_mode == "nb":
                angle_str1 = f'{scan["gamma_angle"]:{padding[3]}}'
                angle_str2 = f'{scan["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{scan["nu_angle"]:{padding[3]}}'
                angle_str4 = f'{scan["unkwn_angle"]:{padding[3]}}'

            line = (
                scan_number_str
                + h_str
                + k_str
                + l_str
                + int_str
                + sigma_str
                + angle_str1
                + angle_str2
                + angle_str3
                + angle_str4
                + "\n"
            )
            out_file.write(line)
