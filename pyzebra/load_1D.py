import os
import re
from collections import defaultdict
from decimal import Decimal

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
    # the first element is `scan_number`, which we don't save to metadata
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
    ("number_of_measurements", int),
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
        decimal = list()

        if metadata["zebra_mode"] == "bi":
            ccl_first_line = CCL_FIRST_LINE_BI
        elif metadata["zebra_mode"] == "nb":
            ccl_first_line = CCL_FIRST_LINE_NB
        ccl_second_line = CCL_SECOND_LINE

        for line in fileobj:
            d = {}

            # first line
            scan_number, *params = line.split()
            for param, (param_name, param_type) in zip(params, ccl_first_line):
                d[param_name] = param_type(param)

            decimal.append(bool(Decimal(d["h_index"]) % 1 == 0))
            decimal.append(bool(Decimal(d["k_index"]) % 1 == 0))
            decimal.append(bool(Decimal(d["l_index"]) % 1 == 0))

            # second line
            next_line = next(fileobj)
            params = next_line.split()
            for param, (param_name, param_type) in zip(params, ccl_second_line):
                d[param_name] = param_type(param)

            d["om"] = np.linspace(
                d["omega_angle"] - (d["number_of_measurements"] / 2) * d["angle_step"],
                d["omega_angle"] + (d["number_of_measurements"] / 2) * d["angle_step"],
                d["number_of_measurements"],
            )

            # subsequent lines with counts
            counts = []
            while len(counts) < d["number_of_measurements"]:
                counts.extend(map(int, next(fileobj).split()))
            d["Counts"] = counts

            scan[int(scan_number)] = d

            if all(decimal):
                metadata["indices"] = "hkl"
            else:
                metadata["indices"] = "real"

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
        data_cols["mag_field"] = metadata["mf"]
        data_cols["omega_angle"] = metadata["omega"]
        data_cols["number_of_measurements"] = len(data_cols["om"])
        data_cols["monitor"] = data_cols["Monitor1"][0]
        data_cols["twotheta_angle"] = metadata["2-theta"]
        data_cols["chi_angle"] = metadata["chi"]
        data_cols["phi_angle"] = metadata["phi"]
        data_cols["nu_angle"] = metadata["nu"]

        scan[1] = dict(data_cols)

    else:
        print("Unknown file extention")

    # utility information
    metadata["data_type"] = data_type
    metadata["area_method"] = "fit"

    return {"meta": metadata, "scan": scan}
