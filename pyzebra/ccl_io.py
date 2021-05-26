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

CCL_FIRST_LINE = (("idx", int), ("h", float), ("k", float), ("l", float))

CCL_ANGLES = {
    "bi": (("twotheta", float), ("omega", float), ("chi", float), ("phi", float)),
    "nb": (("gamma", float), ("omega", float), ("nu", float), ("skip_angle", float)),
}

CCL_SECOND_LINE = (
    ("n_points", int),
    ("angle_step", float),
    ("monitor", float),
    ("temp", float),
    ("mf", float),
    ("date", str),
    ("time", str),
    ("scan_motor", str),
)

EXPORT_TARGETS = {"fullprof": (".comm", ".incomm"), "jana": (".col", ".incol")}


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
            variable, value = line.split("=", 1)
            variable = variable.strip()
            value = value.strip()

            if variable in META_VARS_STR:
                metadata[variable] = value

            elif variable in META_VARS_FLOAT:
                if variable == "2-theta":  # fix that angle name not to be an expression
                    variable = "twotheta"
                if variable in ("a", "b", "c", "alpha", "beta", "gamma"):
                    variable += "_cell"
                metadata[variable] = float(value)

            elif variable in META_UB_MATRIX:
                if "ub" not in metadata:
                    metadata["ub"] = np.zeros((3, 3))
                row = int(variable[-2]) - 1
                metadata["ub"][row, :] = list(map(float, value.split()))

        if "#data" in line:
            # this is the end of metadata and the start of data section
            break

    # handle older files that don't contain "zebra_mode" metadata
    if "zebra_mode" not in metadata:
        metadata["zebra_mode"] = "nb"

    # read data
    scan = []
    if data_type == ".ccl":
        ccl_first_line = CCL_FIRST_LINE + CCL_ANGLES[metadata["zebra_mode"]]
        ccl_second_line = CCL_SECOND_LINE

        for line in fileobj:
            # skip empty/whitespace lines before start of any scan
            if not line or line.isspace():
                continue

            s = {}

            # first line
            for param, (param_name, param_type) in zip(line.split(), ccl_first_line):
                s[param_name] = param_type(param)

            # second line
            next_line = next(fileobj)
            for param, (param_name, param_type) in zip(next_line.split(), ccl_second_line):
                s[param_name] = param_type(param)

            if s["scan_motor"] != "om":
                raise Exception("Unsupported variable name in ccl file.")

            # "om" -> "omega"
            s["scan_motor"] = "omega"
            # overwrite metadata, because it only refers to the scan center
            half_dist = (s["n_points"] - 1) / 2 * s["angle_step"]
            s["omega"] = np.linspace(s["omega"] - half_dist, s["omega"] + half_dist, s["n_points"])

            # subsequent lines with counts
            counts = []
            while len(counts) < s["n_points"]:
                counts.extend(map(float, next(fileobj).split()))
            s["counts"] = np.array(counts)

            if s["h"].is_integer() and s["k"].is_integer() and s["l"].is_integer():
                s["h"], s["k"], s["l"] = map(int, (s["h"], s["k"], s["l"]))

            scan.append({**metadata, **s})

    elif data_type == ".dat":
        # TODO: this might need to be adapted in the future, when "gamma" will be added to dat files
        if metadata["zebra_mode"] == "nb":
            metadata["gamma"] = metadata["twotheta"]

        s = defaultdict(list)

        match = re.search("Scanning Variables: (.*), Steps: (.*)", next(fileobj))
        if match.group(1) == "h, k, l":
            steps = match.group(2).split()
            for step, ind in zip(steps, "hkl"):
                if float(step) != 0:
                    scan_motor = ind
                    break
        else:
            scan_motor = match.group(1)

        s["scan_motor"] = scan_motor

        match = re.search("(.*) Points, Mode: (.*), Preset (.*)", next(fileobj))
        if match.group(2) != "Monitor":
            raise Exception("Unknown mode in dat file.")
        s["monitor"] = float(match.group(3))

        col_names = next(fileobj).split()

        for line in fileobj:
            if "END-OF-DATA" in line:
                # this is the end of data
                break

            for name, val in zip(col_names, line.split()):
                s[name].append(float(val))

        for name in col_names:
            s[name] = np.array(s[name])

        # "om" -> "omega"
        if s["scan_motor"] == "om":
            s["scan_motor"] = "omega"
            s["omega"] = s["om"]
            del s["om"]

        # "tt" -> "temp"
        elif s["scan_motor"] == "tt":
            s["scan_motor"] = "temp"
            s["temp"] = s["tt"]
            del s["tt"]

        # "mf" stays "mf"
        # "phi" stays "phi"

        if "h" not in s:
            s["h"] = s["k"] = s["l"] = float("nan")

        for param in ("mf", "temp"):
            if param not in metadata:
                s[param] = 0

        s["idx"] = 1

        scan.append({**metadata, **s})

    else:
        print("Unknown file extention")

    return scan


def export_1D(data, path, export_target, hkl_precision=2):
    """Exports data in the .comm/.incomm format for fullprof or .col/.incol format for jana.

    Scans with integer/real hkl values are saved in .comm/.incomm or .col/.incol files
    correspondingly. If no scans are present for a particular output format, that file won't be
    created.
    """
    if export_target not in EXPORT_TARGETS:
        raise ValueError(f"Unknown export target: {export_target}.")

    zebra_mode = data[0]["zebra_mode"]
    exts = EXPORT_TARGETS[export_target]
    file_content = {ext: [] for ext in exts}

    for scan in data:
        if "fit" not in scan:
            continue

        idx_str = f"{scan['idx']:6}"

        h, k, l = scan["h"], scan["k"], scan["l"]
        hkl_are_integers = isinstance(h, int)  # if True, other indices are of type 'int' too
        if hkl_are_integers:
            hkl_str = f"{h:4}{k:4}{l:4}"
        else:
            hkl_str = f"{h:8.{hkl_precision}f}{k:8.{hkl_precision}f}{l:8.{hkl_precision}f}"

        area_n, area_s = scan["area"]
        area_str = f"{area_n:10.2f}{area_s:10.2f}"

        ang_str = ""
        for angle, _ in CCL_ANGLES[zebra_mode]:
            if angle == scan["scan_motor"]:
                angle_center = (np.min(scan[angle]) + np.max(scan[angle])) / 2
            else:
                angle_center = scan[angle]

            if angle == "twotheta" and export_target == "jana":
                angle_center /= 2

            ang_str = ang_str + f"{angle_center:8g}"

        if export_target == "jana":
            ang_str = ang_str + f"{scan['temp']:8}" + f"{scan['monitor']:8}"

        ref = file_content[exts[0]] if hkl_are_integers else file_content[exts[1]]
        ref.append(idx_str + hkl_str + area_str + ang_str + "\n")

    for ext, content in file_content.items():
        if content:
            with open(path + ext, "w") as out_file:
                out_file.writelines(content)
