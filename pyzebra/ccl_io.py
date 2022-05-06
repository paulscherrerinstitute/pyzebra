import os
import re
from ast import literal_eval
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

META_UB_MATRIX = ("ub1j", "ub2j", "ub3j", "UB")

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
        dataset = parse_1D(infile, data_type=ext)

    return dataset


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
                if variable == "UB":
                    metadata["ub"] = np.array(literal_eval(value)).reshape(3, 3)
                else:
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
    dataset = []
    if data_type == ".ccl":
        ccl_first_line = CCL_FIRST_LINE + CCL_ANGLES[metadata["zebra_mode"]]
        ccl_second_line = CCL_SECOND_LINE

        for line in fileobj:
            # skip empty/whitespace lines before start of any scan
            if not line or line.isspace():
                continue

            scan = {}
            scan["export"] = True

            # first line
            for param, (param_name, param_type) in zip(line.split(), ccl_first_line):
                scan[param_name] = param_type(param)

            # second line
            next_line = next(fileobj)
            for param, (param_name, param_type) in zip(next_line.split(), ccl_second_line):
                scan[param_name] = param_type(param)

            if "scan_motor" not in scan:
                scan["scan_motor"] = "om"

            if scan["scan_motor"] != "om":
                raise Exception("Unsupported variable name in ccl file.")

            # "om" -> "omega"
            scan["scan_motor"] = "omega"
            scan["scan_motors"] = ["omega", ]
            # overwrite metadata, because it only refers to the scan center
            half_dist = (scan["n_points"] - 1) / 2 * scan["angle_step"]
            scan["omega"] = np.linspace(
                scan["omega"] - half_dist, scan["omega"] + half_dist, scan["n_points"]
            )

            # subsequent lines with counts
            counts = []
            while len(counts) < scan["n_points"]:
                counts.extend(map(float, next(fileobj).split()))
            scan["counts"] = np.array(counts)
            scan["counts_err"] = np.sqrt(np.maximum(scan["counts"], 1))

            if scan["h"].is_integer() and scan["k"].is_integer() and scan["l"].is_integer():
                scan["h"], scan["k"], scan["l"] = map(int, (scan["h"], scan["k"], scan["l"]))

            dataset.append({**metadata, **scan})

    elif data_type == ".dat":
        # TODO: this might need to be adapted in the future, when "gamma" will be added to dat files
        if metadata["zebra_mode"] == "nb":
            metadata["gamma"] = metadata["twotheta"]

        scan = defaultdict(list)
        scan["export"] = True

        match = re.search("Scanning Variables: (.*), Steps: (.*)", next(fileobj))
        motors = [motor.lower() for motor in match.group(1).split(", ")]
        steps = [float(step) for step in match.group(2).split()]

        match = re.search("(.*) Points, Mode: (.*), Preset (.*)", next(fileobj))
        if match.group(2) != "Monitor":
            raise Exception("Unknown mode in dat file.")
        scan["n_points"] = int(match.group(1))
        scan["monitor"] = float(match.group(3))

        col_names = list(map(str.lower, next(fileobj).split()))

        for line in fileobj:
            if "END-OF-DATA" in line:
                # this is the end of data
                break

            for name, val in zip(col_names, line.split()):
                scan[name].append(float(val))

        for name in col_names:
            scan[name] = np.array(scan[name])

        scan["counts_err"] = np.sqrt(np.maximum(scan["counts"], 1))

        scan["scan_motors"] = []
        for motor, step in zip(motors, steps):
            if step == 0:
                # it's not a scan motor, so keep only the median value
                scan[motor] = np.median(scan[motor])
            else:
                scan["scan_motors"].append(motor)

        # "om" -> "omega"
        if "om" in scan["scan_motors"]:
            scan["scan_motors"][scan["scan_motors"].index("om")] = "omega"
            scan["omega"] = scan["om"]
            del scan["om"]

        # "tt" -> "temp"
        if "tt" in scan["scan_motors"]:
            scan["scan_motors"][scan["scan_motors"].index("tt")] = "temp"
            scan["temp"] = scan["tt"]
            del scan["tt"]

        # "mf" stays "mf"
        # "phi" stays "phi"

        scan["scan_motor"] = scan["scan_motors"][0]

        if "h" not in scan:
            scan["h"] = scan["k"] = scan["l"] = float("nan")

        for param in ("mf", "temp"):
            if param not in metadata:
                scan[param] = 0

        scan["idx"] = 1

        dataset.append({**metadata, **scan})

    else:
        print("Unknown file extention")

    return dataset


def export_1D(dataset, path, export_target, hkl_precision=2):
    """Exports data in the .comm/.incomm format for fullprof or .col/.incol format for jana.

    Scans with integer/real hkl values are saved in .comm/.incomm or .col/.incol files
    correspondingly. If no scans are present for a particular output format, that file won't be
    created.
    """
    if export_target not in EXPORT_TARGETS:
        raise ValueError(f"Unknown export target: {export_target}.")

    zebra_mode = dataset[0]["zebra_mode"]
    exts = EXPORT_TARGETS[export_target]
    file_content = {ext: [] for ext in exts}

    for scan in dataset:
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


def export_ccl_compare(dataset1, dataset2, path, export_target, hkl_precision=2):
    """Exports compare data in the .comm/.incomm format for fullprof or .col/.incol format for jana.

    Scans with integer/real hkl values are saved in .comm/.incomm or .col/.incol files
    correspondingly. If no scans are present for a particular output format, that file won't be
    created.
    """
    if export_target not in EXPORT_TARGETS:
        raise ValueError(f"Unknown export target: {export_target}.")

    zebra_mode = dataset1[0]["zebra_mode"]
    exts = EXPORT_TARGETS[export_target]
    file_content = {ext: [] for ext in exts}

    for scan1, scan2 in zip(dataset1, dataset2):
        if "fit" not in scan1:
            continue

        idx_str = f"{scan1['idx']:6}"

        h, k, l = scan1["h"], scan1["k"], scan1["l"]
        hkl_are_integers = isinstance(h, int)  # if True, other indices are of type 'int' too
        if hkl_are_integers:
            hkl_str = f"{h:4}{k:4}{l:4}"
        else:
            hkl_str = f"{h:8.{hkl_precision}f}{k:8.{hkl_precision}f}{l:8.{hkl_precision}f}"

        area_n1, area_s1 = scan1["area"]
        area_n2, area_s2 = scan2["area"]
        area_n = area_n1 - area_n2
        area_s = np.sqrt(area_s1 ** 2 + area_s2 ** 2)
        area_str = f"{area_n:10.2f}{area_s:10.2f}"

        ang_str = ""
        for angle, _ in CCL_ANGLES[zebra_mode]:
            if angle == scan1["scan_motor"]:
                angle_center = (np.min(scan1[angle]) + np.max(scan1[angle])) / 2
            else:
                angle_center = scan1[angle]

            if angle == "twotheta" and export_target == "jana":
                angle_center /= 2

            ang_str = ang_str + f"{angle_center:8g}"

        if export_target == "jana":
            ang_str = ang_str + f"{scan1['temp']:8}" + f"{scan1['monitor']:8}"

        ref = file_content[exts[0]] if hkl_are_integers else file_content[exts[1]]
        ref.append(idx_str + hkl_str + area_str + ang_str + "\n")

    for ext, content in file_content.items():
        if content:
            with open(path + ext, "w") as out_file:
                out_file.writelines(content)


def export_param_study(dataset, param_data, path):
    file_content = []
    for scan, param in zip(dataset, param_data):
        if "fit" not in scan:
            continue

        if not file_content:
            title_str = f"{'param':12}"
            for fit_param_name in scan["fit"].params:
                title_str = title_str + f"{fit_param_name:20}" + f"{'std_' + fit_param_name:20}"
            title_str = title_str + "file"
            file_content.append(title_str + "\n")

        param_str = f"{param:<12.2f}"

        fit_str = ""
        for fit_param in scan["fit"].params.values():
            fit_param_val = fit_param.value
            fit_param_std = fit_param.stderr
            if fit_param_std is None:
                fit_param_std = 0
            fit_str = fit_str + f"{fit_param_val:<20.2f}" + f"{fit_param_std:<20.2f}"

        _, fname_str = os.path.split(scan["original_filename"])

        file_content.append(param_str + fit_str + fname_str + "\n")

    if file_content:
        with open(path, "w") as out_file:
            out_file.writelines(file_content)
