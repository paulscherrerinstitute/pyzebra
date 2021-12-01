import os

import numpy as np
from lmfit.models import Gaussian2dModel, GaussianModel, LinearModel, PseudoVoigtModel, VoigtModel
from scipy.integrate import simpson, trapezoid

from .ccl_io import CCL_ANGLES

PARAM_PRECISIONS = {
    "twotheta": 0.1,
    "chi": 0.1,
    "nu": 0.1,
    "phi": 0.05,
    "omega": 0.05,
    "gamma": 0.05,
    "temp": 1,
    "mf": 0.001,
    "ub": 0.01,
}

MAX_RANGE_GAP = {
    "omega": 0.5,
}

AREA_METHODS = ("fit_area", "int_area")


def normalize_dataset(dataset, monitor=100_000):
    for scan in dataset:
        monitor_ratio = monitor / scan["monitor"]
        scan["counts"] *= monitor_ratio
        scan["counts_err"] *= monitor_ratio
        scan["monitor"] = monitor


def merge_duplicates(dataset):
    merged = np.zeros(len(dataset), dtype=np.bool)
    for ind_into, scan_into in enumerate(dataset):
        for ind_from, scan_from in enumerate(dataset[ind_into + 1 :], start=ind_into + 1):
            if _parameters_match(scan_into, scan_from) and not merged[ind_from]:
                merge_scans(scan_into, scan_from)
                merged[ind_from] = True


def _parameters_match(scan1, scan2):
    zebra_mode = scan1["zebra_mode"]
    if zebra_mode != scan2["zebra_mode"]:
        return False

    for param in ("ub", "temp", "mf", *(vars[0] for vars in CCL_ANGLES[zebra_mode])):
        if param.startswith("skip"):
            # ignore skip parameters, like the last angle in 'nb' zebra mode
            continue

        if param == scan1["scan_motor"] == scan2["scan_motor"]:
            # check if ranges of variable parameter overlap
            range1 = scan1[param]
            range2 = scan2[param]
            # maximum gap between ranges of the scanning parameter (default 0)
            max_range_gap = MAX_RANGE_GAP.get(param, 0)
            if max(range1[0] - range2[-1], range2[0] - range1[-1]) > max_range_gap:
                return False

        elif np.max(np.abs(scan1[param] - scan2[param])) > PARAM_PRECISIONS[param]:
            return False

    return True


def merge_datasets(dataset_into, dataset_from):
    scan_motors_into = dataset_into[0]["scan_motors"]
    scan_motors_from = dataset_from[0]["scan_motors"]
    if scan_motors_into != scan_motors_from:
        print(f"Scan motors mismatch between datasets: {scan_motors_into} vs {scan_motors_from}")
        return

    merged = np.zeros(len(dataset_from), dtype=np.bool)
    for scan_into in dataset_into:
        for ind, scan_from in enumerate(dataset_from):
            if _parameters_match(scan_into, scan_from) and not merged[ind]:
                merge_scans(scan_into, scan_from)
                merged[ind] = True

    for scan_from in dataset_from:
        dataset_into.append(scan_from)


def merge_scans(scan_into, scan_from):
    if "init_scan" not in scan_into:
        scan_into["init_scan"] = scan_into.copy()

    if "merged_scans" not in scan_into:
        scan_into["merged_scans"] = []

    if scan_from in scan_into["merged_scans"]:
        return

    scan_into["merged_scans"].append(scan_from)

    scan_motor = scan_into["scan_motor"]  # the same as scan_from["scan_motor"]

    pos_all = np.array([])
    val_all = np.array([])
    err_all = np.array([])
    for scan in [scan_into["init_scan"], *scan_into["merged_scans"]]:
        pos_all = np.append(pos_all, scan[scan_motor])
        val_all = np.append(val_all, scan["counts"])
        err_all = np.append(err_all, scan["counts_err"] ** 2)

    sort_index = np.argsort(pos_all)
    pos_all = pos_all[sort_index]
    val_all = val_all[sort_index]
    err_all = err_all[sort_index]

    pos_tmp = pos_all[:1]
    val_tmp = val_all[:1]
    err_tmp = err_all[:1]
    num_tmp = np.array([1])
    for pos, val, err in zip(pos_all[1:], val_all[1:], err_all[1:]):
        if pos - pos_tmp[-1] < 0.0005:
            # the repeated motor position
            val_tmp[-1] += val
            err_tmp[-1] += err
            num_tmp[-1] += 1
        else:
            # a new motor position
            pos_tmp = np.append(pos_tmp, pos)
            val_tmp = np.append(val_tmp, val)
            err_tmp = np.append(err_tmp, err)
            num_tmp = np.append(num_tmp, 1)

    scan_into[scan_motor] = pos_tmp
    scan_into["counts"] = val_tmp / num_tmp
    scan_into["counts_err"] = np.sqrt(err_tmp) / num_tmp

    scan_from["export"] = False

    fname1 = os.path.basename(scan_into["original_filename"])
    fname2 = os.path.basename(scan_from["original_filename"])
    print(f'Merging scans: {scan_into["idx"]} ({fname1}) <-- {scan_from["idx"]} ({fname2})')


def restore_scan(scan):
    if "merged_scans" in scan:
        for merged_scan in scan["merged_scans"]:
            merged_scan["export"] = True

    if "init_scan" in scan:
        tmp = scan["init_scan"]
        scan.clear()
        scan.update(tmp)
        # force scan export to True, otherwise in the sequence of incorrectly merged scans
        # a <- b <- c the scan b will be restored with scan["export"] = False if restoring executed
        # in the same order, i.e. restore a -> restore b
        scan["export"] = True


def fit_scan(scan, model_dict, fit_from=None, fit_to=None):
    if fit_from is None:
        fit_from = -np.inf
    if fit_to is None:
        fit_to = np.inf

    y_fit = scan["counts"]
    y_err = scan["counts_err"]
    x_fit = scan[scan["scan_motor"]]

    # apply fitting range
    fit_ind = (fit_from <= x_fit) & (x_fit <= fit_to)
    if not np.any(fit_ind):
        print(f"No data in fit range for scan {scan['idx']}")
        return

    y_fit = y_fit[fit_ind]
    y_err = y_err[fit_ind]
    x_fit = x_fit[fit_ind]

    model = None
    for model_index, (model_name, model_param) in enumerate(model_dict.items()):
        model_name, _ = model_name.split("-")
        prefix = f"f{model_index}_"

        if model_name == "linear":
            _model = LinearModel(prefix=prefix)
        elif model_name == "gaussian":
            _model = GaussianModel(prefix=prefix)
        elif model_name == "voigt":
            _model = VoigtModel(prefix=prefix)
        elif model_name == "pvoigt":
            _model = PseudoVoigtModel(prefix=prefix)
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")

        _init_guess = _model.guess(y_fit, x=x_fit)

        for param_index, param_name in enumerate(model_param["param"]):
            param_hints = {}
            for hint_name in ("value", "vary", "min", "max"):
                tmp = model_param[hint_name][param_index]
                if tmp is None:
                    param_hints[hint_name] = getattr(_init_guess[prefix + param_name], hint_name)
                else:
                    param_hints[hint_name] = tmp

            if "center" in param_name:
                if np.isneginf(param_hints["min"]):
                    param_hints["min"] = np.min(x_fit)

                if np.isposinf(param_hints["max"]):
                    param_hints["max"] = np.max(x_fit)

            if "sigma" in param_name:
                if np.isposinf(param_hints["max"]):
                    param_hints["max"] = np.max(x_fit) - np.min(x_fit)

            _model.set_param_hint(param_name, **param_hints)

        if model is None:
            model = _model
        else:
            model += _model

    scan["fit"] = model.fit(y_fit, x=x_fit, weights=1 / y_err)


def get_area(scan, area_method, lorentz):
    if "fit" not in scan:
        return

    if area_method not in AREA_METHODS:
        raise ValueError(f"Unknown area method: {area_method}.")

    if area_method == "fit_area":
        area_v = 0
        area_s = 0
        for name, param in scan["fit"].params.items():
            if "amplitude" in name:
                area_v += np.nan if param.value is None else param.value
                area_s += np.nan if param.stderr is None else param.stderr

    else:  # area_method == "int_area"
        y_val = scan["counts"]
        x_val = scan[scan["scan_motor"]]
        y_bkg = scan["fit"].eval_components(x=x_val)["f0_"]
        area_v = simpson(y_val, x=x_val) - trapezoid(y_bkg, x=x_val)
        area_s = np.sqrt(area_v)

    if lorentz:
        # lorentz correction to area
        if scan["zebra_mode"] == "bi":
            twotheta = np.deg2rad(scan["twotheta"])
            corr_factor = np.sin(twotheta)
        else:  # zebra_mode == "nb":
            gamma = np.deg2rad(scan["gamma"])
            nu = np.deg2rad(scan["nu"])
            corr_factor = np.sin(gamma) * np.cos(nu)

        area_v = np.abs(area_v * corr_factor)
        area_s = np.abs(area_s * corr_factor)

    scan["area"] = (area_v, area_s)


def fit_event(scan, fr_from, fr_to, y_from, y_to, x_from, x_to):
    data_roi = scan["data"][fr_from:fr_to, y_from:y_to, x_from:x_to]

    model = GaussianModel()
    fr = np.arange(fr_from, fr_to)
    counts_per_fr = np.sum(data_roi, axis=(1, 2))
    params = model.guess(counts_per_fr, fr)
    result = model.fit(counts_per_fr, x=fr, params=params)
    frC = result.params["center"].value
    intensity = result.params["height"].value

    counts_std = counts_per_fr.std()
    counts_mean = counts_per_fr.mean()
    snr = 0 if counts_std == 0 else counts_mean / counts_std

    model = Gaussian2dModel()
    xs, ys = np.meshgrid(np.arange(x_from, x_to), np.arange(y_from, y_to))
    xs = xs.flatten()
    ys = ys.flatten()
    counts = np.sum(data_roi, axis=0).flatten()
    params = model.guess(counts, xs, ys)
    result = model.fit(counts, x=xs, y=ys, params=params)
    xC = result.params["centerx"].value
    yC = result.params["centery"].value

    scan["fit"] = {"frame": frC, "x_pos": xC, "y_pos": yC, "intensity": intensity, "snr": snr}
