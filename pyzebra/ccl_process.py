import itertools
import os

import numpy as np
from lmfit.models import GaussianModel, LinearModel, PseudoVoigtModel, VoigtModel
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
    merged = np.zeros(len(dataset_from), dtype=np.bool)
    for scan_into in dataset_into:
        for ind, scan_from in enumerate(dataset_from):
            if _parameters_match(scan_into, scan_from) and not merged[ind]:
                merge_scans(scan_into, scan_from)
                merged[ind] = True

    for scan_from in dataset_from:
        dataset_into.append(scan_from)


def merge_scans(scan_into, scan_from):
    # TODO: does it need to be "scan_motor" instead of omega for a generalized solution?
    if "init_omega" not in scan_into:
        scan_into["init_omega"] = scan_into["omega"]
        scan_into["init_counts"] = scan_into["counts"]

    omega = np.concatenate((scan_into["omega"], scan_from["omega"]))
    counts = np.concatenate((scan_into["counts"], scan_from["counts"]))

    index = np.argsort(omega)

    scan_into["omega"] = omega[index]
    scan_into["counts"] = counts[index]

    scan_from["active"] = False

    fname1 = os.path.basename(scan_into["original_filename"])
    fname2 = os.path.basename(scan_from["original_filename"])
    print(f'Merging scans: {scan_into["idx"]} ({fname1}) <-- {scan_from["idx"]} ({fname2})')


def restore_scan(scan):
    if "init_omega" in scan:
        scan["omega"] = scan["init_omega"]
        scan["counts"] = scan["init_counts"]
        del scan["init_omega"]
        del scan["init_counts"]


def fit_scan(scan, model_dict, fit_from=None, fit_to=None):
    if fit_from is None:
        fit_from = -np.inf
    if fit_to is None:
        fit_to = np.inf

    y_fit = scan["counts"]
    x_fit = scan[scan["scan_motor"]]

    # apply fitting range
    fit_ind = (fit_from <= x_fit) & (x_fit <= fit_to)
    y_fit = y_fit[fit_ind]
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

    weights = [1 / np.sqrt(val) if val != 0 else 1 for val in y_fit]
    scan["fit"] = model.fit(y_fit, x=x_fit, weights=weights)


def get_area(scan, area_method, lorentz):
    if area_method not in AREA_METHODS:
        raise ValueError(f"Unknown area method: {area_method}.")

    if area_method == "fit_area":
        for name, param in scan["fit"].params.items():
            if "amplitude" in name:
                if param.stderr is None:
                    area_n = np.nan
                    area_s = np.nan
                else:
                    area_n = param.value
                    area_s = param.stderr
                # TODO: take into account multiple peaks
                break
        else:
            area_n = np.nan
            area_s = np.nan

    else:  # area_method == "int_area"
        y_val = scan["counts"]
        x_val = scan[scan["scan_motor"]]
        y_bkg = scan["fit"].eval_components(x=x_val)["f0_"]
        area_n = simpson(y_val, x=x_val) - trapezoid(y_bkg, x=x_val)
        area_s = np.sqrt(area_n)

    if lorentz:
        # lorentz correction to area
        if scan["zebra_mode"] == "bi":
            twotheta = np.deg2rad(scan["twotheta"])
            corr_factor = np.sin(twotheta)
        else:  # zebra_mode == "nb":
            gamma = np.deg2rad(scan["gamma"])
            nu = np.deg2rad(scan["nu"])
            corr_factor = np.sin(gamma) * np.cos(nu)

        area_n = np.abs(area_n * corr_factor)
        area_s = np.abs(area_s * corr_factor)

    scan["area"] = (area_n, area_s)
