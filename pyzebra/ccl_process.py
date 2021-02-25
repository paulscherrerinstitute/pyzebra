import itertools

import numpy as np
from lmfit.models import GaussianModel, LinearModel, PseudoVoigtModel, VoigtModel

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


def normalize_dataset(dataset, monitor=100_000):
    for scan in dataset:
        monitor_ratio = monitor / scan["monitor"]
        scan["Counts"] *= monitor_ratio
        scan["monitor"] = monitor


def merge_duplicates(dataset):
    for scan_i, scan_j in itertools.combinations(dataset, 2):
        if _parameters_match(scan_i, scan_j):
            merge_scans(scan_i, scan_j)


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


def merge_datasets(dataset1, dataset2):
    for scan_j in dataset2:
        for scan_i in dataset1:
            if _parameters_match(scan_i, scan_j):
                merge_scans(scan_i, scan_j)
                break

        dataset1.append(scan_j)


def merge_scans(scan1, scan2):
    omega = np.concatenate((scan1["omega"], scan2["omega"]))
    counts = np.concatenate((scan1["Counts"], scan2["Counts"]))

    index = np.argsort(omega)

    scan1["omega"] = omega[index]
    scan1["Counts"] = counts[index]

    scan2["active"] = False
    print(f'Merging scans: {scan1["idx"]} <-- {scan2["idx"]}')


def _create_fit_model(model_dict):
    model = None
    for model_index, (model_name, model_param) in enumerate(model_dict.items()):
        model_name, _ = model_name.split("-")
        prefix = f"f{model_index}_"

        if model_name == "background":
            _model = LinearModel(prefix=prefix, name="background")
        elif model_name == "gauss":
            _model = GaussianModel(prefix=prefix, name="gauss")
        elif model_name == "voigt":
            _model = VoigtModel(prefix=prefix)
        elif model_name == "pseudovoigt":
            _model = PseudoVoigtModel(prefix=prefix)
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")

        for param_index, param_name in enumerate(model_param["param"]):
            param_hints = {}
            for hint_name in ("value", "vary", "min", "max"):
                tmp = model_param[hint_name][param_index]
                if tmp is not None:
                    param_hints[hint_name] = tmp
            _model.set_param_hint(param_name, **param_hints)

        if model is None:
            model = _model
        else:
            model += _model

    return model


def fit_scan(scan, model_dict):
    model = _create_fit_model(model_dict)
    scan["fit_result"] = model.fit(scan["Counts"], x=scan[scan["scan_motor"]])
