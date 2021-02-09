import itertools

import numpy as np

from .ccl_io import CCL_ANGLES

PARAM_PRECISIONS = {
    "twotheta": 0.1,
    "chi": 0.1,
    "nu": 0.1,
    "phi": 0.05,
    "omega": 5,
    "gamma": 0.05,
    "temp": 1,
    "mf": 0.001,
    "ub": 0.01,
}


def normalize_dataset(dataset, monitor=100_000):
    for scan in dataset:
        monitor_ratio = monitor / scan["monitor"]
        scan["Counts"] *= monitor_ratio
        scan["monitor"] = monitor


def merge_duplicates(dataset):
    for scan_i, scan_j in itertools.combinations(dataset, 2):
        if _parameters_match(scan_i, scan_j):
            _merge_scans(scan_i, scan_j)


def _parameters_match(scan1, scan2):
    zebra_mode = scan1["zebra_mode"]
    if zebra_mode != scan2["zebra_mode"]:
        return False

    for param in ("ub", "temp", "mf", *(vars[0] for vars in CCL_ANGLES[zebra_mode])):
        if np.max(np.abs(scan1[param] - scan2[param])) > PARAM_PRECISIONS[param]:
            return False

    return True


def merge_datasets(dataset1, dataset2):
    for scan_j in dataset2:
        for scan_i in dataset1:
            if _parameters_match(scan_i, scan_j):
                _merge_scans(scan_i, scan_j)
                break
        else:
            dataset1.append(scan_j)


def _merge_scans(scan1, scan2):
    om = np.concatenate((scan1["om"], scan2["om"]))
    counts = np.concatenate((scan1["Counts"], scan2["Counts"]))

    index = np.argsort(om)

    scan1["om"] = om[index]
    scan1["Counts"] = counts[index]

    print(f'Scan {scan2["idx"]} merged into {scan1["idx"]}')
