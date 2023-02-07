import os

import numpy as np

SINQ_PATH = "/afs/psi.ch/project/sinqdata"
ZEBRA_PROPOSALS_PATH = os.path.join(SINQ_PATH, "{year}/zebra/{proposal}")


def find_proposal_path(proposal):
    for entry in os.scandir(SINQ_PATH):
        if entry.is_dir() and len(entry.name) == 4 and entry.name.isdigit():
            proposal_path = ZEBRA_PROPOSALS_PATH.format(year=entry.name, proposal=proposal)
            if os.path.isdir(proposal_path):
                # found it
                break
    else:
        raise ValueError(f"Can not find data for proposal '{proposal}'")

    return proposal_path


def parse_hkl(fileobj, data_type):
    next(fileobj)
    fields = map(str.lower, next(fileobj).strip("!").strip().split())
    next(fileobj)
    data = np.loadtxt(fileobj, unpack=True)
    res = dict(zip(fields, data))

    # adapt to .ccl/.dat files naming convention
    res["counts"] = res.pop("f2")

    if data_type == ".hkl":
        for ind in ("h", "k", "l"):
            res[ind] = res[ind].astype(int)

    return res
