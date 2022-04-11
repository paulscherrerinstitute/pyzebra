import io
import os
import tempfile
import subprocess

SXTAL_REFGEN_PATH = "/afs/psi.ch/project/sinq/rhel7/bin/Sxtal_Refgen"

_zebraBI_default_geom = """!
GEOM          2 Bissecting - HiCHI
BLFR          z-up
DIST_UNITS    mm
ANGL_UNITS    deg
DET_TYPE      Point  ipsd 1
DIST_DET      488
DIM_XY         1.0   1.0    1    1
GAPS_DET       0  0

SETTING       1 0 0   0 1 0   0 0 1
NUM_ANG  4
ANG_LIMITS         Min      Max    Offset
        Gamma     0.0    128.0     0.00
        Omega     0.0     64.0     0.00
        Chi      80.0    211.0     0.00
        Phi       0.0    360.0     0.00

DET_OFF      0       0      0
"""

_zebraNB_default_geom = """!
GEOM          3 Normal Beam
BLFR          z-up
DIST_UNITS    mm
ANGL_UNITS    deg
DET_TYPE      Point  ipsd 1
DIST_DET      448
DIM_XY         1.0   1.0    1    1
GAPS_DET       0  0

SETTING       1 0 0   0 1 0   0 0 1
NUM_ANG  3
ANG_LIMITS         Min      Max    Offset
        Gamma      0.0     128.0     0.00
        Omega   -180.0     180.0     0.00
        Nu       -15.0      15.0     0.00

DET_OFF      0       0      0
"""


def get_zebraBI_default_geom_file():
    return io.StringIO(_zebraBI_default_geom)


def get_zebraNB_default_geom_file():
    return io.StringIO(_zebraNB_default_geom)


def read_ang_limits(fileobj):
    ang_lims = dict()
    # locate angular limits in .geom text file
    for line in fileobj:
        if line.startswith("ANG_LIMITS"):
            break

    # read angular limits
    for line in fileobj:
        if not line or line.isspace():
            break

        ang, ang_min, ang_max, ang_offset = line.split()
        ang_lims[ang.lower()] = [ang_min, ang_max, ang_offset]

    return ang_lims


def export_geom(path, ang_lims):
    if "chi" in ang_lims:  # BI geometry
        default_file = get_zebraBI_default_geom_file()
    else:  # NB geometry
        default_file = get_zebraNB_default_geom_file()

    with open(path, "w") as out_file:
        for line in default_file:
            out_file.write(line)

            if line.startswith("ANG_LIMITS"):
                for _ in range(4):
                    next_line = next(default_file)
                    ang, _, _, _ = next_line.split()
                    vals = ang_lims[ang.lower()]
                    out_file.write(f"{'':<8}{ang:<10}{vals[0]:<10}{vals[1]:<10}{vals[2]:<10}\n")


def calc_ub_matrix(params):
    with tempfile.TemporaryDirectory() as temp_dir:
        cfl_file = os.path.join(temp_dir, "ub_matrix.cfl")

        with open(cfl_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} {value}\n")

        comp_proc = subprocess.run(
            [SXTAL_REFGEN_PATH, cfl_file],
            cwd=temp_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(" ".join(comp_proc.args))
        print(comp_proc.stdout)

        sfa_file = os.path.join(temp_dir, "ub_matrix.sfa")
        ub_matrix = []
        with open(sfa_file, "r") as f:
            for line in f:
                if "BL_M" in line:  # next 3 lines contain the matrix
                    for _ in range(3):
                        next_line = next(f)
                        *vals, _ = next_line.split(maxsplit=3)
                        ub_matrix.extend(vals)

    return ub_matrix
