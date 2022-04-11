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

_zebra_default_cfl = """TITLE mymaterial
SPGR  P 63 2 2
CELL  5.73 5.73 11.89 90 90 120

WAVE 1.383

UBMAT
0.000000    0.000000    0.084104
0.000000    0.174520   -0.000000
0.201518    0.100759    0.000000

INSTR  zebra.geom

ORDER   1 2 3

ANGOR   gamma

HLIM    -25 25 -25 25 -25 25
SRANG   0.0  0.7

Mag_Structure
lattiCE P 1
kvect        0.0 0.0 0.0
magcent
symm  x,y,z
msym  u,v,w, 0.0
End_Mag_Structure
"""


def get_zebraBI_default_geom_file():
    return io.StringIO(_zebraBI_default_geom)


def get_zebraNB_default_geom_file():
    return io.StringIO(_zebraNB_default_geom)


def get_zebra_default_cfl_file():
    return io.StringIO(_zebra_default_cfl)


def read_geom_file(fileobj):
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


def export_geom_file(path, ang_lims):
    if "chi" in ang_lims:  # BI geometry
        default_file = get_zebraBI_default_geom_file()
        n_ang = 4
    else:  # NB geometry
        default_file = get_zebraNB_default_geom_file()
        n_ang = 3

    with open(path, "w") as out_file:
        for line in default_file:
            out_file.write(line)

            if line.startswith("ANG_LIMITS"):
                for _ in range(n_ang):
                    next_line = next(default_file)
                    ang, _, _, _ = next_line.split()
                    vals = ang_lims[ang.lower()]
                    out_file.write(f"{'':<8}{ang:<10}{vals[0]:<10}{vals[1]:<10}{vals[2]:<10}\n")


def calc_ub_matrix(params):
    with tempfile.TemporaryDirectory() as temp_dir:
        cfl_file = os.path.join(temp_dir, "ub_matrix.cfl")

        with open(cfl_file, "w") as fileobj:
            for key, value in params.items():
                fileobj.write(f"{key} {value}\n")

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
        with open(sfa_file, "r") as fileobj:
            for line in fileobj:
                if "BL_M" in line:  # next 3 lines contain the matrix
                    for _ in range(3):
                        next_line = next(fileobj)
                        *vals, _ = next_line.split(maxsplit=3)
                        ub_matrix.extend(vals)

    return ub_matrix


def read_cfl_file(fileobj):
    params = {"SPGR": None, "CELL": None, "WAVE": None, "UBMAT": None, "HLIM": None, "SRANG": None}
    param_names = tuple(params)
    for line in fileobj:
        if line.startswith(param_names):
            if line.startswith("UBMAT"):  # next 3 lines contain the matrix
                param, val = "UBMAT", []
                for _ in range(3):
                    next_line = next(fileobj)
                    val.extend(next_line.split(maxsplit=2))
            else:
                param, val = line.split(maxsplit=1)

            params[param] = val

    return params
