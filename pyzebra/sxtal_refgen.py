import io

SXTAL_REFGEN_PATH = "/afs/psi.ch/project/sinq/rhel7/bin/Sxtal_Refgen"

_zebraBI_default_geom = """GEOM          2 Bissecting - HiCHI
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
        Gamma     2.0    60.5     0.00
        Omega     0.0    120.0     0.00
        Chi      80.0    210.0     0.00
        Phi       0.0    360.0     0.00

DET_OFF      0       0      0"""

_zebraNB_default_geom = """!  File with an example of instrument SXTAL file
!  Case of D3
!
INFO          Lifting arm diffractometer (hot neutrons)
NAME          D3
GEOM          3 Normal Beam
BLFR          z-up
DIST_UNITS    mm
ANGL_UNITS    deg
DET_TYPE      Point  ipsd 1
DIST_DET      660
DIM_XY         1.0   1.0    1    1
GAPS_DET       0  0

SETTING       1 0 0   0 1 0   0 0 1
NUM_ANG  4
ANG_LIMITS         Min      Max    Offset
        Gamma     -1.5    128.     0.00
        Omega   -180.0   180.0     0.00
        Nu        -15      15     0.00
        Phi        0.0     45.0     0.00

DET_OFF      0       0      0"""


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
        ang_lims[ang.lower()] = [float(ang_min), float(ang_max), float(ang_offset)]

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
