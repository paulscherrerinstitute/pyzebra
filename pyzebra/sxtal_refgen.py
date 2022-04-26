import io
import os
import subprocess
import tempfile
from math import ceil, floor

import numpy as np

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
        Gamma     0.0    128.0     0.00
        Omega     0.0     64.0     0.00
        Chi      80.0    211.0     0.00
        Phi       0.0    360.0     0.00

DET_OFF      0       0      0
"""

_zebraNB_default_geom = """GEOM          3 Normal Beam
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
    for line in fileobj:
        if "!" in line:  # remove comments that start with ! sign
            line, _ = line.split(sep="!", maxsplit=1)

        if line.startswith("GEOM"):
            _, val = line.split(maxsplit=1)
            if val.startswith("2"):
                ang_lims["geom"] = "bi"
            else:  # val.startswith("3")
                ang_lims["geom"] = "nb"

        elif line.startswith("ANG_LIMITS"):
            # read angular limits
            for line in fileobj:
                if not line or line.isspace():
                    break

                ang, ang_min, ang_max, ang_offset = line.split()
                ang_lims[ang.lower()] = [ang_min, ang_max, ang_offset]

            if "2theta" in ang_lims:  # treat 2theta as gamma
                ang_lims["gamma"] = ang_lims.pop("2theta")

    return ang_lims


def export_geom_file(path, ang_lims, template=None):
    if ang_lims["geom"] == "bi":
        template_file = get_zebraBI_default_geom_file()
        n_ang = 4
    else:  # ang_lims["geom"] == "nb"
        template_file = get_zebraNB_default_geom_file()
        n_ang = 3

    if template is not None:
        template_file = template

    with open(path, "w") as out_file:
        for line in template_file:
            out_file.write(line)

            if line.startswith("ANG_LIMITS"):
                for _ in range(n_ang):
                    next_line = next(template_file)
                    ang, _, _, _ = next_line.split()

                    if ang == "2theta":  # treat 2theta as gamma
                        ang = "Gamma"
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
    params = {
        "SPGR": None,
        "CELL": None,
        "WAVE": None,
        "UBMAT": None,
        "HLIM": None,
        "SRANG": None,
        "lattiCE": None,
        "kvect": None,
    }
    param_names = tuple(params)

    for line in fileobj:
        line = line.strip()
        if "!" in line:  # remove comments that start with ! sign
            line, _ = line.split(sep="!", maxsplit=1)

        if line.startswith(param_names):
            if line.startswith("UBMAT"):  # next 3 lines contain the matrix
                param, val = "UBMAT", []
                for _ in range(3):
                    next_line = next(fileobj).strip()
                    val.extend(next_line.split(maxsplit=2))
            else:
                param, val = line.split(maxsplit=1)

            params[param] = val

    return params


def read_cif_file(fileobj):
    params = {"SPGR": None, "CELL": None, "ATOM": []}

    cell_params = {
        "_cell_length_a": None,
        "_cell_length_b": None,
        "_cell_length_c": None,
        "_cell_angle_alpha": None,
        "_cell_angle_beta": None,
        "_cell_angle_gamma": None,
    }
    cell_param_names = tuple(cell_params)

    atom_param_pos = {
        "_atom_site_label": 0,
        "_atom_site_type_symbol": None,
        "_atom_site_fract_x": None,
        "_atom_site_fract_y": None,
        "_atom_site_fract_z": None,
        "_atom_site_U_iso_or_equiv": None,
        "_atom_site_occupancy": None,
    }
    atom_param_names = tuple(atom_param_pos)

    for line in fileobj:
        line = line.strip()
        if line.startswith("_space_group_name_H-M_alt"):
            _, val = line.split(maxsplit=1)
            params["SPGR"] = val.strip("'")

        elif line.startswith(cell_param_names):
            param, val = line.split(maxsplit=1)
            cell_params[param] = val

        elif line.startswith("_atom_site_label"):  # assume this is the start of atom data
            for ind, line in enumerate(fileobj, start=1):
                line = line.strip()

                # read fields
                if line.startswith("_atom_site"):
                    if line.startswith(atom_param_names):
                        atom_param_pos[line] = ind
                    continue

                # read data till an empty line
                if not line:
                    break
                vals = line.split()
                params["ATOM"].append(" ".join([vals[ind] for ind in atom_param_pos.values()]))

    if None not in cell_params.values():
        params["CELL"] = " ".join(cell_params.values())

    return params


def export_cfl_file(path, params, template=None):
    param_names = tuple(params)
    if template is None:
        template_file = get_zebra_default_cfl_file()
    else:
        template_file = template

    atom_done = False
    with open(path, "w") as out_file:
        for line in template_file:
            if line.startswith(param_names):
                if line.startswith("UBMAT"):  # only UBMAT values are not on the same line
                    out_file.write(line)
                    for i in range(3):
                        next(template_file)
                        out_file.write(" ".join(params["UBMAT"][3 * i : 3 * (i + 1)]) + "\n")

                elif line.startswith("ATOM"):
                    if "ATOM" in params:
                        # replace all ATOM with values in params
                        while line.startswith("ATOM"):
                            line = next(template_file)
                        for atom_line in params["ATOM"]:
                            out_file.write(f"ATOM {atom_line}\n")
                        atom_done = True

                else:
                    param, _ = line.split(maxsplit=1)
                    out_file.write(f"{param} {params[param]}\n")

            elif line.startswith("INSTR"):
                # replace it with a default name
                out_file.write("INSTR  zebra.geom\n")

            else:
                out_file.write(line)

        # append ATOM data if it's present and a template did not contain it
        if "ATOM" in params and not atom_done:
            out_file.write("\n")
            for atom_line in params["ATOM"]:
                out_file.write(f"ATOM {atom_line}\n")


def sort_hkl_file_bi(file_in, file_out, priority, chunks):
    with open(file_in) as fileobj:
        file_in_data = fileobj.readlines()

    data = np.genfromtxt(file_in, skip_header=3)
    stt = data[:, 4]
    omega = data[:, 5]
    chi = data[:, 6]
    phi = data[:, 7]

    lines = file_in_data[3:]
    lines_update = []

    angles = {"2theta": stt, "omega": omega, "chi": chi, "phi": phi}

    # Reverse flag
    to_reverse = False
    to_reverse_p2 = False
    to_reverse_p3 = False

    # Get indices within first priority
    ang_p1 = angles[priority[0]]
    begin_p1 = floor(min(ang_p1))
    end_p1 = ceil(max(ang_p1))
    delta_p1 = chunks[0]
    for p1 in range(begin_p1, end_p1, delta_p1):
        ind_p1 = [j for j, x in enumerate(ang_p1) if p1 <= x and x < p1 + delta_p1]

        stt_new = [stt[x] for x in ind_p1]
        omega_new = [omega[x] for x in ind_p1]
        chi_new = [chi[x] for x in ind_p1]
        phi_new = [phi[x] for x in ind_p1]
        lines_new = [lines[x] for x in ind_p1]

        angles_p2 = {"stt": stt_new, "omega": omega_new, "chi": chi_new, "phi": phi_new}

        # Get indices for second priority
        ang_p2 = angles_p2[priority[1]]
        if len(ang_p2) > 0 and to_reverse_p2:
            begin_p2 = ceil(max(ang_p2))
            end_p2 = floor(min(ang_p2))
            delta_p2 = -chunks[1]
        elif len(ang_p2) > 0 and not to_reverse_p2:
            end_p2 = ceil(max(ang_p2))
            begin_p2 = floor(min(ang_p2))
            delta_p2 = chunks[1]
        else:
            end_p2 = 0
            begin_p2 = 0
            delta_p2 = 1

        to_reverse_p2 = not to_reverse_p2

        for p2 in range(begin_p2, end_p2, delta_p2):
            min_p2 = min([p2, p2 + delta_p2])
            max_p2 = max([p2, p2 + delta_p2])
            ind_p2 = [j for j, x in enumerate(ang_p2) if min_p2 <= x and x < max_p2]

            stt_new2 = [stt_new[x] for x in ind_p2]
            omega_new2 = [omega_new[x] for x in ind_p2]
            chi_new2 = [chi_new[x] for x in ind_p2]
            phi_new2 = [phi_new[x] for x in ind_p2]
            lines_new2 = [lines_new[x] for x in ind_p2]

            angles_p3 = {"stt": stt_new2, "omega": omega_new2, "chi": chi_new2, "phi": phi_new2}

            # Get indices for third priority
            ang_p3 = angles_p3[priority[2]]
            if len(ang_p3) > 0 and to_reverse_p3:
                begin_p3 = ceil(max(ang_p3)) + chunks[2]
                end_p3 = floor(min(ang_p3)) - chunks[2]
                delta_p3 = -chunks[2]
            elif len(ang_p3) > 0 and not to_reverse_p3:
                end_p3 = ceil(max(ang_p3)) + chunks[2]
                begin_p3 = floor(min(ang_p3)) - chunks[2]
                delta_p3 = chunks[2]
            else:
                end_p3 = 0
                begin_p3 = 0
                delta_p3 = 1

            to_reverse_p3 = not to_reverse_p3

            for p3 in range(begin_p3, end_p3, delta_p3):
                min_p3 = min([p3, p3 + delta_p3])
                max_p3 = max([p3, p3 + delta_p3])
                ind_p3 = [j for j, x in enumerate(ang_p3) if min_p3 <= x and x < max_p3]

                angle_new3 = [angles_p3[priority[3]][x] for x in ind_p3]

                ind_final = [x for _, x in sorted(zip(angle_new3, ind_p3), reverse=to_reverse)]

                to_reverse = not to_reverse

                for i in ind_final:
                    lines_update.append(lines_new2[i])

    with open(file_out, "w") as fileobj:
        for _ in range(3):
            fileobj.write(file_in_data.pop(0))

        fileobj.writelines(lines_update)


def sort_hkl_file_nb(file_in, file_out, priority, chunks):
    with open(file_in) as fileobj:
        file_in_data = fileobj.readlines()

    data = np.genfromtxt(file_in, skip_header=3)
    gamma = data[:, 4]
    omega = data[:, 5]
    nu = data[:, 6]

    lines = file_in_data[3:]
    lines_update = []

    angles = {"gamma": gamma, "omega": omega, "nu": nu}

    to_reverse = False
    to_reverse_p2 = False

    # Get indices within first priority
    ang_p1 = angles[priority[0]]
    begin_p1 = floor(min(ang_p1))
    end_p1 = ceil(max(ang_p1))
    delta_p1 = chunks[0]
    for p1 in range(begin_p1, end_p1, delta_p1):
        ind_p1 = [j for j, x in enumerate(ang_p1) if p1 <= x and x < p1 + delta_p1]

        # Get angles from within nu range
        lines_new = [lines[x] for x in ind_p1]
        gamma_new = [gamma[x] for x in ind_p1]
        omega_new = [omega[x] for x in ind_p1]
        nu_new = [nu[x] for x in ind_p1]

        angles_p2 = {"gamma": gamma_new, "omega": omega_new, "nu": nu_new}

        # Get indices for second priority
        ang_p2 = angles_p2[priority[1]]
        if len(gamma_new) > 0 and to_reverse_p2:
            begin_p2 = ceil(max(ang_p2))
            end_p2 = floor(min(ang_p2))
            delta_p2 = -chunks[1]
        elif len(gamma_new) > 0 and not to_reverse_p2:
            end_p2 = ceil(max(ang_p2))
            begin_p2 = floor(min(ang_p2))
            delta_p2 = chunks[1]
        else:
            end_p2 = 0
            begin_p2 = 0
            delta_p2 = 1

        to_reverse_p2 = not to_reverse_p2

        for p2 in range(begin_p2, end_p2, delta_p2):
            min_p2 = min([p2, p2 + delta_p2])
            max_p2 = max([p2, p2 + delta_p2])
            ind_p2 = [j for j, x in enumerate(ang_p2) if min_p2 <= x and x < max_p2]

            angle_new2 = [angles_p2[priority[2]][x] for x in ind_p2]

            ind_final = [x for _, x in sorted(zip(angle_new2, ind_p2), reverse=to_reverse)]

            to_reverse = not to_reverse

            for i in ind_final:
                lines_update.append(lines_new[i])

    with open(file_out, "w") as fileobj:
        for _ in range(3):
            fileobj.write(file_in_data.pop(0))

        fileobj.writelines(lines_update)
