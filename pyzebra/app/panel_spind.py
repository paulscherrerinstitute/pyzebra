import ast
import math
import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    DataTable,
    Panel,
    Spinner,
    TableColumn,
    TextAreaInput,
    TextInput,
)
from scipy.optimize import curve_fit

import pyzebra


def create():
    path_prefix_textinput = TextInput(title="Path prefix:", value="")
    selection_list = TextAreaInput(title="ROIs:", rows=7)
    lattice_const_textinput = TextInput(
        title="Lattice constants:", value="8.3211,8.3211,8.3211,90.00,90.00,90.00"
    )
    max_res_spinner = Spinner(title="max-res", value=2, step=0.01)
    seed_pool_size_spinner = Spinner(title="seed-pool-size", value=5, step=0.01)
    seed_len_tol_spinner = Spinner(title="seed-len-tol", value=0.02, step=0.01)
    seed_angle_tol_spinner = Spinner(title="seed-angle-tol", value=1, step=0.01)
    eval_hkl_tol_spinner = Spinner(title="eval-hkl-tol", value=0.15, step=0.01)

    def process_button_callback():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_peak_list_dir = os.path.join(temp_dir, "peak_list")
            os.mkdir(temp_peak_list_dir)
            temp_event_file = os.path.join(temp_peak_list_dir, "event-0.txt")
            temp_hkl_file = os.path.join(temp_dir, "hkl.h5")
            roi_dict = ast.literal_eval(selection_list.value)

            subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "2",
                    "python",
                    "spind/gen_hkl_table.py",
                    lattice_const_textinput.value,
                    "--max-res",
                    str(max_res_spinner.value),
                    "-o",
                    temp_hkl_file,
                ],
                check=True,
            )

            prepare_event_file(temp_event_file, roi_dict, path_prefix_textinput.value)

            subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "2",
                    "python",
                    "spind/SPIND.py",
                    temp_peak_list_dir,
                    temp_hkl_file,
                    "-o",
                    temp_dir,
                    "--seed-pool-size",
                    str(seed_pool_size_spinner.value),
                    "--seed-len-tol",
                    str(seed_len_tol_spinner.value),
                    "--seed-angle-tol",
                    str(seed_angle_tol_spinner.value),
                    "--eval-hkl-tol",
                    str(eval_hkl_tol_spinner.value),
                ],
                check=True,
            )

            try:
                with open(os.path.join(temp_dir, "spind.txt")) as f_out:
                    spind_res = defaultdict(list)
                    for line in f_out:
                        c1, c2, c3, c4, c5, *c_rest = line.split()
                        spind_res["label"].append(c1)
                        spind_res["crystal_id"].append(c2)
                        spind_res["match_rate"].append(c3)
                        spind_res["matched_peaks"].append(c4)
                        spind_res["column_5"].append(c5)

                        # last digits are spind UB matrix
                        vals = list(map(float, c_rest))
                        ub_matrix_spind = np.array(vals).reshape(3, 3)
                        ub_matrix = np.linalg.inv(np.transpose(ub_matrix_spind)) * 1e10
                        spind_res["ub_matrix"].append(str(ub_matrix))

                    results_table_source.data.update(spind_res)

            except FileNotFoundError:
                print("No results from spind")

    process_button = Button(label="Process", button_type="primary")
    process_button.on_click(process_button_callback)

    results_table_source = ColumnDataSource(dict())
    results_table = DataTable(
        source=results_table_source,
        columns=[
            TableColumn(field="label", title="Label", width=50),
            TableColumn(field="crystal_id", title="Crystal ID", width=100),
            TableColumn(field="match_rate", title="Match Rate", width=100),
            TableColumn(field="matched_peaks", title="Matched Peaks", width=100),
            TableColumn(field="column_5", title="", width=100),
            TableColumn(field="ub_matrix", title="UB Matrix", width=250),
        ],
        height=300,
        width=700,
        fit_columns=False,
        index_position=None,
    )

    tab_layout = row(
        column(
            path_prefix_textinput,
            selection_list,
            lattice_const_textinput,
            max_res_spinner,
            seed_pool_size_spinner,
            seed_len_tol_spinner,
            seed_angle_tol_spinner,
            eval_hkl_tol_spinner,
            process_button,
        ),
        results_table,
    )

    return Panel(child=tab_layout, title="spind")


def gauss(x, *p):
    """Defines Gaussian function
    Args:
        A - amplitude, mu - position of the center, sigma - width
    Returns:
        Gaussian function
    """
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def prepare_event_file(export_filename, roi_dict, path_prefix=""):
    p0 = [1.0, 0.0, 1.0]
    maxfev = 100000
    with open(export_filename, "w") as f:
        for file, rois in roi_dict.items():
            dat = pyzebra.read_detector_data(path_prefix + file + ".hdf")

            wave = dat["wave"]
            ddist = dat["ddist"]

            pol_angle = dat["pol_angle"][0]
            rot_angle = dat["rot_angle"][0]
            tlt_angle = dat["tlt_angle"][0]
            chi_angle = dat["chi_angle"][0]
            phi_angle = dat["phi_angle"][0]

            var_angle = dat["variable"]
            var_angle_name = dat["variable_name"]

            for roi in rois:
                x0, xN, y0, yN, fr0, frN = roi
                data_roi = dat["data"][fr0:frN, y0:yN, x0:xN]

                cnts = np.sum(data_roi, axis=(1, 2))
                coeff, _ = curve_fit(gauss, range(len(cnts)), cnts, p0=p0, maxfev=maxfev)

                m = cnts.mean()
                sd = cnts.std()
                snr_cnts = np.where(sd == 0, 0, m / sd)

                frC = fr0 + coeff[1]
                var_F = var_angle[math.floor(frC)]
                var_C = var_angle[math.ceil(frC)]
                frStep = frC - math.floor(frC)
                var_step = var_C - var_F
                var_p = var_F + var_step * frStep

                if var_angle_name == "pol_angle":
                    pol_angle = var_p
                elif var_angle_name == "rot_angle":
                    rot_angle = var_p
                elif var_angle_name == "tlt_angle":
                    tlt_angle = var_p
                elif var_angle_name == "chi_angle":
                    chi_angle = var_p
                elif var_angle_name == "phi_angle":
                    phi_angle = var_p

                intensity = coeff[1] * abs(coeff[2] * var_step) * math.sqrt(2) * math.sqrt(np.pi)

                projX = np.sum(data_roi, axis=(0, 1))
                coeff, _ = curve_fit(gauss, range(len(projX)), projX, p0=p0, maxfev=maxfev)
                x_pos = x0 + coeff[1]

                projY = np.sum(data_roi, axis=(0, 2))
                coeff, _ = curve_fit(gauss, range(len(projY)), projY, p0=p0, maxfev=maxfev)
                y_pos = y0 + coeff[1]

                ga, nu = pyzebra.det2pol(ddist, pol_angle, tlt_angle, x_pos, y_pos)
                diff_vector = pyzebra.z1frmd(wave, ga, rot_angle, chi_angle, phi_angle, nu)
                d_spacing = float(pyzebra.dandth(wave, diff_vector)[0])
                dv1, dv2, dv3 = diff_vector.flatten() * 1e10

                f.write(f"{x_pos} {y_pos} {intensity} {snr_cnts} {dv1} {dv2} {dv3} {d_spacing}\n")
