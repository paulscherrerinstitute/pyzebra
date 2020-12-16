import ast
import math
import os
import subprocess
import tempfile

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Button, Panel, Spinner, TextAreaInput, TextInput
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
                    "temp_dir",
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
                    full_out = ""
                    for line in f_out:
                        ub_matrix = np.inv(np.transpose(list(map(float, line.split()[-9:]))))
                        full_out = full_out + str(ub_matrix)
                    output_textarea.value = full_out
            except FileNotFoundError:
                print("No results from spind")

    process_button = Button(label="Process", button_type="primary")
    process_button.on_click(process_button_callback)

    output_textarea = TextAreaInput(title="Output UB matrix:", rows=7)

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
        output_textarea,
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

            # here is assumed that only omega_angle can vary
            sttC = dat["pol_angle"][0]
            om = dat["rot_angle"]
            nuC = dat["tlt_angle"][0]
            ddist = dat["ddist"]

            for roi in rois:
                x0, xN, y0, yN, fr0, frN = roi
                data_roi = dat["data"][fr0:frN, y0:yN, x0:xN]

                # omega fit Here one should change to any rot_angle, i.e. phi
                cnts = np.sum(data_roi, axis=(1, 2))
                coeff, _ = curve_fit(gauss, range(len(cnts)), cnts, p0=p0, maxfev=maxfev)

                m = cnts.mean()
                sd = cnts.std()
                snr_cnts = np.where(sd == 0, 0, m / sd)

                frC = fr0 + coeff[1]
                omF = om[math.floor(frC)]
                omC = om[math.ceil(frC)]
                frStep = frC - math.floor(frC)
                omStep = omC - omF
                omP = omF + omStep * frStep
                Int = coeff[1] * abs(coeff[2] * omStep) * math.sqrt(2) * math.sqrt(np.pi)

                # gamma fit
                projX = np.sum(data_roi, axis=(0, 1))
                coeff, _ = curve_fit(gauss, range(len(projX)), projX, p0=p0, maxfev=maxfev)
                x = x0 + coeff[1]
                x_pos = x0 + round(coeff[1])

                # nu fit
                projY = np.sum(data_roi, axis=(0, 2))
                coeff, _ = curve_fit(gauss, range(len(projY)), projY, p0=p0, maxfev=maxfev)
                y = y0 + coeff[1]
                y_pos = y0 + round(coeff[1])

                ga, nu = pyzebra.det2pol(ddist, sttC, nuC, x, y)
                diff_vector = pyzebra.z1frmd(
                    dat["wave"], ga, omP, dat["chi_angle"][0], dat["phi_angle"][0], nu[0]
                )
                d_spacing = float(pyzebra.dandth(dat["wave"], diff_vector)[0])
                dv1, dv2, dv3 = diff_vector.flatten() * 1e10

                f.write(f"{x_pos} {y_pos} {Int} {snr_cnts} {dv1} {dv2} {dv3} {d_spacing}\n")
