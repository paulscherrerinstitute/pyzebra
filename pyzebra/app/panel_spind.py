import os
import subprocess
import tempfile

import numpy as np
from bokeh.io import curdoc
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

import pyzebra


def create():
    doc = curdoc()
    events_data = doc.events_data

    npeaks_spinner = Spinner(title="Number of peaks from hdf_view panel:", disabled=True)
    lattice_const_textinput = TextInput(title="Lattice constants:")
    max_res_spinner = Spinner(title="max-res:", value=2, step=0.01, width=145)
    seed_pool_size_spinner = Spinner(title="seed-pool-size:", value=5, step=0.01, width=145)
    seed_len_tol_spinner = Spinner(title="seed-len-tol:", value=0.02, step=0.01, width=145)
    seed_angle_tol_spinner = Spinner(title="seed-angle-tol:", value=1, step=0.01, width=145)
    eval_hkl_tol_spinner = Spinner(title="eval-hkl-tol:", value=0.15, step=0.01, width=145)

    diff_vec = []
    ub_matrices = []

    def process_button_callback():
        # drop table selection to clear result fields
        results_table_source.selected.indices = []

        nonlocal diff_vec
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_peak_list_dir = os.path.join(temp_dir, "peak_list")
            os.mkdir(temp_peak_list_dir)
            temp_event_file = os.path.join(temp_peak_list_dir, "event-0.txt")
            temp_hkl_file = os.path.join(temp_dir, "hkl.h5")

            comp_proc = subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "2",
                    "python",
                    os.path.join(doc.spind_path, "gen_hkl_table.py"),
                    lattice_const_textinput.value,
                    "--max-res",
                    str(max_res_spinner.value),
                    "-o",
                    temp_hkl_file,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(" ".join(comp_proc.args))
            print(comp_proc.stdout)

            # prepare an event file
            diff_vec = []
            with open(temp_event_file, "w") as f:
                npeaks = len(next(iter(doc.events_data.values())))
                for ind in range(npeaks):
                    wave = events_data["wave"][ind]
                    ddist = events_data["ddist"][ind]
                    x_pos = events_data["x_pos"][ind]
                    y_pos = events_data["y_pos"][ind]
                    intensity = events_data["intensity"][ind]
                    snr_cnts = events_data["snr_cnts"][ind]
                    gamma = events_data["gamma"][ind]
                    omega = events_data["omega"][ind]
                    chi = events_data["chi"][ind]
                    phi = events_data["phi"][ind]
                    nu = events_data["nu"][ind]

                    ga, nu = pyzebra.det2pol(ddist, gamma, nu, x_pos, y_pos)
                    diff_vector = pyzebra.z1frmd(wave, ga, omega, chi, phi, nu)
                    d_spacing = float(pyzebra.dandth(wave, diff_vector)[0])
                    diff_vector = diff_vector.flatten() * 1e10
                    dv1, dv2, dv3 = diff_vector

                    diff_vec.append(diff_vector)
                    f.write(
                        f"{x_pos} {y_pos} {intensity} {snr_cnts} {dv1} {dv2} {dv3} {d_spacing}\n"
                    )

            print(f"Content of {temp_event_file}:")
            with open(temp_event_file) as f:
                print(f.read())

            comp_proc = subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "2",
                    "python",
                    os.path.join(doc.spind_path, "SPIND.py"),
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
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(" ".join(comp_proc.args))
            print(comp_proc.stdout)

            spind_out_file = os.path.join(temp_dir, "spind.txt")
            spind_res = dict(
                label=[], crystal_id=[], match_rate=[], matched_peaks=[], column_5=[], ub_matrix=[]
            )
            try:
                with open(spind_out_file) as f_out:
                    for line in f_out:
                        c1, c2, c3, c4, c5, *c_rest = line.split()
                        spind_res["label"].append(c1)
                        spind_res["crystal_id"].append(c2)
                        spind_res["match_rate"].append(c3)
                        spind_res["matched_peaks"].append(c4)
                        spind_res["column_5"].append(c5)

                        # last digits are spind UB matrix
                        vals = list(map(float, c_rest))
                        ub_matrix_spind = np.transpose(np.array(vals).reshape(3, 3))
                        ub_matrices.append(ub_matrix_spind)
                        spind_res["ub_matrix"].append(str(ub_matrix_spind * 1e-10))

                print(f"Content of {spind_out_file}:")
                with open(spind_out_file) as f:
                    print(f.read())

            except FileNotFoundError:
                print("No results from spind")

            results_table_source.data.update(spind_res)

    process_button = Button(label="Process", button_type="primary")
    process_button.on_click(process_button_callback)

    if doc.spind_path is None:
        process_button.disabled = True

    ub_matrix_textareainput = TextAreaInput(title="UB matrix:", rows=7, width=400)
    hkl_textareainput = TextAreaInput(title="hkl values:", rows=7, width=400)

    def results_table_select_callback(_attr, old, new):
        if new:
            ind = new[0]
            ub_matrix_spind = ub_matrices[ind]
            res = ""
            for vec in diff_vec:
                res += f"{np.linalg.inv(ub_matrix_spind) @ vec}\n"
            ub_matrix_textareainput.value = str(ub_matrix_spind * 1e-10)
            hkl_textareainput.value = res
        else:
            ub_matrix_textareainput.value = ""
            hkl_textareainput.value = ""

    results_table_source = ColumnDataSource(
        dict(label=[], crystal_id=[], match_rate=[], matched_peaks=[], column_5=[], ub_matrix=[])
    )
    results_table = DataTable(
        source=results_table_source,
        columns=[
            TableColumn(field="label", title="Label", width=50),
            TableColumn(field="crystal_id", title="Crystal ID", width=100),
            TableColumn(field="match_rate", title="Match Rate", width=100),
            TableColumn(field="matched_peaks", title="Matched Peaks", width=100),
            TableColumn(field="column_5", title="", width=100),
            TableColumn(field="ub_matrix", title="UB Matrix", width=700),
        ],
        height=300,
        width=1200,
        autosize_mode="none",
        index_position=None,
    )

    results_table_source.selected.on_change("indices", results_table_select_callback)

    tab_layout = row(
        column(
            npeaks_spinner,
            lattice_const_textinput,
            row(max_res_spinner, seed_pool_size_spinner),
            row(seed_len_tol_spinner, seed_angle_tol_spinner),
            row(eval_hkl_tol_spinner),
            process_button,
        ),
        column(results_table, row(ub_matrix_textareainput, hkl_textareainput)),
    )

    async def update_npeaks_spinner():
        npeaks = len(next(iter(doc.events_data.values())))
        npeaks_spinner.value = npeaks
        # TODO: check cell parameter for consistency?
        if npeaks:
            lattice_const_textinput.value = ",".join(map(str, doc.events_data["cell"][0]))

    doc.add_periodic_callback(update_npeaks_spinner, 1000)

    return Panel(child=tab_layout, title="spind")
