import base64
import io
import os
import subprocess
import tempfile

from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    DataRange1d,
    Div,
    FileInput,
    MultiSelect,
    NumericInput,
    Panel,
    Plot,
    RadioGroup,
    Select,
    Spacer,
    TextAreaInput,
    TextInput,
)

import pyzebra


def create():
    ang_lims = None
    cif_data = None
    params = None
    res_files = {}

    anglim_div = Div(text="Angular min/max limits:")
    sttgamma_ti = TextInput(title="stt/gamma", width=100)
    omega_ti = TextInput(title="omega", width=100)
    chinu_ti = TextInput(title="chi/nu", width=100)
    phi_ti = TextInput(title="phi", width=100)

    def _update_ang_lims(ang_lims):
        sttgamma_ti.value = " ".join(ang_lims["gamma"][:2])
        omega_ti.value = " ".join(ang_lims["omega"][:2])
        if ang_lims["geom"] == "nb":
            chinu_ti.value = " ".join(ang_lims["nu"][:2])
            phi_ti.value = ""
        else:  # ang_lims["geom"] == "bi"
            chinu_ti.value = " ".join(ang_lims["chi"][:2])
            phi_ti.value = " ".join(ang_lims["phi"][:2])

    def _update_params(params):
        if "WAVE" in params:
            wavelen_input.value = float(params["WAVE"])
        if "SPGR" in params:
            cryst_space_group.value = params["SPGR"]
        if "CELL" in params:
            cryst_cell.value = params["CELL"]
        if "UBMAT" in params:
            ub_matrix.value = " ".join(params["UBMAT"])
        if "HLIM" in params:
            ranges_hkl.value = params["HLIM"]
        if "SRANG" in params:
            ranges_expression.value = params["SRANG"]

    def open_geom_callback(_attr, _old, new):
        nonlocal ang_lims
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            ang_lims = pyzebra.read_geom_file(fileobj)
        _update_ang_lims(ang_lims)

    open_geom_div = Div(text="or open GEOM:")
    open_geom = FileInput(accept=".geom", width=200)
    open_geom.on_change("value", open_geom_callback)

    def open_cfl_callback(_attr, _old, new):
        nonlocal params
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            params = pyzebra.read_cfl_file(fileobj)
            _update_params(params)

    open_cfl_div = Div(text="or open CFL:")
    open_cfl = FileInput(accept=".cfl", width=200)
    open_cfl.on_change("value", open_cfl_callback)

    def open_cif_callback(_attr, _old, new):
        nonlocal cif_data
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            cif_data = pyzebra.read_cif_file(fileobj)
            _update_params(cif_data)

    open_cif_div = Div(text="or open CIF:")
    open_cif = FileInput(accept=".cif", width=200)
    open_cif.on_change("value", open_cif_callback)

    wavelen_input = NumericInput(title="\u200B", width=70, mode="float")

    def wavelen_select_callback(_attr, _old, new):
        if new:
            wavelen_input.value = float(new)
        else:
            wavelen_input.value = None

    wavelen_select = Select(
        title="Wavelength:", options=["", "0.788", "1.178", "1.383", "2.305"], width=70
    )
    wavelen_select.on_change("value", wavelen_select_callback)

    cryst_div = Div(text="Crystal structure:")
    cryst_space_group = TextInput(title="space group", width=100)
    cryst_cell = TextInput(title="cell", width=500)

    def ub_matrix_calc_callback():
        params = dict()
        params["SPGR"] = cryst_space_group.value
        params["CELL"] = cryst_cell.value
        ub = pyzebra.calc_ub_matrix(params)
        ub_matrix.value = " ".join(ub)

    ub_matrix_calc = Button(label="UB matrix:", button_type="primary", width=100)
    ub_matrix_calc.on_click(ub_matrix_calc_callback)

    ub_matrix = TextInput(title="\u200B", width=600)

    ranges_div = Div(text="Ranges:")
    ranges_hkl = TextInput(title="HKL", value="-25 25 -25 25 -25 25")
    ranges_expression = TextInput(title="sin(​θ​)/λ", value="0.0 0.7", width=200)

    def geom_radiogroup_callback(_attr, _old, new):
        nonlocal ang_lims, params
        if new == 0:
            geom_file = pyzebra.get_zebraBI_default_geom_file()
        else:
            geom_file = pyzebra.get_zebraNB_default_geom_file()
        cfl_file = pyzebra.get_zebra_default_cfl_file()

        ang_lims = pyzebra.read_geom_file(geom_file)
        _update_ang_lims(ang_lims)
        params = pyzebra.read_cfl_file(cfl_file)
        _update_params(params)

    geom_radiogroup_div = Div(text="Geometry:")
    geom_radiogroup = RadioGroup(labels=["bisecting", "normal beam"], width=150)
    geom_radiogroup.on_change("active", geom_radiogroup_callback)
    geom_radiogroup.active = 0

    mag_struct_div = Div(text="Magnetic structure (optional):")
    mag_struct_lattice = TextInput(title="lattice", width=150)
    mag_struct_kvec = TextAreaInput(title="k vector", width=150)

    def go1_button_callback():
        if open_geom.value:
            geom_template = io.StringIO(base64.b64decode(open_geom.value).decode())
        else:
            geom_template = None

        if open_cfl.value:
            cfl_template = io.StringIO(base64.b64decode(open_cfl.value).decode())
        else:
            cfl_template = None

        ang_lims["gamma"][0], ang_lims["gamma"][1] = sttgamma_ti.value.strip().split()
        ang_lims["omega"][0], ang_lims["omega"][1] = omega_ti.value.strip().split()
        if ang_lims["geom"] == "nb":
            ang_lims["nu"][0], ang_lims["nu"][1] = chinu_ti.value.strip().split()
        else:  # ang_lims["geom"] == "bi"
            ang_lims["chi"][0], ang_lims["chi"][1] = chinu_ti.value.strip().split()
            ang_lims["phi"][0], ang_lims["phi"][1] = phi_ti.value.strip().split()

        if cif_data:
            params.update(cif_data)

        params["WAVE"] = str(wavelen_input.value)
        params["SPGR"] = cryst_space_group.value
        params["CELL"] = cryst_cell.value
        params["UBMAT"] = ub_matrix.value.split()
        params["HLIM"] = ranges_hkl.value
        params["SRANG"] = ranges_expression.value

        with tempfile.TemporaryDirectory() as temp_dir:
            geom_path = os.path.join(temp_dir, "zebra.geom")
            pyzebra.export_geom_file(geom_path, ang_lims, geom_template)

            print(f"Content of {geom_path}:")
            with open(geom_path) as f:
                print(f.read())

            cfl_path = os.path.join(temp_dir, "zebra.cfl")
            pyzebra.export_cfl_file(cfl_path, params, cfl_template)

            print(f"Content of {cfl_path}:")
            with open(cfl_path) as f:
                print(f.read())

            comp_proc = subprocess.run(
                [pyzebra.SXTAL_REFGEN_PATH, cfl_path],
                cwd=temp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(" ".join(comp_proc.args))
            print(comp_proc.stdout)

            # display created lists
            with open(os.path.join(temp_dir, "zebra.hkl")) as f:
                res_files["zebra.hkl"] = f.read()

            with open(os.path.join(temp_dir, "zebra.mhkl")) as f:
                res_files["zebra.mhkl"] = f.read()

            created_lists.options = list(res_files)

    go1_button = Button(label="GO", button_type="primary", width=50)
    go1_button.on_click(go1_button_callback)

    sorting_div = Div(text="Sorting:")
    sorting_1 = TextInput(title="1st", width=50)
    sorting_1_dt = TextInput(title="Δ", width=50)
    sorting_2 = TextInput(title="2nd", width=50)
    sorting_2_dt = TextInput(title="Δ", width=50)
    sorting_3 = TextInput(title="3rd", width=50)
    sorting_3_dt = TextInput(title="Δ", width=50)
    sorting_go = Button(label="GO", button_type="primary", width=50)

    def created_lists_callback(_attr, _old, new):
        preview_lists.value = res_files[new[0]]

    created_lists = MultiSelect(title="Created lists:", width=200, height=150)
    created_lists.on_change("value", created_lists_callback)
    preview_lists = TextAreaInput(title="Preview selected list:", width=600, height=150)

    download_file = Button(label="Download file", button_type="success", width=200)
    plot_list = Button(label="Plot selected list", button_type="primary", width=200)

    measured_data_div = Div(text="Measured data:")
    measured_data = FileInput(accept=".comm,.incomm", width=200)
    plot_file = Button(label="Plot selected file", button_type="primary", width=200)

    plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=450, plot_width=500)
    plot.toolbar.logo = None

    hkl_normal = TextInput(title="HKL normal", width=100)
    delta = TextInput(title="delta", width=100)
    in_plane_x = TextInput(title="in-plane X", width=100)
    in_plane_y = TextInput(title="in-plane Y", width=100)

    disting_opt_div = Div(text="Distinguish options:")
    disting_opt_cb = CheckboxGroup(
        labels=["files (symbols)", "intensities (size)", "k vectors nucl/magn (colors)"], width=200,
    )
    disting_opt_rb = RadioGroup(labels=["scan direction", "resolution ellipsoid"], width=200)

    clear_plot = Button(label="Clear plot", button_type="warning", width=200)

    util_column1 = column(
        row(column(geom_radiogroup_div, geom_radiogroup), column(open_cfl_div, open_cfl)),
        row(wavelen_select, wavelen_input, column(open_cif_div, open_cif)),
    )

    anglim_layout = column(
        anglim_div,
        row(sttgamma_ti, Spacer(width=10), omega_ti),
        row(chinu_ti, Spacer(width=10), phi_ti),
    )

    column1_layout = column(
        row(util_column1, row(anglim_layout, column(open_geom_div, open_geom))),
        row(cryst_div, cryst_space_group, cryst_cell),
        row(column(Spacer(height=18), ub_matrix_calc), ub_matrix),
        row(ranges_div, ranges_hkl, ranges_expression),
        row(
            mag_struct_div,
            mag_struct_lattice,
            mag_struct_kvec,
            column(Spacer(height=18), go1_button),
        ),
        row(
            sorting_div,
            sorting_1,
            sorting_1_dt,
            Spacer(width=50),
            sorting_2,
            sorting_2_dt,
            Spacer(width=50),
            sorting_3,
            sorting_3_dt,
            column(Spacer(height=18), sorting_go),
        ),
        row(created_lists, preview_lists),
        row(download_file, plot_list),
    )

    column2_layout = column(
        row(column(measured_data_div, measured_data), plot_file),
        plot,
        row(hkl_normal, delta, Spacer(width=50), in_plane_x, in_plane_y),
        row(disting_opt_div, disting_opt_cb, disting_opt_rb),
        row(clear_plot),
    )

    tab_layout = row(column1_layout, column2_layout)

    return Panel(child=tab_layout, title="ccl prepare")
