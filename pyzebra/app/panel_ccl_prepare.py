import base64
import io

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
    def _update_ang_lims(ang_lims):
        sttgamma_min.value, sttgamma_max.value, _ = ang_lims["gamma"]
        omega_min.value, omega_max.value, _ = ang_lims["omega"]
        chinu_min.value, chinu_max.value, _ = ang_lims.get("chi") or ang_lims.get("nu")
        phi_min.value, phi_max.value, _ = ang_lims["phi"]

    def geom_radiogroup_callback(_attr, _old, new):
        if new == 0:
            geom_file = pyzebra.get_zebraBI_default_geom_file()
        else:
            geom_file = pyzebra.get_zebraNB_default_geom_file()

        _update_ang_lims(pyzebra.read_ang_limits(geom_file))

    geom_radiogroup_div = Div(text="Geometry:")
    geom_radiogroup = RadioGroup(labels=["bisecting", "normal beam"], width=150)
    geom_radiogroup.on_change("active", geom_radiogroup_callback)

    def open_geom_callback(_attr, _old, new):
        with io.StringIO(base64.b64decode(new).decode()) as geom_file:
            _update_ang_lims(pyzebra.read_ang_limits(geom_file))

    open_geom_div = Div(text="or open GEOM:")
    open_geom = FileInput(accept=".geom", width=200)
    open_geom.on_change("value", open_geom_callback)

    anglim_div = Div(text="Angular min/max limits:")
    sttgamma_min = NumericInput(title="stt/gamma", width=50, mode="float")
    sttgamma_max = NumericInput(title="\u200B", width=50, mode="float")
    omega_min = NumericInput(title="omega", width=50, mode="float")
    omega_max = NumericInput(title="\u200B", width=50, mode="float")
    chinu_min = NumericInput(title="chi/nu", width=50, mode="float")
    chinu_max = NumericInput(title="\u200B", width=50, mode="float")
    phi_min = NumericInput(title="phi", width=50, mode="float")
    phi_max = NumericInput(title="\u200B", width=50, mode="float")

    open_cfl_div = Div(text="or open CFL:")
    open_cfl = FileInput(accept=".cfl", width=200)

    open_cif_div = Div(text="or open CIF:")
    open_cif = FileInput(accept=".cif", width=200)

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
    ranges_hkl = TextInput(title="HKL")
    ranges_expression = TextInput(title="sin(​θ​)/λ", width=200)

    mag_struct_div = Div(text="Magnetic structure (optional):")
    mag_struct_lattice = TextInput(title="lattice", width=150)
    mag_struct_kvec = TextAreaInput(title="k vector", width=150)
    mag_struct_go = Button(label="GO", button_type="primary", width=50)

    sorting_div = Div(text="Sorting:")
    sorting_1 = TextInput(title="1st", width=50)
    sorting_1_dt = TextInput(title="Δ", width=50)
    sorting_2 = TextInput(title="2nd", width=50)
    sorting_2_dt = TextInput(title="Δ", width=50)
    sorting_3 = TextInput(title="3rd", width=50)
    sorting_3_dt = TextInput(title="Δ", width=50)
    sorting_go = Button(label="GO", button_type="primary", width=50)

    created_lists = MultiSelect(title="Created lists:", width=200, height=150)
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
        row(sttgamma_min, sttgamma_max, Spacer(width=10), omega_min, omega_max),
        row(chinu_min, chinu_max, Spacer(width=10), phi_min, phi_max),
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
            column(Spacer(height=18), mag_struct_go),
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
