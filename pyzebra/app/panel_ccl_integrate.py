import base64
import io
import os
import tempfile
import types
from copy import deepcopy

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    Button,
    CheckboxEditor,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    DataTable,
    Div,
    Dropdown,
    FileInput,
    Grid,
    Legend,
    Line,
    LinearAxis,
    MultiSelect,
    NumberEditor,
    Panel,
    PanTool,
    Plot,
    RadioButtonGroup,
    ResetTool,
    Scatter,
    Select,
    Spacer,
    Span,
    Spinner,
    TableColumn,
    TextAreaInput,
    TextInput,
    Toggle,
    WheelZoomTool,
    Whisker,
)

import pyzebra
from pyzebra.ccl_io import AREA_METHODS


javaScript = """
setTimeout(function() {
    if (js_data.data['cont'][0] === "") return 0;
    const filename = 'output' + js_data.data['ext'][0]
    const blob = new Blob([js_data.data['cont'][0]], {type: 'text/plain'})
    const link = document.createElement('a');
    document.body.appendChild(link);
    const url = window.URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(link);
}, 3000);
"""

PROPOSAL_PATH = "/afs/psi.ch/project/sinqdata/2020/zebra/"


def create():
    det_data = {}
    fit_params = {}
    js_data = {
        ".comm": ColumnDataSource(data=dict(cont=[], ext=[])),
        ".incomm": ColumnDataSource(data=dict(cont=[], ext=[])),
    }

    def proposal_textinput_callback(_attr, _old, new):
        ccl_path = os.path.join(PROPOSAL_PATH, new.strip())
        ccl_file_list = []
        for file in os.listdir(ccl_path):
            if file.endswith((".ccl", ".dat")):
                ccl_file_list.append((os.path.join(ccl_path, file), file))
        file_select.options = ccl_file_list
        file_select.value = ccl_file_list[0][0]

    proposal_textinput = TextInput(title="Enter proposal number:", default_size=145)
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def _init_datatable():
        scan_list = [s["idx"] for s in det_data]
        hkl = [f'{s["h"]} {s["k"]} {s["l"]}' for s in det_data]
        export = [s.get("active", True) for s in det_data]
        scan_table_source.data.update(
            scan=scan_list, hkl=hkl, fit=[0] * len(scan_list), export=export,
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        merge_options = [(str(i), f"{i} ({idx})") for i, idx in enumerate(scan_list)]
        merge_source_select.options = merge_options
        merge_source_select.value = merge_options[0][0]
        merge_dest_select.options = merge_options
        merge_dest_select.value = merge_options[0][0]

    def ccl_file_select_callback(_attr, _old, _new):
        pass

    file_select = Select(title="Available .ccl/.dat files:")
    file_select.on_change("value", ccl_file_select_callback)

    def file_open_button_callback():
        nonlocal det_data
        with open(file_select.value) as file:
            _, ext = os.path.splitext(file_select.value)
            det_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(det_data, monitor_spinner.value)
        pyzebra.merge_duplicates(det_data)

        _init_datatable()

    file_open_button = Button(label="Open", default_size=100)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        with open(file_select.value) as file:
            _, ext = os.path.splitext(file_select.value)
            append_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(append_data, monitor_spinner.value)
        pyzebra.merge_datasets(det_data, append_data)

        _init_datatable()

    file_append_button = Button(label="Append", default_size=100)
    file_append_button.on_click(file_append_button_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            det_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(det_data, monitor_spinner.value)
        pyzebra.merge_duplicates(det_data)

        _init_datatable()

    upload_div = Div(text="or upload .ccl/.dat file:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".ccl,.dat")
    upload_button.on_change("value", upload_button_callback)

    def append_upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(append_upload_button.filename)
            append_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(append_data, monitor_spinner.value)
        pyzebra.merge_datasets(det_data, append_data)

        _init_datatable()

    append_upload_div = Div(text="append extra file:", margin=(5, 5, 0, 5))
    append_upload_button = FileInput(accept=".ccl,.dat")
    append_upload_button.on_change("value", append_upload_button_callback)

    def monitor_spinner_callback(_attr, old, new):
        if det_data:
            pyzebra.normalize_dataset(det_data, new)
            _update_plot(_get_selected_scan())

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in det_data]
        scan_table_source.data.update(fit=fit_ok)

    def _update_plot(scan):
        scan_motor = scan["scan_motor"]

        y = scan["Counts"]
        x = scan[scan_motor]

        plot.axis[0].axis_label = scan_motor
        plot_scatter_source.data.update(x=x, y=y, y_upper=y + np.sqrt(y), y_lower=y - np.sqrt(y))

        fit = scan.get("fit")
        if fit is not None:
            x_fit = np.linspace(x[0], x[-1], 100)
            plot_fit_source.data.update(x=x_fit, y=fit.eval(x=x_fit))

            for i, model in enumerate(fit_params):
                if "background" in model:
                    comps = fit.eval_components(x=x_fit)
                    plot_bkg_source.data.update(x=x_fit, y=comps[f"f{i}_"])
                    break
            else:
                plot_bkg_source.data.update(x=[], y=[])

            fit_output_textinput.value = fit.fit_report()

            # numfit_min, numfit_max = fit["numfit"]
            # if numfit_min is None:
            #     numfit_min_span.location = None
            # else:
            #     numfit_min_span.location = x[numfit_min]

            # if numfit_max is None:
            #     numfit_max_span.location = None
            # else:
            #     numfit_max_span.location = x[numfit_max]

        else:
            plot_fit_source.data.update(x=[], y=[])
            plot_bkg_source.data.update(x=[], y=[])
            fit_output_textinput.value = ""
            numfit_min_span.location = None
            numfit_max_span.location = None

    # Main plot
    plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=470, plot_width=700)

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_scatter_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot_scatter = plot.add_glyph(
        plot_scatter_source, Scatter(x="x", y="y", line_color="steelblue")
    )
    plot.add_layout(Whisker(source=plot_scatter_source, base="x", upper="y_upper", lower="y_lower"))

    plot_fit_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_fit = plot.add_glyph(plot_fit_source, Line(x="x", y="y"))

    plot_bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_bkg = plot.add_glyph(
        plot_bkg_source, Line(x="x", y="y", line_color="green", line_dash="dashed")
    )

    numfit_min_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(numfit_min_span)

    numfit_max_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(numfit_max_span)

    plot.add_layout(
        Legend(
            items=[("data", [plot_scatter]), ("best fit", [plot_fit]), ("background", [plot_bkg])],
            location="top_left",
        )
    )

    plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    plot.toolbar.logo = None

    # Scan select
    def scan_table_select_callback(_attr, old, new):
        if not new:
            # skip empty selections
            return

        # Avoid selection of multiple indicies (via Shift+Click or Ctrl+Click)
        if len(new) > 1:
            # drop selection to the previous one
            scan_table_source.selected.indices = old
            return

        if len(old) > 1:
            # skip unnecessary update caused by selection drop
            return

        _update_plot(det_data[new[0]])

    scan_table_source = ColumnDataSource(dict(scan=[], hkl=[], fit=[], export=[]))
    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="scan", title="Scan", width=50),
            TableColumn(field="hkl", title="hkl", width=100),
            TableColumn(field="fit", title="Fit", width=50),
            TableColumn(field="export", title="Export", editor=CheckboxEditor(), width=50),
        ],
        width=310,  # +60 because of the index column
        fit_columns=False,
        editable=True,
    )

    scan_table_source.selected.on_change("indices", scan_table_select_callback)

    def _get_selected_scan():
        return det_data[scan_table_source.selected.indices[0]]

    merge_dest_select = Select(title="destination:", width=100)
    merge_source_select = Select(title="source:", width=100)

    def merge_button_callback():
        scan_dest_ind = int(merge_dest_select.value)
        scan_source_ind = int(merge_source_select.value)

        if scan_dest_ind == scan_source_ind:
            print("WARNING: Selected scans for merging are identical")
            return

        pyzebra.merge_scans(det_data[scan_dest_ind], det_data[scan_source_ind])
        _update_plot(_get_selected_scan())

    merge_button = Button(label="Merge scans", width=145)
    merge_button.on_click(merge_button_callback)

    integ_from = Spinner(title="Integrate from:", default_size=145, disabled=True)
    integ_to = Spinner(title="to:", default_size=145, disabled=True)

    def fitparams_add_dropdown_callback(click):
        # bokeh requires (str, str) for MultiSelect options
        new_tag = f"{click.item}-{fitparams_select.tags[0]}"
        fitparams_select.options.append((new_tag, click.item))
        fit_params[new_tag] = fitparams_factory(click.item)
        fitparams_select.tags[0] += 1

    fitparams_add_dropdown = Dropdown(
        label="Add fit function",
        menu=[
            ("Background", "background"),
            ("Gauss", "gauss"),
            ("Voigt", "voigt"),
            ("Pseudo Voigt", "pseudovoigt"),
            # ("Pseudo Voigt1", "pseudovoigt1"),
        ],
        default_size=145,
    )
    fitparams_add_dropdown.on_click(fitparams_add_dropdown_callback)

    def fitparams_select_callback(_attr, old, new):
        # Avoid selection of multiple indicies (via Shift+Click or Ctrl+Click)
        if len(new) > 1:
            # drop selection to the previous one
            fitparams_select.value = old
            return

        if len(old) > 1:
            # skip unnecessary update caused by selection drop
            return

        if new:
            fitparams_table_source.data.update(fit_params[new[0]])
        else:
            fitparams_table_source.data.update(dict(param=[], value=[], vary=[], min=[], max=[]))

    fitparams_select = MultiSelect(options=[], height=120, default_size=145)
    fitparams_select.tags = [0]
    fitparams_select.on_change("value", fitparams_select_callback)

    def fitparams_remove_button_callback():
        if fitparams_select.value:
            sel_tag = fitparams_select.value[0]
            del fit_params[sel_tag]
            for elem in fitparams_select.options:
                if elem[0] == sel_tag:
                    fitparams_select.options.remove(elem)
                    break

            fitparams_select.value = []

    fitparams_remove_button = Button(label="Remove fit function", default_size=145)
    fitparams_remove_button.on_click(fitparams_remove_button_callback)

    def fitparams_factory(function):
        if function == "background":
            params = ["slope", "intercept"]
        elif function == "gauss":
            params = ["center", "sigma", "amplitude"]
        elif function == "voigt":
            params = ["center", "sigma", "amplitude", "gamma"]
        elif function == "pseudovoigt":
            params = ["center", "sigma", "amplitude", "fraction"]
        elif function == "pseudovoigt1":
            params = ["center", "g_sigma", "l_sigma", "amplitude", "fraction"]
        else:
            raise ValueError("Unknown fit function")

        n = len(params)
        fitparams = dict(
            param=params, value=[None] * n, vary=[True] * n, min=[None] * n, max=[None] * n,
        )

        return fitparams

    fitparams_table_source = ColumnDataSource(dict(param=[], value=[], vary=[], min=[], max=[]))
    fitparams_table = DataTable(
        source=fitparams_table_source,
        columns=[
            TableColumn(field="param", title="Parameter"),
            TableColumn(field="value", title="Value", editor=NumberEditor()),
            TableColumn(field="vary", title="Vary", editor=CheckboxEditor()),
            TableColumn(field="min", title="Min", editor=NumberEditor()),
            TableColumn(field="max", title="Max", editor=NumberEditor()),
        ],
        height=200,
        width=350,
        index_position=None,
        editable=True,
        auto_edit=True,
    )

    # start with `background` and `gauss` fit functions added
    fitparams_add_dropdown_callback(types.SimpleNamespace(item="background"))
    fitparams_add_dropdown_callback(types.SimpleNamespace(item="gauss"))
    fitparams_select.value = ["gauss-1"]  # add selection to gauss

    fit_output_textinput = TextAreaInput(title="Fit results:", width=750, height=200)

    def fit_all_button_callback():
        for scan in det_data:
            pyzebra.fit_scan(scan, fit_params)

        _update_plot(_get_selected_scan())
        _update_table()

    fit_all_button = Button(label="Fit All", button_type="primary", default_size=145)
    fit_all_button.on_click(fit_all_button_callback)

    def fit_button_callback():
        scan = _get_selected_scan()
        pyzebra.fit_scan(scan, fit_params)

        _update_plot(scan)
        _update_table()

    fit_button = Button(label="Fit Current", default_size=145)
    fit_button.on_click(fit_button_callback)

    area_method_radiobutton = RadioButtonGroup(
        labels=["Fit area", "Int area"], active=0, default_size=145, disabled=True
    )

    bin_size_spinner = Spinner(
        title="Bin size:", value=1, low=1, step=1, default_size=145, disabled=True
    )

    lorentz_toggle = Toggle(label="Lorentz Correction", default_size=145)

    preview_output_textinput = TextAreaInput(title="Export file preview:", width=500, height=400)

    def preview_output_button_callback():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = deepcopy(det_data)
            for s, export in enumerate(scan_table_source.data["export"]):
                if not export:
                    if "fit" in export_data[s]:
                        del export_data[s]["fit"]

            pyzebra.export_1D(
                export_data,
                temp_file,
                area_method=AREA_METHODS[int(area_method_radiobutton.active)],
                lorentz=lorentz_toggle.active,
                hkl_precision=int(hkl_precision_select.value),
            )

            exported_content = ""
            for ext in (".comm", ".incomm"):
                fname = temp_file + ext
                if os.path.isfile(fname):
                    with open(fname) as f:
                        exported_content += f"{ext} file:\n" + f.read()

            preview_output_textinput.value = exported_content

    preview_output_button = Button(label="Preview file", default_size=200)
    preview_output_button.on_click(preview_output_button_callback)

    hkl_precision_select = Select(
        title="hkl precision:", options=["2", "3", "4"], value="2", default_size=80
    )

    def save_button_callback():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = deepcopy(det_data)
            for s, export in enumerate(scan_table_source.data["export"]):
                if not export:
                    del export_data[s]

            pyzebra.export_1D(
                export_data,
                temp_file,
                area_method=AREA_METHODS[int(area_method_radiobutton.active)],
                lorentz=lorentz_toggle.active,
                hkl_precision=int(hkl_precision_select.value),
            )

            for ext in (".comm", ".incomm"):
                fname = temp_file + ext
                if os.path.isfile(fname):
                    with open(fname) as f:
                        cont = f.read()
                else:
                    cont = ""
                js_data[ext].data.update(cont=[cont], ext=[ext])

    save_button = Button(label="Download file", button_type="success", default_size=200)
    save_button.on_click(save_button_callback)
    save_button.js_on_click(CustomJS(args={"js_data": js_data[".comm"]}, code=javaScript))
    save_button.js_on_click(CustomJS(args={"js_data": js_data[".incomm"]}, code=javaScript))

    fitpeak_controls = row(
        column(fitparams_add_dropdown, fitparams_select, fitparams_remove_button),
        fitparams_table,
        Spacer(width=20),
        column(
            row(integ_from, integ_to),
            row(bin_size_spinner, column(Spacer(height=19), lorentz_toggle)),
            row(area_method_radiobutton),
            row(fit_button, fit_all_button),
        ),
    )

    scan_layout = column(
        scan_table,
        row(column(Spacer(height=19), merge_button), merge_dest_select, merge_source_select),
    )

    export_layout = column(
        preview_output_textinput,
        row(
            hkl_precision_select, column(Spacer(height=19), row(preview_output_button, save_button))
        ),
    )

    tab_layout = column(
        row(
            proposal_textinput,
            file_select,
            column(Spacer(height=19), row(file_open_button, file_append_button)),
            Spacer(width=100),
            column(upload_div, upload_button),
            column(append_upload_div, append_upload_button),
            monitor_spinner,
        ),
        row(scan_layout, plot, Spacer(width=30), export_layout),
        row(fitpeak_controls, fit_output_textinput),
    )

    return Panel(child=tab_layout, title="ccl integrate")
