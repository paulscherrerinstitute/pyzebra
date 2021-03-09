import base64
import io
import itertools
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
    HoverTool,
    Legend,
    Line,
    LinearAxis,
    MultiLine,
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
    Tabs,
    TextAreaInput,
    TextInput,
    Toggle,
    WheelZoomTool,
    Whisker,
)
from bokeh.palettes import Category10, Turbo256
from bokeh.transform import linear_cmap

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


def color_palette(n_colors):
    palette = itertools.cycle(Category10[10])
    return list(itertools.islice(palette, n_colors))


def create():
    det_data = []
    fit_params = {}
    js_data = {
        ".comm": ColumnDataSource(data=dict(cont=[], ext=[])),
        ".incomm": ColumnDataSource(data=dict(cont=[], ext=[])),
    }

    def proposal_textinput_callback(_attr, _old, new):
        full_proposal_path = os.path.join(PROPOSAL_PATH, new.strip())
        dat_file_list = []
        for file in os.listdir(full_proposal_path):
            if file.endswith(".dat"):
                dat_file_list.append((os.path.join(full_proposal_path, file), file))
        file_select.options = dat_file_list
        file_select.value = dat_file_list[0][0]

    proposal_textinput = TextInput(title="Enter proposal number:", default_size=145)
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def _init_datatable():
        scan_list = [s["idx"] for s in det_data]
        file_list = []
        for scan in det_data:
            file_list.append(os.path.basename(scan["original_filename"]))

        scan_table_source.data.update(
            file=file_list,
            scan=scan_list,
            param=[""] * len(scan_list),
            fit=[0] * len(scan_list),
            export=[True] * len(scan_list),
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        param_select.value = "user defined"

    def file_select_callback(_attr, _old, _new):
        pass

    file_select = Select(title="Available .dat files")
    file_select.on_change("value", file_select_callback)

    def file_open_button_callback():
        nonlocal det_data
        with open(file_select.value) as file:
            _, ext = os.path.splitext(file_select.value)
            det_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(det_data, monitor_spinner.value)

        _init_datatable()

    file_open_button = Button(label="Open", default_size=100)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        with open(file_select.value) as file:
            _, ext = os.path.splitext(file_select.value)
            append_data = pyzebra.parse_1D(file, ext)

        pyzebra.normalize_dataset(append_data, monitor_spinner.value)
        det_data.extend(append_data)

        _init_datatable()

    file_append_button = Button(label="Append", default_size=100)
    file_append_button.on_click(file_append_button_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        det_data = []
        for f_str, f_name in zip(new, upload_button.filename):
            with io.StringIO(base64.b64decode(f_str).decode()) as file:
                _, ext = os.path.splitext(f_name)
                if det_data:
                    append_data = pyzebra.parse_1D(file, ext)
                    pyzebra.normalize_dataset(append_data, monitor_spinner.value)
                    det_data.extend(append_data)
                else:
                    det_data = pyzebra.parse_1D(file, ext)
                    pyzebra.normalize_dataset(det_data, monitor_spinner.value)

        _init_datatable()

    upload_div = Div(text="or upload .dat files:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".dat", multiple=True)
    upload_button.on_change("value", upload_button_callback)

    def append_upload_button_callback(_attr, _old, new):
        for f_str, f_name in zip(new, append_upload_button.filename):
            with io.StringIO(base64.b64decode(f_str).decode()) as file:
                _, ext = os.path.splitext(f_name)
                append_data = pyzebra.parse_1D(file, ext)

            pyzebra.normalize_dataset(append_data, monitor_spinner.value)
            det_data.extend(append_data)

        _init_datatable()

    append_upload_div = Div(text="append extra files:", margin=(5, 5, 0, 5))
    append_upload_button = FileInput(accept=".dat", multiple=True)
    append_upload_button.on_change("value", append_upload_button_callback)

    def monitor_spinner_callback(_attr, _old, new):
        if det_data:
            pyzebra.normalize_dataset(det_data, new)
            _update_plot()

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in det_data]
        scan_table_source.data.update(fit=fit_ok)

    def _update_plot():
        _update_single_scan_plot(_get_selected_scan())
        _update_overview()

    def _update_single_scan_plot(scan):
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
                if "linear" in model:
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

    def _update_overview():
        xs = []
        ys = []
        param = []
        x = []
        y = []
        par = []
        for s, p in enumerate(scan_table_source.data["param"]):
            if p:
                scan = det_data[s]
                scan_motor = scan["scan_motor"]
                xs.append(scan[scan_motor])
                x.extend(scan[scan_motor])
                ys.append(scan["Counts"])
                y.extend([float(p)] * len(scan[scan_motor]))
                param.append(float(p))
                par.extend(scan["Counts"])

        if det_data:
            scan_motor = det_data[0]["scan_motor"]
            ov_plot.axis[0].axis_label = scan_motor
            ov_param_plot.axis[0].axis_label = scan_motor

        ov_plot_mline_source.data.update(xs=xs, ys=ys, param=param, color=color_palette(len(xs)))

        if y:
            mapper["transform"].low = np.min([np.min(y) for y in ys])
            mapper["transform"].high = np.max([np.max(y) for y in ys])
        ov_param_plot_scatter_source.data.update(x=x, y=y, param=par)

    # Main plot
    plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=450, plot_width=700)

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
            items=[("data", [plot_scatter]), ("best fit", [plot_fit]), ("linear", [plot_bkg])],
            location="top_left",
        )
    )

    plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    plot.toolbar.logo = None

    # Overview multilines plot
    ov_plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=400, plot_width=700)

    ov_plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    ov_plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    ov_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    ov_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    ov_plot_mline_source = ColumnDataSource(dict(xs=[], ys=[], param=[], color=[]))
    ov_plot.add_glyph(ov_plot_mline_source, MultiLine(xs="xs", ys="ys", line_color="color"))

    hover_tool = HoverTool(tooltips=[("param", "@param")])
    ov_plot.add_tools(PanTool(), WheelZoomTool(), hover_tool, ResetTool())

    ov_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    ov_plot.toolbar.logo = None

    # Overview perams plot
    ov_param_plot = Plot(
        x_range=DataRange1d(), y_range=DataRange1d(), plot_height=400, plot_width=700
    )

    ov_param_plot.add_layout(LinearAxis(axis_label="Param"), place="left")
    ov_param_plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    ov_param_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    ov_param_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    ov_param_plot_scatter_source = ColumnDataSource(dict(x=[], y=[], param=[]))
    mapper = linear_cmap(field_name="param", palette=Turbo256, low=0, high=50)
    ov_param_plot.add_glyph(
        ov_param_plot_scatter_source,
        Scatter(x="x", y="y", line_color=mapper, fill_color=mapper, size=10),
    )

    ov_param_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    ov_param_plot.toolbar.logo = None

    # Plot tabs
    plots = Tabs(
        tabs=[
            Panel(child=plot, title="single scan"),
            Panel(child=ov_plot, title="overview"),
            Panel(child=ov_param_plot, title="overview map"),
        ]
    )

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

        _update_plot()

    scan_table_source = ColumnDataSource(dict(file=[], scan=[], param=[], fit=[], export=[]))
    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="file", title="file", width=150),
            TableColumn(field="scan", title="scan", width=50),
            TableColumn(field="param", title="param", width=50),
            TableColumn(field="fit", title="Fit", width=50),
            TableColumn(field="export", title="Export", editor=CheckboxEditor(), width=50),
        ],
        width=410,  # +60 because of the index column
        editable=True,
        fit_columns=False,
    )

    def scan_table_source_callback(_attr, _old, _new):
        if scan_table_source.selected.indices:
            _update_plot()

    scan_table_source.selected.on_change("indices", scan_table_select_callback)
    scan_table_source.on_change("data", scan_table_source_callback)

    def _get_selected_scan():
        return det_data[scan_table_source.selected.indices[0]]

    def param_select_callback(_attr, _old, new):
        if new == "user defined":
            param = [""] * len(det_data)
        else:
            param = [scan[new] for scan in det_data]

        scan_table_source.data["param"] = param

    param_select = Select(
        title="Parameter:",
        options=["user defined", "temp", "mf"],
        value="user defined",
        default_size=145,
    )
    param_select.on_change("value", param_select_callback)

    fit_from_spinner = Spinner(title="Fit from:", default_size=145)
    fit_to_spinner = Spinner(title="to:", default_size=145)

    def fitparams_add_dropdown_callback(click):
        # bokeh requires (str, str) for MultiSelect options
        new_tag = f"{click.item}-{fitparams_select.tags[0]}"
        fitparams_select.options.append((new_tag, click.item))
        fit_params[new_tag] = fitparams_factory(click.item)
        fitparams_select.tags[0] += 1

    fitparams_add_dropdown = Dropdown(
        label="Add fit function",
        menu=[
            ("Linear", "linear"),
            ("Gaussian", "gaussian"),
            ("Voigt", "voigt"),
            ("Pseudo Voigt", "pvoigt"),
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
        if function == "linear":
            params = ["slope", "intercept"]
        elif function == "gaussian":
            params = ["amplitude", "center", "sigma"]
        elif function == "voigt":
            params = ["amplitude", "center", "sigma", "gamma"]
        elif function == "pvoigt":
            params = ["amplitude", "center", "sigma", "fraction"]
        elif function == "pseudovoigt1":
            params = ["amplitude", "center", "g_sigma", "l_sigma", "fraction"]
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
    fitparams_add_dropdown_callback(types.SimpleNamespace(item="linear"))
    fitparams_add_dropdown_callback(types.SimpleNamespace(item="gaussian"))
    fitparams_select.value = ["gaussian-1"]  # add selection to gauss

    fit_output_textinput = TextAreaInput(title="Fit results:", width=750, height=200)

    def fit_all_button_callback():
        for scan in det_data:
            pyzebra.fit_scan(
                scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
            )

        _update_plot()
        _update_table()

    fit_all_button = Button(label="Fit All", button_type="primary", default_size=145)
    fit_all_button.on_click(fit_all_button_callback)

    def fit_button_callback():
        scan = _get_selected_scan()
        pyzebra.fit_scan(
            scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
        )

        _update_plot()
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

    preview_output_textinput = TextAreaInput(title="Export file preview:", width=450, height=400)

    def preview_output_button_callback():
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
            )

            exported_content = ""
            for ext in (".comm", ".incomm"):
                fname = temp_file + ext
                if os.path.isfile(fname):
                    with open(fname) as f:
                        exported_content += f"{ext} file:\n" + f.read()

            preview_output_textinput.value = exported_content

    preview_output_button = Button(label="Preview file", default_size=220)
    preview_output_button.on_click(preview_output_button_callback)

    def save_button_callback():
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
            )

            for ext in (".comm", ".incomm"):
                fname = temp_file + ext
                if os.path.isfile(fname):
                    with open(fname) as f:
                        cont = f.read()
                else:
                    cont = ""
                js_data[ext].data.update(cont=[cont], ext=[ext])

    save_button = Button(label="Download file", button_type="success", default_size=220)
    save_button.on_click(save_button_callback)
    save_button.js_on_click(CustomJS(args={"js_data": js_data[".comm"]}, code=javaScript))
    save_button.js_on_click(CustomJS(args={"js_data": js_data[".incomm"]}, code=javaScript))

    fitpeak_controls = row(
        column(fitparams_add_dropdown, fitparams_select, fitparams_remove_button),
        fitparams_table,
        Spacer(width=20),
        column(
            row(fit_from_spinner, fit_to_spinner),
            row(bin_size_spinner, column(Spacer(height=19), lorentz_toggle)),
            row(area_method_radiobutton),
            row(fit_button, fit_all_button),
        ),
    )

    scan_layout = column(scan_table, row(param_select))

    export_layout = column(preview_output_textinput, row(preview_output_button, save_button))

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
        row(scan_layout, plots, Spacer(width=30), export_layout),
        row(fitpeak_controls, fit_output_textinput),
    )

    return Panel(child=tab_layout, title="param study")
