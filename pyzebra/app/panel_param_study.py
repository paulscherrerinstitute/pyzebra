import base64
import io
import itertools
import os
import tempfile
import types

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    Button,
    CellEditor,
    CheckboxEditor,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    DataTable,
    Div,
    Dropdown,
    FileInput,
    Grid,
    HoverTool,
    Image,
    Legend,
    Line,
    LinearAxis,
    LinearColorMapper,
    MultiLine,
    MultiSelect,
    NumberEditor,
    Panel,
    PanTool,
    Plot,
    RadioGroup,
    Range1d,
    ResetTool,
    Scatter,
    Select,
    Spacer,
    Span,
    Spinner,
    TableColumn,
    Tabs,
    TextAreaInput,
    WheelZoomTool,
    Whisker,
)
from bokeh.palettes import Category10, Plasma256
from scipy import interpolate

import pyzebra
from pyzebra.ccl_process import AREA_METHODS

javaScript = """
let j = 0;
for (let i = 0; i < js_data.data['fname'].length; i++) {
    if (js_data.data['content'][i] === "") continue;

    setTimeout(function() {
        const blob = new Blob([js_data.data['content'][i]], {type: 'text/plain'})
        const link = document.createElement('a');
        document.body.appendChild(link);
        const url = window.URL.createObjectURL(blob);
        link.href = url;
        link.download = js_data.data['fname'][i] + js_data.data['ext'][i];
        link.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(link);
    }, 100 * j)

    j++;
}
"""


def color_palette(n_colors):
    palette = itertools.cycle(Category10[10])
    return list(itertools.islice(palette, n_colors))


def create():
    doc = curdoc()
    dataset = []
    fit_params = {}
    js_data = ColumnDataSource(data=dict(content=[""], fname=[""], ext=[""]))

    def file_select_update_for_proposal():
        proposal_path = proposal_textinput.name
        if proposal_path:
            file_list = []
            for file in os.listdir(proposal_path):
                if file.endswith((".ccl", ".dat")):
                    file_list.append((os.path.join(proposal_path, file), file))
            file_select.options = file_list
            file_open_button.disabled = False
            file_append_button.disabled = False
        else:
            file_select.options = []
            file_open_button.disabled = True
            file_append_button.disabled = True

    doc.add_periodic_callback(file_select_update_for_proposal, 5000)

    def proposal_textinput_callback(_attr, _old, _new):
        file_select_update_for_proposal()

    proposal_textinput = doc.proposal_textinput
    proposal_textinput.on_change("name", proposal_textinput_callback)

    def _init_datatable():
        scan_list = [s["idx"] for s in dataset]
        export = [s["export"] for s in dataset]
        if param_select.value == "user defined":
            param = [None] * len(dataset)
        else:
            param = [scan[param_select.value] for scan in dataset]

        file_list = []
        for scan in dataset:
            file_list.append(os.path.basename(scan["original_filename"]))

        scan_table_source.data.update(
            file=file_list, scan=scan_list, param=param, fit=[0] * len(scan_list), export=export,
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        scan_motor_select.options = dataset[0]["scan_motors"]
        scan_motor_select.value = dataset[0]["scan_motor"]

        merge_options = [(str(i), f"{i} ({idx})") for i, idx in enumerate(scan_list)]
        merge_from_select.options = merge_options
        merge_from_select.value = merge_options[0][0]

    file_select = MultiSelect(title="Available .ccl/.dat files:", width=210, height=250)

    def file_open_button_callback():
        nonlocal dataset
        new_data = []
        for f_path in file_select.value:
            with open(f_path) as file:
                f_name = os.path.basename(f_path)
                base, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)

            if not new_data:  # first file
                new_data = file_data
                pyzebra.merge_duplicates(new_data)
                js_data.data.update(fname=[base])
            else:
                pyzebra.merge_datasets(new_data, file_data)

        if new_data:
            dataset = new_data
            _init_datatable()
            append_upload_button.disabled = False

    file_open_button = Button(label="Open New", width=100, disabled=True)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        file_data = []
        for f_path in file_select.value:
            with open(f_path) as file:
                f_name = os.path.basename(f_path)
                _, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)
            pyzebra.merge_datasets(dataset, file_data)

        if file_data:
            _init_datatable()

    file_append_button = Button(label="Append", width=100, disabled=True)
    file_append_button.on_click(file_append_button_callback)

    def upload_button_callback(_attr, _old, _new):
        nonlocal dataset
        new_data = []
        for f_str, f_name in zip(upload_button.value, upload_button.filename):
            with io.StringIO(base64.b64decode(f_str).decode()) as file:
                base, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)

            if not new_data:  # first file
                new_data = file_data
                pyzebra.merge_duplicates(new_data)
                js_data.data.update(fname=[base])
            else:
                pyzebra.merge_datasets(new_data, file_data)

        if new_data:
            dataset = new_data
            _init_datatable()
            append_upload_button.disabled = False

    upload_div = Div(text="or upload new .ccl/.dat files:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".ccl,.dat", multiple=True, width=200)
    # for on_change("value", ...) or on_change("filename", ...),
    # see https://github.com/bokeh/bokeh/issues/11461
    upload_button.on_change("filename", upload_button_callback)

    def append_upload_button_callback(_attr, _old, _new):
        file_data = []
        for f_str, f_name in zip(append_upload_button.value, append_upload_button.filename):
            with io.StringIO(base64.b64decode(f_str).decode()) as file:
                _, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)
            pyzebra.merge_datasets(dataset, file_data)

        if file_data:
            _init_datatable()

    append_upload_div = Div(text="append extra files:", margin=(5, 5, 0, 5))
    append_upload_button = FileInput(accept=".ccl,.dat", multiple=True, width=200, disabled=True)
    # for on_change("value", ...) or on_change("filename", ...),
    # see https://github.com/bokeh/bokeh/issues/11461
    append_upload_button.on_change("filename", append_upload_button_callback)

    def monitor_spinner_callback(_attr, _old, new):
        if dataset:
            pyzebra.normalize_dataset(dataset, new)
            _update_single_scan_plot()
            _update_overview()

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    def scan_motor_select_callback(_attr, _old, new):
        if dataset:
            for scan in dataset:
                scan["scan_motor"] = new
            _update_single_scan_plot()
            _update_overview()

    scan_motor_select = Select(title="Scan motor:", options=[], width=145)
    scan_motor_select.on_change("value", scan_motor_select_callback)

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in dataset]
        export = [scan["export"] for scan in dataset]
        if param_select.value == "user defined":
            param = [None] * len(dataset)
        else:
            param = [scan[param_select.value] for scan in dataset]

        scan_table_source.data.update(fit=fit_ok, export=export, param=param)

    def _update_single_scan_plot():
        scan = _get_selected_scan()
        scan_motor = scan["scan_motor"]

        y = scan["counts"]
        y_err = scan["counts_err"]
        x = scan[scan_motor]

        plot.axis[0].axis_label = scan_motor
        plot_scatter_source.data.update(x=x, y=y, y_upper=y + y_err, y_lower=y - y_err)

        fit = scan.get("fit")
        if fit is not None:
            x_fit = np.linspace(x[0], x[-1], 100)
            plot_fit_source.data.update(x=x_fit, y=fit.eval(x=x_fit))

            x_bkg = []
            y_bkg = []
            xs_peak = []
            ys_peak = []
            comps = fit.eval_components(x=x_fit)
            for i, model in enumerate(fit_params):
                if "linear" in model:
                    x_bkg = x_fit
                    y_bkg = comps[f"f{i}_"]

                elif any(val in model for val in ("gaussian", "voigt", "pvoigt")):
                    xs_peak.append(x_fit)
                    ys_peak.append(comps[f"f{i}_"])

            plot_bkg_source.data.update(x=x_bkg, y=y_bkg)
            plot_peak_source.data.update(xs=xs_peak, ys=ys_peak)

            fit_output_textinput.value = fit.fit_report()

        else:
            plot_fit_source.data.update(x=[], y=[])
            plot_bkg_source.data.update(x=[], y=[])
            plot_peak_source.data.update(xs=[], ys=[])
            fit_output_textinput.value = ""

    def _update_overview():
        xs = []
        ys = []
        param = []
        x = []
        y = []
        par = []
        for s, p in enumerate(scan_table_source.data["param"]):
            if p is not None:
                scan = dataset[s]
                scan_motor = scan["scan_motor"]
                xs.append(scan[scan_motor])
                x.extend(scan[scan_motor])
                ys.append(scan["counts"])
                y.extend([float(p)] * len(scan[scan_motor]))
                param.append(float(p))
                par.extend(scan["counts"])

        if dataset:
            scan_motor = dataset[0]["scan_motor"]
            ov_plot.axis[0].axis_label = scan_motor
            ov_param_plot.axis[0].axis_label = scan_motor

        ov_plot_mline_source.data.update(xs=xs, ys=ys, param=param, color=color_palette(len(xs)))

        ov_param_plot_scatter_source.data.update(x=x, y=y)

        if y:
            x1, x2 = min(x), max(x)
            y1, y2 = min(y), max(y)
            grid_x, grid_y = np.meshgrid(
                np.linspace(x1, x2, ov_param_plot.inner_width),
                np.linspace(y1, y2, ov_param_plot.inner_height),
            )
            image = interpolate.griddata((x, y), par, (grid_x, grid_y))
            ov_param_plot_image_source.data.update(
                image=[image], x=[x1], y=[y1], dw=[x2 - x1], dh=[y2 - y1]
            )

            x_range = ov_param_plot.x_range
            x_range.start, x_range.end = x1, x2
            x_range.reset_start, x_range.reset_end = x1, x2
            x_range.bounds = (x1, x2)

            y_range = ov_param_plot.y_range
            y_range.start, y_range.end = y1, y2
            y_range.reset_start, y_range.reset_end = y1, y2
            y_range.bounds = (y1, y2)

        else:
            ov_param_plot_image_source.data.update(image=[], x=[], y=[], dw=[], dh=[])

    def _update_param_plot():
        x = []
        y = []
        y_lower = []
        y_upper = []
        fit_param = fit_param_select.value
        for s, p in zip(dataset, scan_table_source.data["param"]):
            if "fit" in s and fit_param:
                x.append(p)
                param_fit_val = s["fit"].params[fit_param].value
                param_fit_std = s["fit"].params[fit_param].stderr
                if param_fit_std is None:
                    param_fit_std = 0
                y.append(param_fit_val)
                y_lower.append(param_fit_val - param_fit_std)
                y_upper.append(param_fit_val + param_fit_std)

        param_plot_scatter_source.data.update(x=x, y=y, y_lower=y_lower, y_upper=y_upper)

    # Main plot
    plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(only_visible=True),
        plot_height=450,
        plot_width=700,
    )

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_scatter_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot_scatter = plot.add_glyph(
        plot_scatter_source, Scatter(x="x", y="y", line_color="steelblue", fill_color="steelblue")
    )
    plot.add_layout(Whisker(source=plot_scatter_source, base="x", upper="y_upper", lower="y_lower"))

    plot_fit_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_fit = plot.add_glyph(plot_fit_source, Line(x="x", y="y"))

    plot_bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_bkg = plot.add_glyph(
        plot_bkg_source, Line(x="x", y="y", line_color="green", line_dash="dashed")
    )

    plot_peak_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot_peak = plot.add_glyph(
        plot_peak_source, MultiLine(xs="xs", ys="ys", line_color="red", line_dash="dashed")
    )

    fit_from_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_from_span)

    fit_to_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_to_span)

    plot.add_layout(
        Legend(
            items=[
                ("data", [plot_scatter]),
                ("best fit", [plot_fit]),
                ("peak", [plot_peak]),
                ("linear", [plot_bkg]),
            ],
            location="top_left",
            click_policy="hide",
        )
    )

    plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    plot.toolbar.logo = None

    # Overview multilines plot
    ov_plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=450, plot_width=700)

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
    ov_param_plot = Plot(x_range=Range1d(), y_range=Range1d(), plot_height=450, plot_width=700)

    ov_param_plot.add_layout(LinearAxis(axis_label="Param"), place="left")
    ov_param_plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    ov_param_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    ov_param_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    color_mapper = LinearColorMapper(palette=Plasma256)
    ov_param_plot_image_source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    ov_param_plot.add_glyph(
        ov_param_plot_image_source,
        Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper),
    )

    ov_param_plot_scatter_source = ColumnDataSource(dict(x=[], y=[]))
    ov_param_plot.add_glyph(
        ov_param_plot_scatter_source, Scatter(x="x", y="y", marker="dot", size=15),
    )

    ov_param_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    ov_param_plot.toolbar.logo = None

    # Parameter plot
    param_plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=400, plot_width=700)

    param_plot.add_layout(LinearAxis(axis_label="Fit parameter"), place="left")
    param_plot.add_layout(LinearAxis(axis_label="Parameter"), place="below")

    param_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    param_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    param_plot_scatter_source = ColumnDataSource(dict(x=[], y=[], y_upper=[], y_lower=[]))
    param_plot.add_glyph(param_plot_scatter_source, Scatter(x="x", y="y"))
    param_plot.add_layout(
        Whisker(source=param_plot_scatter_source, base="x", upper="y_upper", lower="y_lower")
    )

    param_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    param_plot.toolbar.logo = None

    def fit_param_select_callback(_attr, _old, _new):
        _update_param_plot()

    fit_param_select = Select(title="Fit parameter", options=[], width=145)
    fit_param_select.on_change("value", fit_param_select_callback)

    # Plot tabs
    plots = Tabs(
        tabs=[
            Panel(child=plot, title="single scan"),
            Panel(child=ov_plot, title="overview"),
            Panel(child=ov_param_plot, title="overview map"),
            Panel(child=column(param_plot, row(fit_param_select)), title="parameter plot"),
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

        _update_single_scan_plot()

    def scan_table_source_callback(_attr, _old, new):
        # unfortunately, we don't know if the change comes from data update or user input
        # also `old` and `new` are the same for non-scalars
        for scan, export in zip(dataset, new["export"]):
            scan["export"] = export
        _update_overview()
        _update_param_plot()
        _update_preview()

    scan_table_source = ColumnDataSource(dict(file=[], scan=[], param=[], fit=[], export=[]))
    scan_table_source.on_change("data", scan_table_source_callback)
    scan_table_source.selected.on_change("indices", scan_table_select_callback)

    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="file", title="file", editor=CellEditor(), width=150),
            TableColumn(field="scan", title="scan", editor=CellEditor(), width=50),
            TableColumn(field="param", title="param", editor=NumberEditor(), width=50),
            TableColumn(field="fit", title="Fit", editor=CellEditor(), width=50),
            TableColumn(field="export", title="Export", editor=CheckboxEditor(), width=50),
        ],
        width=410,  # +60 because of the index column
        height=350,
        editable=True,
        autosize_mode="none",
    )

    merge_from_select = Select(title="scan:", width=145)

    def merge_button_callback():
        scan_into = _get_selected_scan()
        scan_from = dataset[int(merge_from_select.value)]

        if scan_into is scan_from:
            print("WARNING: Selected scans for merging are identical")
            return

        pyzebra.merge_scans(scan_into, scan_from)
        _update_table()
        _update_single_scan_plot()
        _update_overview()

    merge_button = Button(label="Merge into current", width=145)
    merge_button.on_click(merge_button_callback)

    def restore_button_callback():
        pyzebra.restore_scan(_get_selected_scan())
        _update_table()
        _update_single_scan_plot()
        _update_overview()

    restore_button = Button(label="Restore scan", width=145)
    restore_button.on_click(restore_button_callback)

    def _get_selected_scan():
        return dataset[scan_table_source.selected.indices[0]]

    def param_select_callback(_attr, _old, _new):
        _update_table()

    param_select = Select(
        title="Parameter:",
        options=["user defined", "temp", "mf", "h", "k", "l"],
        value="user defined",
        width=145,
    )
    param_select.on_change("value", param_select_callback)

    def fit_from_spinner_callback(_attr, _old, new):
        fit_from_span.location = new

    fit_from_spinner = Spinner(title="Fit from:", width=145)
    fit_from_spinner.on_change("value", fit_from_spinner_callback)

    def fit_to_spinner_callback(_attr, _old, new):
        fit_to_span.location = new

    fit_to_spinner = Spinner(title="to:", width=145)
    fit_to_spinner.on_change("value", fit_to_spinner_callback)

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
        width=145,
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

    fitparams_select = MultiSelect(options=[], height=120, width=145)
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

    fitparams_remove_button = Button(label="Remove fit function", width=145)
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

        if function == "linear":
            fitparams["value"] = [0, 1]
            fitparams["vary"] = [False, True]
            fitparams["min"] = [None, 0]

        elif function == "gaussian":
            fitparams["min"] = [0, None, None]

        return fitparams

    fitparams_table_source = ColumnDataSource(dict(param=[], value=[], vary=[], min=[], max=[]))
    fitparams_table = DataTable(
        source=fitparams_table_source,
        columns=[
            TableColumn(field="param", title="Parameter", editor=CellEditor()),
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

    def proc_all_button_callback():
        for scan in dataset:
            if scan["export"]:
                pyzebra.fit_scan(
                    scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
                )
                pyzebra.get_area(
                    scan,
                    area_method=AREA_METHODS[area_method_radiobutton.active],
                    lorentz=lorentz_checkbox.active,
                )

        _update_single_scan_plot()
        _update_overview()
        _update_table()

        for scan in dataset:
            if "fit" in scan:
                options = list(scan["fit"].params.keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
        scan = _get_selected_scan()
        pyzebra.fit_scan(
            scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
        )
        pyzebra.get_area(
            scan,
            area_method=AREA_METHODS[area_method_radiobutton.active],
            lorentz=lorentz_checkbox.active,
        )

        _update_single_scan_plot()
        _update_overview()
        _update_table()

        for scan in dataset:
            if "fit" in scan:
                options = list(scan["fit"].params.keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    area_method_div = Div(text="Intensity:", margin=(5, 5, 0, 5))
    area_method_radiobutton = RadioGroup(labels=["Function", "Area"], active=0, width=145)

    lorentz_checkbox = CheckboxGroup(labels=["Lorentz Correction"], width=145, margin=(13, 5, 5, 5))

    export_preview_textinput = TextAreaInput(title="Export file preview:", width=450, height=400)

    def _update_preview():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = []
            param_data = []
            for scan, param in zip(dataset, scan_table_source.data["param"]):
                if scan["export"] and param:
                    export_data.append(scan)
                    param_data.append(param)

            pyzebra.export_param_study(export_data, param_data, temp_file)

            exported_content = ""
            file_content = []

            fname = temp_file
            if os.path.isfile(fname):
                with open(fname) as f:
                    content = f.read()
                    exported_content += content
            else:
                content = ""
            file_content.append(content)

            js_data.data.update(content=file_content)
            export_preview_textinput.value = exported_content

    save_button = Button(label="Download File", button_type="success", width=220)
    save_button.js_on_click(CustomJS(args={"js_data": js_data}, code=javaScript))

    fitpeak_controls = row(
        column(fitparams_add_dropdown, fitparams_select, fitparams_remove_button),
        fitparams_table,
        Spacer(width=20),
        column(fit_from_spinner, lorentz_checkbox, area_method_div, area_method_radiobutton),
        column(fit_to_spinner, proc_button, proc_all_button),
    )

    scan_layout = column(
        scan_table,
        row(monitor_spinner, scan_motor_select, param_select),
        row(column(Spacer(height=19), row(restore_button, merge_button)), merge_from_select),
    )

    import_layout = column(
        file_select,
        row(file_open_button, file_append_button),
        upload_div,
        upload_button,
        append_upload_div,
        append_upload_button,
    )

    export_layout = column(export_preview_textinput, row(save_button))

    tab_layout = column(
        row(import_layout, scan_layout, plots, Spacer(width=30), export_layout),
        row(fitpeak_controls, fit_output_textinput),
    )

    return Panel(child=tab_layout, title="param study")
