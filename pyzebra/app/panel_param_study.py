import itertools
import os
import tempfile

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CellEditor,
    CheckboxEditor,
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    LinearColorMapper,
    NumberEditor,
    Panel,
    Range1d,
    Select,
    Spacer,
    Span,
    TableColumn,
    Tabs,
    TextAreaInput,
    Whisker,
)
from bokeh.palettes import Category10, Plasma256
from bokeh.plotting import figure
from scipy import interpolate

import pyzebra
from pyzebra import app


def color_palette(n_colors):
    palette = itertools.cycle(Category10[10])
    return list(itertools.islice(palette, n_colors))


def create():
    dataset = []
    app_dlfiles = app.DownloadFiles(n_files=1)

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
            file=file_list, scan=scan_list, param=param, fit=[0] * len(scan_list), export=export
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        scan_motor_select.options = dataset[0]["scan_motors"]
        scan_motor_select.value = dataset[0]["scan_motor"]

        merge_options = [(str(i), f"{i} ({idx})") for i, idx in enumerate(scan_list)]
        merge_from_select.options = merge_options
        merge_from_select.value = merge_options[0][0]

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
        scatter_source.data.update(x=x, y=y, y_upper=y + y_err, y_lower=y - y_err)

        fit = scan.get("fit")
        if fit is not None:
            x_fit = np.linspace(x[0], x[-1], 100)
            fit_source.data.update(x=x_fit, y=fit.eval(x=x_fit))

            x_bkg = []
            y_bkg = []
            xs_peak = []
            ys_peak = []
            comps = fit.eval_components(x=x_fit)
            for i, model in enumerate(app_fitctrl.params):
                if "linear" in model:
                    x_bkg = x_fit
                    y_bkg = comps[f"f{i}_"]

                elif any(val in model for val in ("gaussian", "voigt", "pvoigt")):
                    xs_peak.append(x_fit)
                    ys_peak.append(comps[f"f{i}_"])

            bkg_source.data.update(x=x_bkg, y=y_bkg)
            peak_source.data.update(xs=xs_peak, ys=ys_peak)

        else:
            fit_source.data.update(x=[], y=[])
            bkg_source.data.update(x=[], y=[])
            peak_source.data.update(xs=[], ys=[])

        app_fitctrl.update_result_textarea(scan)

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

        ov_mline_source.data.update(xs=xs, ys=ys, param=param, color=color_palette(len(xs)))

        ov_param_scatter_source.data.update(x=x, y=y)

        if y:
            x1, x2 = min(x), max(x)
            y1, y2 = min(y), max(y)
            grid_x, grid_y = np.meshgrid(
                np.linspace(x1, x2, ov_param_plot.inner_width),
                np.linspace(y1, y2, ov_param_plot.inner_height),
            )
            image = interpolate.griddata((x, y), par, (grid_x, grid_y))
            ov_param_image_source.data.update(
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
            ov_param_image_source.data.update(image=[], x=[], y=[], dw=[], dh=[])

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

        param_scatter_source.data.update(x=x, y=y, y_lower=y_lower, y_upper=y_upper)

    def _monitor_change():
        _update_single_scan_plot()
        _update_overview()

    app_inputctrl = app.InputControls(
        dataset, app_dlfiles, on_file_open=_init_datatable, on_monitor_change=_monitor_change
    )

    # Main plot
    plot = figure(
        x_axis_label="Scan motor",
        y_axis_label="Counts",
        plot_height=450,
        plot_width=700,
        tools="pan,wheel_zoom,reset",
    )

    scatter_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot.circle(
        source=scatter_source, line_color="steelblue", fill_color="steelblue", legend_label="data"
    )
    plot.add_layout(Whisker(source=scatter_source, base="x", upper="y_upper", lower="y_lower"))

    fit_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(source=fit_source, legend_label="best fit")

    bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(source=bkg_source, line_color="green", line_dash="dashed", legend_label="linear")

    peak_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot.multi_line(source=peak_source, line_color="red", line_dash="dashed", legend_label="peak")

    fit_from_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_from_span)

    fit_to_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_to_span)

    plot.y_range.only_visible = True
    plot.toolbar.logo = None
    plot.legend.click_policy = "hide"

    # Overview multilines plot
    ov_plot = figure(
        x_axis_label="Scan motor",
        y_axis_label="Counts",
        plot_height=450,
        plot_width=700,
        tools="pan,wheel_zoom,reset",
    )

    ov_mline_source = ColumnDataSource(dict(xs=[], ys=[], param=[], color=[]))
    ov_plot.multi_line(source=ov_mline_source, line_color="color")

    ov_plot.add_tools(HoverTool(tooltips=[("param", "@param")]))

    ov_plot.toolbar.logo = None

    # Overview params plot
    ov_param_plot = figure(
        x_axis_label="Scan motor",
        y_axis_label="Param",
        x_range=Range1d(),
        y_range=Range1d(),
        plot_height=450,
        plot_width=700,
        tools="pan,wheel_zoom,reset",
    )

    color_mapper = LinearColorMapper(palette=Plasma256)
    ov_param_image_source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    ov_param_plot.image(source=ov_param_image_source, color_mapper=color_mapper)

    ov_param_scatter_source = ColumnDataSource(dict(x=[], y=[]))
    ov_param_plot.dot(source=ov_param_scatter_source, size=15, color="black")

    ov_param_plot.toolbar.logo = None

    # Parameter plot
    param_plot = figure(
        x_axis_label="Parameter",
        y_axis_label="Fit parameter",
        plot_height=400,
        plot_width=700,
        tools="pan,wheel_zoom,reset",
    )

    param_scatter_source = ColumnDataSource(dict(x=[], y=[], y_upper=[], y_lower=[]))
    param_plot.circle(source=param_scatter_source)
    param_plot.add_layout(
        Whisker(source=param_scatter_source, base="x", upper="y_upper", lower="y_lower")
    )

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

    app_fitctrl = app.FitControls()

    def fit_from_spinner_callback(_attr, _old, new):
        fit_from_span.location = new

    app_fitctrl.from_spinner.on_change("value", fit_from_spinner_callback)

    def fit_to_spinner_callback(_attr, _old, new):
        fit_to_span.location = new

    app_fitctrl.to_spinner.on_change("value", fit_to_spinner_callback)

    def proc_all_button_callback():
        app_fitctrl.fit_dataset(dataset)

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
        app_fitctrl.fit_scan(_get_selected_scan())

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

            app_dlfiles.set_contents(file_content)
            export_preview_textinput.value = exported_content

    area_method_div = Div(text="Intensity:", margin=(5, 5, 0, 5))
    fitpeak_controls = row(
        column(
            app_fitctrl.add_function_button,
            app_fitctrl.function_select,
            app_fitctrl.remove_function_button,
        ),
        app_fitctrl.params_table,
        Spacer(width=20),
        column(
            app_fitctrl.from_spinner,
            app_fitctrl.lorentz_checkbox,
            area_method_div,
            app_fitctrl.area_method_radiogroup,
        ),
        column(app_fitctrl.to_spinner, proc_button, proc_all_button),
    )

    scan_layout = column(
        scan_table,
        row(app_inputctrl.monitor_spinner, scan_motor_select, param_select),
        row(column(Spacer(height=19), row(restore_button, merge_button)), merge_from_select),
    )

    upload_div = Div(text="or upload new .ccl/.dat files:", margin=(5, 5, 0, 5))
    append_upload_div = Div(text="append extra files:", margin=(5, 5, 0, 5))
    import_layout = column(
        app_inputctrl.filelist_select,
        row(app_inputctrl.open_button, app_inputctrl.append_button),
        upload_div,
        app_inputctrl.upload_button,
        append_upload_div,
        app_inputctrl.append_upload_button,
    )

    export_layout = column(export_preview_textinput, row(app_dlfiles.button))

    tab_layout = column(
        row(import_layout, scan_layout, plots, Spacer(width=30), export_layout),
        row(fitpeak_controls, app_fitctrl.result_textarea),
    )

    return Panel(child=tab_layout, title="param study")
