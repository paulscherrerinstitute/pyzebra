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
    Panel,
    Select,
    Spacer,
    Span,
    TableColumn,
    TextAreaInput,
    Whisker,
)
from bokeh.plotting import figure

import pyzebra
from pyzebra import EXPORT_TARGETS, app


def create():
    dataset = []
    app_dlfiles = app.DownloadFiles(n_files=2)

    def _init_datatable():
        scan_list = [s["idx"] for s in dataset]
        hkl = [f'{s["h"]} {s["k"]} {s["l"]}' for s in dataset]
        export = [s["export"] for s in dataset]

        twotheta = [np.median(s["twotheta"]) if "twotheta" in s else None for s in dataset]
        gamma = [np.median(s["gamma"]) if "gamma" in s else None for s in dataset]
        omega = [np.median(s["omega"]) if "omega" in s else None for s in dataset]
        chi = [np.median(s["chi"]) if "chi" in s else None for s in dataset]
        phi = [np.median(s["phi"]) if "phi" in s else None for s in dataset]
        nu = [np.median(s["nu"]) if "nu" in s else None for s in dataset]

        scan_table_source.data.update(
            scan=scan_list,
            hkl=hkl,
            fit=[0] * len(scan_list),
            export=export,
            twotheta=twotheta,
            gamma=gamma,
            omega=omega,
            chi=chi,
            phi=phi,
            nu=nu,
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        merge_options = [(str(i), f"{i} ({idx})") for i, idx in enumerate(scan_list)]
        merge_from_select.options = merge_options
        merge_from_select.value = merge_options[0][0]

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in dataset]
        export = [scan["export"] for scan in dataset]
        scan_table_source.data.update(fit=fit_ok, export=export)

    def _update_plot():
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

    app_inputctrl = app.InputControls(
        dataset, app_dlfiles, on_file_open=_init_datatable, on_monitor_change=_update_plot
    )

    # Main plot
    plot = figure(
        x_axis_label="Scan motor",
        y_axis_label="Counts",
        height=470,
        width=700,
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

    def scan_table_source_callback(_attr, _old, new):
        # unfortunately, we don't know if the change comes from data update or user input
        # also `old` and `new` are the same for non-scalars
        for scan, export in zip(dataset, new["export"]):
            scan["export"] = export
        _update_preview()

    scan_table_source = ColumnDataSource(
        dict(
            scan=[],
            hkl=[],
            fit=[],
            export=[],
            twotheta=[],
            gamma=[],
            omega=[],
            chi=[],
            phi=[],
            nu=[],
        )
    )
    scan_table_source.on_change("data", scan_table_source_callback)
    scan_table_source.selected.on_change("indices", scan_table_select_callback)

    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="scan", title="Scan", editor=CellEditor(), width=50),
            TableColumn(field="hkl", title="hkl", editor=CellEditor(), width=100),
            TableColumn(field="fit", title="Fit", editor=CellEditor(), width=50),
            TableColumn(field="export", title="Export", editor=CheckboxEditor(), width=50),
            TableColumn(field="twotheta", title="2theta", editor=CellEditor(), width=50),
            TableColumn(field="gamma", title="gamma", editor=CellEditor(), width=50),
            TableColumn(field="omega", title="omega", editor=CellEditor(), width=50),
            TableColumn(field="chi", title="chi", editor=CellEditor(), width=50),
            TableColumn(field="phi", title="phi", editor=CellEditor(), width=50),
            TableColumn(field="nu", title="nu", editor=CellEditor(), width=50),
        ],
        width=310,  # +60 because of the index column, but excluding twotheta onwards
        height=350,
        autosize_mode="none",
        editable=True,
    )

    def _get_selected_scan():
        return dataset[scan_table_source.selected.indices[0]]

    merge_from_select = Select(title="scan:", width=145)

    def merge_button_callback():
        scan_into = _get_selected_scan()
        scan_from = dataset[int(merge_from_select.value)]

        if scan_into is scan_from:
            print("WARNING: Selected scans for merging are identical")
            return

        pyzebra.merge_scans(scan_into, scan_from)
        _update_table()
        _update_plot()

    merge_button = Button(label="Merge into current", width=145)
    merge_button.on_click(merge_button_callback)

    def restore_button_callback():
        pyzebra.restore_scan(_get_selected_scan())
        _update_table()
        _update_plot()

    restore_button = Button(label="Restore scan", width=145)
    restore_button.on_click(restore_button_callback)

    app_fitctrl = app.FitControls()

    def fit_from_spinner_callback(_attr, _old, new):
        fit_from_span.location = new

    app_fitctrl.from_spinner.on_change("value", fit_from_spinner_callback)

    def fit_to_spinner_callback(_attr, _old, new):
        fit_to_span.location = new

    app_fitctrl.to_spinner.on_change("value", fit_to_spinner_callback)

    def proc_all_button_callback():
        app_fitctrl.fit_dataset(dataset)

        _update_plot()
        _update_table()

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
        app_fitctrl.fit_scan(_get_selected_scan())

        _update_plot()
        _update_table()

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    export_preview_textinput = TextAreaInput(title="Export file(s) preview:", width=500, height=400)

    def _update_preview():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = []
            for scan in dataset:
                if scan["export"]:
                    export_data.append(scan)

            pyzebra.export_1D(
                export_data,
                temp_file,
                export_target_select.value,
                hkl_precision=int(hkl_precision_select.value),
            )

            exported_content = ""
            file_content = []
            for ext in EXPORT_TARGETS[export_target_select.value]:
                fname = temp_file + ext
                if os.path.isfile(fname):
                    with open(fname) as f:
                        content = f.read()
                        exported_content += f"{ext} file:\n" + content
                else:
                    content = ""
                file_content.append(content)

            app_dlfiles.set_contents(file_content)
            export_preview_textinput.value = exported_content

    def export_target_select_callback(_attr, _old, new):
        app_dlfiles.set_extensions(EXPORT_TARGETS[new])
        _update_preview()

    export_target_select = Select(
        title="Export target:", options=list(EXPORT_TARGETS.keys()), value="fullprof", width=80
    )
    export_target_select.on_change("value", export_target_select_callback)
    app_dlfiles.set_extensions(EXPORT_TARGETS[export_target_select.value])

    def hkl_precision_select_callback(_attr, _old, _new):
        _update_preview()

    hkl_precision_select = Select(
        title="hkl precision:", options=["2", "3", "4"], value="2", width=80
    )
    hkl_precision_select.on_change("value", hkl_precision_select_callback)

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
        row(app_inputctrl.monitor_spinner, column(Spacer(height=19), restore_button)),
        row(column(Spacer(height=19), merge_button), merge_from_select),
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

    export_layout = column(
        export_preview_textinput,
        row(
            export_target_select,
            hkl_precision_select,
            column(Spacer(height=19), row(app_dlfiles.button)),
        ),
    )

    tab_layout = column(
        row(import_layout, scan_layout, plot, Spacer(width=30), export_layout),
        row(fitpeak_controls, app_fitctrl.result_textarea),
    )

    return Panel(child=tab_layout, title="ccl integrate")
