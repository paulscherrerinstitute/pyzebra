import base64
import io
import os
import tempfile

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CellEditor,
    CheckboxEditor,
    ColumnDataSource,
    DataTable,
    Div,
    FileInput,
    MultiSelect,
    Panel,
    RadioGroup,
    Select,
    Spacer,
    Span,
    Spinner,
    TableColumn,
    TextAreaInput,
    Whisker,
)
from bokeh.plotting import figure

import pyzebra
from pyzebra import EXPORT_TARGETS, app


def create():
    doc = curdoc()
    dataset1 = []
    dataset2 = []
    app_dlfiles = app.DownloadFiles(n_files=2)

    def file_select_update_for_proposal():
        proposal_path = proposal_textinput.name
        if proposal_path:
            file_list = []
            for file in os.listdir(proposal_path):
                if file.endswith((".ccl")):
                    file_list.append((os.path.join(proposal_path, file), file))
            file_select.options = file_list
            file_open_button.disabled = False
        else:
            file_select.options = []
            file_open_button.disabled = True

    doc.add_periodic_callback(file_select_update_for_proposal, 5000)

    def proposal_textinput_callback(_attr, _old, _new):
        file_select_update_for_proposal()

    proposal_textinput = doc.proposal_textinput
    proposal_textinput.on_change("name", proposal_textinput_callback)

    def _init_datatable():
        # dataset2 should have the same metadata as dataset1
        scan_list = [s["idx"] for s in dataset1]
        hkl = [f'{s["h"]} {s["k"]} {s["l"]}' for s in dataset1]
        export = [s["export"] for s in dataset1]

        twotheta = [np.median(s["twotheta"]) if "twotheta" in s else None for s in dataset1]
        gamma = [np.median(s["gamma"]) if "gamma" in s else None for s in dataset1]
        omega = [np.median(s["omega"]) if "omega" in s else None for s in dataset1]
        chi = [np.median(s["chi"]) if "chi" in s else None for s in dataset1]
        phi = [np.median(s["phi"]) if "phi" in s else None for s in dataset1]
        nu = [np.median(s["nu"]) if "nu" in s else None for s in dataset1]

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

    file_select = MultiSelect(title="Select 2 .ccl files:", width=210, height=250)

    def file_open_button_callback():
        if len(file_select.value) != 2:
            print("WARNING: Select exactly 2 .ccl files.")
            return

        new_data1 = []
        new_data2 = []
        for ind, f_path in enumerate(file_select.value):
            with open(f_path) as file:
                f_name = os.path.basename(f_path)
                base, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    return

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)
            pyzebra.merge_duplicates(file_data)

            if ind == 0:
                app_dlfiles.set_names([base, base])
                new_data1 = file_data
            else:  # ind = 1
                new_data2 = file_data

        # ignore extra scans at the end of the longest of the two files
        min_len = min(len(new_data1), len(new_data2))
        new_data1 = new_data1[:min_len]
        new_data2 = new_data2[:min_len]

        nonlocal dataset1, dataset2
        dataset1 = new_data1
        dataset2 = new_data2
        _init_datatable()

    file_open_button = Button(label="Open New", width=100, disabled=True)
    file_open_button.on_click(file_open_button_callback)

    def upload_button_callback(_attr, _old, _new):
        if len(upload_button.filename) != 2:
            print("WARNING: Upload exactly 2 .ccl files.")
            return

        new_data1 = []
        new_data2 = []
        for ind, (f_str, f_name) in enumerate(zip(upload_button.value, upload_button.filename)):
            with io.StringIO(base64.b64decode(f_str).decode()) as file:
                base, ext = os.path.splitext(f_name)
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {f_name}")
                    return

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)
            pyzebra.merge_duplicates(file_data)

            if ind == 0:
                app_dlfiles.set_names([base, base])
                new_data1 = file_data
            else:  # ind = 1
                new_data2 = file_data

        # ignore extra scans at the end of the longest of the two files
        min_len = min(len(new_data1), len(new_data2))
        new_data1 = new_data1[:min_len]
        new_data2 = new_data2[:min_len]

        nonlocal dataset1, dataset2
        dataset1 = new_data1
        dataset2 = new_data2
        _init_datatable()

    upload_div = Div(text="or upload 2 .ccl files:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".ccl", multiple=True, width=200)
    # for on_change("value", ...) or on_change("filename", ...),
    # see https://github.com/bokeh/bokeh/issues/11461
    upload_button.on_change("filename", upload_button_callback)

    def monitor_spinner_callback(_attr, old, new):
        if dataset1 and dataset2:
            pyzebra.normalize_dataset(dataset1, new)
            pyzebra.normalize_dataset(dataset2, new)
            _update_plot()

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in dataset1]
        export = [scan["export"] for scan in dataset1]
        scan_table_source.data.update(fit=fit_ok, export=export)

    def _update_plot():
        scatter_sources = [scatter1_source, scatter2_source]
        fit_sources = [fit1_source, fit2_source]
        bkg_sources = [bkg1_source, bkg2_source]
        peak_sources = [peak1_source, peak2_source]
        fit_output = ""

        for ind, scan in enumerate(_get_selected_scan()):
            scatter_source = scatter_sources[ind]
            fit_source = fit_sources[ind]
            bkg_source = bkg_sources[ind]
            peak_source = peak_sources[ind]
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
                if fit_output:
                    fit_output = fit_output + "\n\n"
                fit_output = fit_output + fit.fit_report()

            else:
                fit_source.data.update(x=[], y=[])
                bkg_source.data.update(x=[], y=[])
                peak_source.data.update(xs=[], ys=[])

        app_fitctrl.result_textarea.value = fit_output

    # Main plot
    plot = figure(
        x_axis_label="Scan motor",
        y_axis_label="Counts",
        plot_height=470,
        plot_width=700,
        tools="pan,wheel_zoom,reset",
    )

    scatter1_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot.circle(
        source=scatter1_source,
        line_color="steelblue",
        fill_color="steelblue",
        legend_label="data 1",
    )
    plot.add_layout(Whisker(source=scatter1_source, base="x", upper="y_upper", lower="y_lower"))

    scatter2_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot.circle(
        source=scatter2_source,
        line_color="firebrick",
        fill_color="firebrick",
        legend_label="data 2",
    )
    plot.add_layout(Whisker(source=scatter2_source, base="x", upper="y_upper", lower="y_lower"))

    fit1_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(source=fit1_source, legend_label="best fit 1")

    fit2_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(source=fit2_source, line_color="firebrick", legend_label="best fit 2")

    bkg1_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(
        source=bkg1_source, line_color="steelblue", line_dash="dashed", legend_label="linear 1"
    )

    bkg2_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.line(
        source=bkg2_source, line_color="firebrick", line_dash="dashed", legend_label="linear 2"
    )

    peak1_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot.multi_line(
        source=peak1_source, line_color="steelblue", line_dash="dashed", legend_label="peak 1"
    )

    peak2_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot.multi_line(
        source=peak2_source, line_color="firebrick", line_dash="dashed", legend_label="peak 2"
    )

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
        for scan1, scan2, export in zip(dataset1, dataset2, new["export"]):
            scan1["export"] = export
            scan2["export"] = export
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
        ind = scan_table_source.selected.indices[0]
        return dataset1[ind], dataset2[ind]

    merge_from_select = Select(title="scan:", width=145)

    def merge_button_callback():
        scan_into1, scan_into2 = _get_selected_scan()
        scan_from1 = dataset1[int(merge_from_select.value)]
        scan_from2 = dataset2[int(merge_from_select.value)]

        if scan_into1 is scan_from1:
            print("WARNING: Selected scans for merging are identical")
            return

        pyzebra.merge_scans(scan_into1, scan_from1)
        pyzebra.merge_scans(scan_into2, scan_from2)
        _update_table()
        _update_plot()

    merge_button = Button(label="Merge into current", width=145)
    merge_button.on_click(merge_button_callback)

    def restore_button_callback():
        scan1, scan2 = _get_selected_scan()
        pyzebra.restore_scan(scan1)
        pyzebra.restore_scan(scan2)
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
        app_fitctrl.fit_dataset(dataset1)
        app_fitctrl.fit_dataset(dataset2)

        _update_plot()
        _update_table()

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
        scan1, scan2 = _get_selected_scan()
        app_fitctrl.fit_scan(scan1)
        app_fitctrl.fit_scan(scan2)

        _update_plot()
        _update_table()

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    intensity_diff_div = Div(text="Intensity difference:", margin=(5, 5, 0, 5))
    intensity_diff_radiobutton = RadioGroup(
        labels=["file1 - file2", "file2 - file1"], active=0, width=145
    )

    export_preview_textinput = TextAreaInput(title="Export file(s) preview:", width=500, height=400)

    def _update_preview():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data1 = []
            export_data2 = []
            for scan1, scan2 in zip(dataset1, dataset2):
                if scan1["export"]:
                    export_data1.append(scan1)
                    export_data2.append(scan2)

            if intensity_diff_radiobutton.active:
                export_data1, export_data2 = export_data2, export_data1

            pyzebra.export_ccl_compare(
                export_data1,
                export_data2,
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
            intensity_diff_div,
            intensity_diff_radiobutton,
        ),
        column(app_fitctrl.to_spinner, proc_button, proc_all_button),
    )

    scan_layout = column(
        scan_table,
        row(monitor_spinner, column(Spacer(height=19), restore_button)),
        row(column(Spacer(height=19), merge_button), merge_from_select),
    )

    import_layout = column(file_select, file_open_button, upload_div, upload_button)

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

    return Panel(child=tab_layout, title="ccl compare")
