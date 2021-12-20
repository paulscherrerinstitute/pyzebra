import base64
import io
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
    Legend,
    Line,
    LinearAxis,
    MultiLine,
    MultiSelect,
    NumberEditor,
    Panel,
    PanTool,
    Plot,
    RadioGroup,
    ResetTool,
    Scatter,
    Select,
    Spacer,
    Span,
    Spinner,
    TableColumn,
    TextAreaInput,
    WheelZoomTool,
    Whisker,
)

import pyzebra
from pyzebra.ccl_io import EXPORT_TARGETS
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


def create():
    doc = curdoc()
    det_data1 = []
    det_data2 = []
    fit_params = {}
    js_data = ColumnDataSource(data=dict(content=["", ""], fname=["", ""], ext=["", ""]))

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
        # det_data2 should have the same metadata to det_data1
        scan_list = [s["idx"] for s in det_data1]
        hkl = [f'{s["h"]} {s["k"]} {s["l"]}' for s in det_data1]
        export = [s["export"] for s in det_data1]

        twotheta = [np.median(s["twotheta"]) if "twotheta" in s else None for s in det_data1]
        gamma = [np.median(s["gamma"]) if "gamma" in s else None for s in det_data1]
        omega = [np.median(s["omega"]) if "omega" in s else None for s in det_data1]
        chi = [np.median(s["chi"]) if "chi" in s else None for s in det_data1]
        phi = [np.median(s["phi"]) if "phi" in s else None for s in det_data1]
        nu = [np.median(s["nu"]) if "nu" in s else None for s in det_data1]

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
                js_data.data.update(fname=[base, base])
                new_data1 = file_data
            else:  # ind = 1
                new_data2 = file_data

        # ignore extra scans at the end of the longest of the two files
        min_len = min(len(new_data1), len(new_data2))
        new_data1 = new_data1[:min_len]
        new_data2 = new_data2[:min_len]

        nonlocal det_data1, det_data2
        det_data1 = new_data1
        det_data2 = new_data2
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
                js_data.data.update(fname=[base, base])
                new_data1 = file_data
            else:  # ind = 1
                new_data2 = file_data

        # ignore extra scans at the end of the longest of the two files
        min_len = min(len(new_data1), len(new_data2))
        new_data1 = new_data1[:min_len]
        new_data2 = new_data2[:min_len]

        nonlocal det_data1, det_data2
        det_data1 = new_data1
        det_data2 = new_data2
        _init_datatable()

    upload_div = Div(text="or upload 2 .ccl files:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".ccl", multiple=True, width=200)
    # for on_change("value", ...) or on_change("filename", ...),
    # see https://github.com/bokeh/bokeh/issues/11461
    upload_button.on_change("filename", upload_button_callback)

    def monitor_spinner_callback(_attr, old, new):
        if det_data1 and det_data2:
            pyzebra.normalize_dataset(det_data1, new)
            pyzebra.normalize_dataset(det_data2, new)
            _update_plot()

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    def _update_table():
        fit_ok = [(1 if "fit" in scan else 0) for scan in det_data1]
        export = [scan["export"] for scan in det_data1]
        scan_table_source.data.update(fit=fit_ok, export=export)

    def _update_plot():
        plot_scatter_source = [plot_scatter1_source, plot_scatter2_source]
        plot_fit_source = [plot_fit1_source, plot_fit2_source]
        plot_bkg_source = [plot_bkg1_source, plot_bkg2_source]
        plot_peak_source = [plot_peak1_source, plot_peak2_source]
        fit_output = ""

        for ind, scan in enumerate(_get_selected_scan()):
            scatter_source = plot_scatter_source[ind]
            fit_source = plot_fit_source[ind]
            bkg_source = plot_bkg_source[ind]
            peak_source = plot_peak_source[ind]
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
                for i, model in enumerate(fit_params):
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

        fit_output_textinput.value = fit_output

    # Main plot
    plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(only_visible=True),
        plot_height=470,
        plot_width=700,
    )

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(axis_label="Scan motor"), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_scatter1_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot_scatter1 = plot.add_glyph(
        plot_scatter1_source, Scatter(x="x", y="y", line_color="steelblue", fill_color="steelblue")
    )
    plot.add_layout(
        Whisker(source=plot_scatter1_source, base="x", upper="y_upper", lower="y_lower")
    )

    plot_scatter2_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot_scatter2 = plot.add_glyph(
        plot_scatter2_source, Scatter(x="x", y="y", line_color="firebrick", fill_color="firebrick")
    )
    plot.add_layout(
        Whisker(source=plot_scatter2_source, base="x", upper="y_upper", lower="y_lower")
    )

    plot_fit1_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_fit1 = plot.add_glyph(plot_fit1_source, Line(x="x", y="y"))

    plot_fit2_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_fit2 = plot.add_glyph(plot_fit2_source, Line(x="x", y="y"))

    plot_bkg1_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_bkg1 = plot.add_glyph(
        plot_bkg1_source, Line(x="x", y="y", line_color="steelblue", line_dash="dashed")
    )

    plot_bkg2_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot_bkg2 = plot.add_glyph(
        plot_bkg2_source, Line(x="x", y="y", line_color="firebrick", line_dash="dashed")
    )

    plot_peak1_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot_peak1 = plot.add_glyph(
        plot_peak1_source, MultiLine(xs="xs", ys="ys", line_color="steelblue", line_dash="dashed")
    )

    plot_peak2_source = ColumnDataSource(dict(xs=[[0]], ys=[[0]]))
    plot_peak2 = plot.add_glyph(
        plot_peak2_source, MultiLine(xs="xs", ys="ys", line_color="firebrick", line_dash="dashed")
    )

    fit_from_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_from_span)

    fit_to_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(fit_to_span)

    plot.add_layout(
        Legend(
            items=[
                ("data 1", [plot_scatter1]),
                ("data 2", [plot_scatter2]),
                ("best fit 1", [plot_fit1]),
                ("best fit 2", [plot_fit2]),
                ("peak 1", [plot_peak1]),
                ("peak 2", [plot_peak2]),
                ("linear 1", [plot_bkg1]),
                ("linear 2", [plot_bkg2]),
            ],
            location="top_left",
            click_policy="hide",
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

        _update_plot()

    def scan_table_source_callback(_attr, _old, new):
        # unfortunately, we don't know if the change comes from data update or user input
        # also `old` and `new` are the same for non-scalars
        for scan1, scan2, export in zip(det_data1, det_data2, new["export"]):
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
        return det_data1[ind], det_data2[ind]

    merge_from_select = Select(title="scan:", width=145)

    def merge_button_callback():
        scan_into1, scan_into2 = _get_selected_scan()
        scan_from1 = det_data1[int(merge_from_select.value)]
        scan_from2 = det_data2[int(merge_from_select.value)]

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
        for scan in [*det_data1, *det_data2]:
            if scan["export"]:
                pyzebra.fit_scan(
                    scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
                )
                pyzebra.get_area(
                    scan,
                    area_method=AREA_METHODS[area_method_radiobutton.active],
                    lorentz=lorentz_checkbox.active,
                )

        _update_plot()
        _update_table()

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
        for scan in _get_selected_scan():
            pyzebra.fit_scan(
                scan, fit_params, fit_from=fit_from_spinner.value, fit_to=fit_to_spinner.value
            )
            pyzebra.get_area(
                scan,
                area_method=AREA_METHODS[area_method_radiobutton.active],
                lorentz=lorentz_checkbox.active,
            )

        _update_plot()
        _update_table()

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    area_method_div = Div(text="Intensity:", margin=(5, 5, 0, 5))
    area_method_radiobutton = RadioGroup(labels=["Function", "Area"], active=0, width=145)

    intensity_diff_div = Div(text="Intensity difference:", margin=(5, 5, 0, 5))
    intensity_diff_radiobutton = RadioGroup(
        labels=["file1 - file2", "file2 - file1"], active=0, width=145
    )

    lorentz_checkbox = CheckboxGroup(labels=["Lorentz Correction"], width=145, margin=(13, 5, 5, 5))

    export_preview_textinput = TextAreaInput(title="Export file(s) preview:", width=500, height=400)

    def _update_preview():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data1 = []
            export_data2 = []
            for scan1, scan2 in zip(det_data1, det_data2):
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

            js_data.data.update(content=file_content)
            export_preview_textinput.value = exported_content

    def export_target_select_callback(_attr, _old, new):
        js_data.data.update(ext=EXPORT_TARGETS[new])
        _update_preview()

    export_target_select = Select(
        title="Export target:", options=list(EXPORT_TARGETS.keys()), value="fullprof", width=80
    )
    export_target_select.on_change("value", export_target_select_callback)
    js_data.data.update(ext=EXPORT_TARGETS[export_target_select.value])

    def hkl_precision_select_callback(_attr, _old, _new):
        _update_preview()

    hkl_precision_select = Select(
        title="hkl precision:", options=["2", "3", "4"], value="2", width=80
    )
    hkl_precision_select.on_change("value", hkl_precision_select_callback)

    save_button = Button(label="Download File(s)", button_type="success", width=200)
    save_button.js_on_click(CustomJS(args={"js_data": js_data}, code=javaScript))

    fitpeak_controls = row(
        column(fitparams_add_dropdown, fitparams_select, fitparams_remove_button),
        fitparams_table,
        Spacer(width=20),
        column(
            fit_from_spinner,
            lorentz_checkbox,
            area_method_div,
            area_method_radiobutton,
            intensity_diff_div,
            intensity_diff_radiobutton,
        ),
        column(fit_to_spinner, proc_button, proc_all_button),
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
            export_target_select, hkl_precision_select, column(Spacer(height=19), row(save_button))
        ),
    )

    tab_layout = column(
        row(import_layout, scan_layout, plot, Spacer(width=30), export_layout),
        row(fitpeak_controls, fit_output_textinput),
    )

    return Panel(child=tab_layout, title="ccl compare")
