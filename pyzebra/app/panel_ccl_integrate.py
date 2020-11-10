import base64
import io
import os
import tempfile
import types
from copy import deepcopy

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Asterisk,
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


javaScript = """
setTimeout(function() {
    const filename = 'output' + js_data.data['ext']
    const blob = new Blob([js_data.data['cont']], {type: 'text/plain'})
    const link = document.createElement('a');
    document.body.appendChild(link);
    const url = window.URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(link);
}, 500);
"""

PROPOSAL_PATH = "/afs/psi.ch/project/sinqdata/2020/zebra/"


def create():
    det_data = {}
    fit_params = {}
    peak_pos_textinput_lock = False
    js_data = ColumnDataSource(data=dict(cont=[], ext=[]))

    def proposal_textinput_callback(_attr, _old, new):
        ccl_path = os.path.join(PROPOSAL_PATH, new.strip())
        ccl_file_list = []
        for file in os.listdir(ccl_path):
            if file.endswith(".ccl"):
                ccl_file_list.append((os.path.join(ccl_path, file), file))
        ccl_file_select.options = ccl_file_list
        ccl_file_select.value = ccl_file_list[0][0]

    proposal_textinput = TextInput(title="Enter proposal number:", default_size=145)
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def _init_datatable():
        scan_list = list(det_data["scan"].keys())
        hkl = [
            f'{int(m["h_index"])} {int(m["k_index"])} {int(m["l_index"])}'
            for m in det_data["scan"].values()
        ]
        scan_table_source.data.update(
            scan=scan_list,
            hkl=hkl,
            peaks=[0] * len(scan_list),
            fit=[0] * len(scan_list),
            export=[True] * len(scan_list),
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

    def ccl_file_select_callback(_attr, _old, new):
        nonlocal det_data
        with open(new) as file:
            _, ext = os.path.splitext(new)
            det_data = pyzebra.parse_1D(file, ext)

        _init_datatable()

    ccl_file_select = Select(title="Available .ccl files")
    ccl_file_select.on_change("value", ccl_file_select_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            det_data = pyzebra.parse_1D(file, ext)

        _init_datatable()

    upload_button = FileInput(accept=".ccl")
    upload_button.on_change("value", upload_button_callback)

    def append_upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(append_upload_button.filename)
            append_data = pyzebra.parse_1D(file, ext)

        added = pyzebra.add_dict(det_data, append_data)
        scan_result = pyzebra.auto(pyzebra.scan_dict(added))
        det_data = pyzebra.merge(added, added, scan_result)

        _init_datatable()

    append_upload_button = FileInput(accept=".ccl,.dat")
    append_upload_button.on_change("value", append_upload_button_callback)

    def _update_table():
        num_of_peaks = [len(scan.get("peak_indexes", [])) for scan in det_data["scan"].values()]
        fit_ok = [(1 if "fit" in scan else 0) for scan in det_data["scan"].values()]
        scan_table_source.data.update(peaks=num_of_peaks, fit=fit_ok)

    def _update_plot(scan):
        nonlocal peak_pos_textinput_lock
        peak_pos_textinput_lock = True

        y = scan["Counts"]
        x = scan["om"]

        plot_scatter_source.data.update(x=x, y=y, y_upper=y + np.sqrt(y), y_lower=y - np.sqrt(y))

        num_of_peaks = len(scan.get("peak_indexes", []))
        if num_of_peaks is not None and num_of_peaks > 0:
            peak_indexes = scan["peak_indexes"]
            if len(peak_indexes) == 1:
                peak_pos_textinput.value = str(scan["om"][peak_indexes[0]])
            else:
                peak_pos_textinput.value = str([scan["om"][ind] for ind in peak_indexes])

            plot_peak_source.data.update(x=scan["om"][peak_indexes], y=scan["peak_heights"])
            plot_line_smooth_source.data.update(x=x, y=scan["smooth_peaks"])
        else:
            peak_pos_textinput.value = None
            plot_peak_source.data.update(x=[], y=[])
            plot_line_smooth_source.data.update(x=[], y=[])

        peak_pos_textinput_lock = False

        fit = scan.get("fit")
        if fit is not None:
            x = scan["fit"]["x_fit"]
            plot_gauss_source.data.update(x=x, y=scan["fit"]["comps"]["gaussian"])
            plot_bkg_source.data.update(x=x, y=scan["fit"]["comps"]["background"])
            params = fit["result"].params
            fit_output_textinput.value = (
                "Gaussian: centre = %9.4f, sigma = %9.4f, area = %9.4f \n"
                "background: slope = %9.4f, intercept = %9.4f \n"
                "Int. area = %9.4f +/- %9.4f \n"
                "fit area = %9.4f +/- %9.4f \n"
                "ratio((fit-int)/fit) = %9.4f"
                % (
                    params["g_cen"].value,
                    params["g_width"].value,
                    params["g_amp"].value,
                    params["slope"].value,
                    params["intercept"].value,
                    fit["int_area"].n,
                    fit["int_area"].s,
                    params["g_amp"].value,
                    params["g_amp"].stderr,
                    (params["g_amp"].value - fit["int_area"].n) / params["g_amp"].value,
                )
            )
            numfit_min, numfit_max = fit["numfit"]
            if numfit_min is None:
                numfit_min_span.location = None
            else:
                numfit_min_span.location = x[numfit_min]

            if numfit_max is None:
                numfit_max_span.location = None
            else:
                numfit_max_span.location = x[numfit_max]

        else:
            plot_gauss_source.data.update(x=[], y=[])
            plot_bkg_source.data.update(x=[], y=[])
            fit_output_textinput.value = ""
            numfit_min_span.location = None
            numfit_max_span.location = None

    # Main plot
    plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=400, plot_width=700)

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(axis_label="Omega"), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_scatter_source = ColumnDataSource(dict(x=[0], y=[0], y_upper=[0], y_lower=[0]))
    plot.add_glyph(plot_scatter_source, Scatter(x="x", y="y", line_color="steelblue"))
    plot.add_layout(Whisker(source=plot_scatter_source, base="x", upper="y_upper", lower="y_lower"))

    plot_line_smooth_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(
        plot_line_smooth_source, Line(x="x", y="y", line_color="steelblue", line_dash="dashed")
    )

    plot_gauss_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_gauss_source, Line(x="x", y="y", line_color="red", line_dash="dashed"))

    plot_bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_bkg_source, Line(x="x", y="y", line_color="green", line_dash="dashed"))

    plot_peak_source = ColumnDataSource(dict(x=[], y=[]))
    plot.add_glyph(plot_peak_source, Asterisk(x="x", y="y", size=10, line_color="red"))

    numfit_min_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(numfit_min_span)

    numfit_max_span = Span(location=None, dimension="height", line_dash="dashed")
    plot.add_layout(numfit_max_span)

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

        _update_plot(det_data["scan"][scan_table_source.data["scan"][new[0]]])

    scan_table_source = ColumnDataSource(dict(scan=[], hkl=[], peaks=[], fit=[], export=[]))
    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="scan", title="scan"),
            TableColumn(field="hkl", title="hkl"),
            TableColumn(field="peaks", title="Peaks"),
            TableColumn(field="fit", title="Fit"),
            TableColumn(field="export", title="Export", editor=CheckboxEditor()),
        ],
        width=250,
        index_position=None,
        editable=True,
    )

    scan_table_source.selected.on_change("indices", scan_table_select_callback)

    def _get_selected_scan():
        selected_index = scan_table_source.selected.indices[0]
        selected_scan_id = scan_table_source.data["scan"][selected_index]
        return det_data["scan"][selected_scan_id]

    def peak_pos_textinput_callback(_attr, _old, new):
        if new is not None and not peak_pos_textinput_lock:
            scan = _get_selected_scan()

            peak_ind = (np.abs(scan["om"] - float(new))).argmin()
            scan["peak_indexes"] = np.array([peak_ind], dtype=np.int64)
            scan["peak_heights"] = np.array([scan["smooth_peaks"][peak_ind]])
            _update_table()
            _update_plot(scan)

    peak_pos_textinput = TextInput(title="Peak position:", default_size=145)
    peak_pos_textinput.on_change("value", peak_pos_textinput_callback)

    peak_int_ratio_spinner = Spinner(
        title="Peak intensity ratio:", value=0.8, step=0.01, low=0, high=1, default_size=145
    )
    peak_prominence_spinner = Spinner(title="Peak prominence:", value=50, low=0, default_size=145)
    smooth_toggle = Toggle(label="Smooth curve", default_size=145)
    window_size_spinner = Spinner(title="Window size:", value=7, step=2, low=1, default_size=145)
    poly_order_spinner = Spinner(title="Poly order:", value=3, low=0, default_size=145)

    integ_from = Spinner(title="Integrate from:", default_size=145)
    integ_to = Spinner(title="to:", default_size=145)

    def fitparam_reset_button_callback():
        ...

    fitparam_reset_button = Button(label="Reset to defaults", default_size=145, disabled=True)
    fitparam_reset_button.on_click(fitparam_reset_button_callback)

    def fitparams_add_dropdown_callback(click):
        new_tag = str(fitparams_select.tags[0])  # bokeh requires (str, str) for MultiSelect options
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
            ("Pseudo Voigt1", "pseudovoigt1"),
        ],
        default_size=145,
        disabled=True,
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
            fitparams_table_source.data.update(dict(param=[], guess=[], vary=[], min=[], max=[]))

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

    fitparams_remove_button = Button(label="Remove fit function", default_size=145, disabled=True)
    fitparams_remove_button.on_click(fitparams_remove_button_callback)

    def fitparams_factory(function):
        if function == "background":
            params = ["slope", "offset"]
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
            param=params, guess=[None] * n, vary=[True] * n, min=[None] * n, max=[None] * n,
        )

        return fitparams

    fitparams_table_source = ColumnDataSource(dict(param=[], guess=[], vary=[], min=[], max=[]))
    fitparams_table = DataTable(
        source=fitparams_table_source,
        columns=[
            TableColumn(field="param", title="Parameter"),
            TableColumn(field="guess", title="Guess", editor=NumberEditor()),
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

    fit_output_textinput = TextAreaInput(title="Fit results:", width=450, height=400)

    def _get_peakfind_params():
        return dict(
            int_threshold=peak_int_ratio_spinner.value,
            prominence=peak_prominence_spinner.value,
            smooth=smooth_toggle.active,
            window_size=window_size_spinner.value,
            poly_order=poly_order_spinner.value,
        )

    def peakfind_all_button_callback():
        peakfind_params = _get_peakfind_params()
        for scan in det_data["scan"].values():
            pyzebra.ccl_findpeaks(scan, **peakfind_params)

        _update_table()
        _update_plot(_get_selected_scan())

    peakfind_all_button = Button(label="Peak Find All", button_type="primary", default_size=145)
    peakfind_all_button.on_click(peakfind_all_button_callback)

    def peakfind_button_callback():
        scan = _get_selected_scan()
        pyzebra.ccl_findpeaks(scan, **_get_peakfind_params())

        _update_table()
        _update_plot(scan)

    peakfind_button = Button(label="Peak Find Current", default_size=145)
    peakfind_button.on_click(peakfind_button_callback)

    def _get_fit_params():
        return dict(
            guess=fit_params["1"]["guess"] + fit_params["0"]["guess"],
            vary=fit_params["1"]["vary"] + fit_params["0"]["vary"],
            constraints_min=fit_params["1"]["min"] + fit_params["0"]["min"],
            constraints_max=fit_params["1"]["max"] + fit_params["0"]["max"],
            numfit_min=integ_from.value,
            numfit_max=integ_to.value,
            binning=bin_size_spinner.value,
        )

    def fit_all_button_callback():
        fit_params = _get_fit_params()
        for scan in det_data["scan"].values():
            # fit_params are updated inplace within `fitccl`
            pyzebra.fitccl(scan, **deepcopy(fit_params))

        _update_plot(_get_selected_scan())
        _update_table()

    fit_all_button = Button(label="Fit All", button_type="primary", default_size=145)
    fit_all_button.on_click(fit_all_button_callback)

    def fit_button_callback():
        scan = _get_selected_scan()
        pyzebra.fitccl(scan, **_get_fit_params())

        _update_plot(scan)
        _update_table()

    fit_button = Button(label="Fit Current", default_size=145)
    fit_button.on_click(fit_button_callback)

    def area_method_radiobutton_callback(_attr, _old, new):
        det_data["meta"]["area_method"] = ("fit", "integ")[new]

    area_method_radiobutton = RadioButtonGroup(
        labels=["Fit", "Integral"], active=0, default_size=145
    )
    area_method_radiobutton.on_change("active", area_method_radiobutton_callback)

    bin_size_spinner = Spinner(title="Bin size:", value=1, low=1, step=1, default_size=145)

    lorentz_toggle = Toggle(label="Lorentz Correction", default_size=145)

    preview_output_textinput = TextAreaInput(title="Export file preview:", width=450, height=400)

    def preview_output_button_callback():
        if det_data["meta"]["indices"] == "hkl":
            ext = ".comm"
        elif det_data["meta"]["indices"] == "real":
            ext = ".incomm"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = deepcopy(det_data)
            for s, export in zip(scan_table_source.data["scan"], scan_table_source.data["export"]):
                if not export:
                    del export_data["scan"][s]
            pyzebra.export_comm(export_data, temp_file, lorentz=lorentz_toggle.active)

            with open(f"{temp_file}{ext}") as f:
                preview_output_textinput.value = f.read()

    preview_output_button = Button(label="Preview file", default_size=220)
    preview_output_button.on_click(preview_output_button_callback)

    def export_results(det_data):
        if det_data["meta"]["indices"] == "hkl":
            ext = ".comm"
        elif det_data["meta"]["indices"] == "real":
            ext = ".incomm"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            export_data = deepcopy(det_data)
            for s, export in zip(scan_table_source.data["scan"], scan_table_source.data["export"]):
                if not export:
                    del export_data["scan"][s]
            pyzebra.export_comm(export_data, temp_file, lorentz=lorentz_toggle.active)

            with open(f"{temp_file}{ext}") as f:
                output_content = f.read()

        return output_content, ext

    def save_button_callback():
        cont, ext = export_results(det_data)
        js_data.data.update(cont=[cont], ext=[ext])

    save_button = Button(label="Download file", button_type="success", default_size=220)
    save_button.on_click(save_button_callback)
    save_button.js_on_click(CustomJS(args={"js_data": js_data}, code=javaScript))

    findpeak_controls = column(
        row(peak_pos_textinput, column(Spacer(height=19), smooth_toggle)),
        row(peak_int_ratio_spinner, peak_prominence_spinner),
        row(window_size_spinner, poly_order_spinner),
        row(peakfind_button, peakfind_all_button),
    )

    fitpeak_controls = row(
        column(fitparams_add_dropdown, fitparams_select, fitparams_remove_button),
        fitparams_table,
        Spacer(width=20),
        column(
            row(integ_from, integ_to),
            row(bin_size_spinner, column(Spacer(height=19), lorentz_toggle)),
            row(fitparam_reset_button, area_method_radiobutton),
            row(fit_button, fit_all_button),
        ),
    )

    export_layout = column(preview_output_textinput, row(preview_output_button, save_button))

    upload_div = Div(text="Or upload .ccl file:")
    append_upload_div = Div(text="append extra .ccl/.dat files:")
    tab_layout = column(
        row(proposal_textinput, ccl_file_select),
        row(
            column(Spacer(height=5), upload_div),
            upload_button,
            column(Spacer(height=5), append_upload_div),
            append_upload_button,
        ),
        row(scan_table, plot, Spacer(width=30), fit_output_textinput, export_layout),
        row(findpeak_controls, Spacer(width=30), fitpeak_controls),
    )

    return Panel(child=tab_layout, title="ccl integrate")
