import base64
import io
import os
import tempfile
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
    FileInput,
    Grid,
    Line,
    LinearAxis,
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
    peak_pos_textinput_lock = False
    js_data = ColumnDataSource(data=dict(cont=[], ext=[]))

    def proposal_textinput_callback(_attr, _old, new):
        ccl_path = os.path.join(PROPOSAL_PATH, new)
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
        num_of_peaks = [scan.get("num_of_peaks", 0) for scan in det_data["scan"].values()]
        fit_ok = [(1 if "fit" in scan else 0) for scan in det_data["scan"].values()]
        scan_table_source.data.update(peaks=num_of_peaks, fit=fit_ok)

    def _update_plot(scan):
        nonlocal peak_pos_textinput_lock
        peak_pos_textinput_lock = True

        y = scan["Counts"]
        x = scan["om"]

        plot_scatter_source.data.update(x=x, y=y, y_upper=y + np.sqrt(y), y_lower=y - np.sqrt(y))

        num_of_peaks = scan.get("num_of_peaks")
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

            scan["num_of_peaks"] = 1
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

    centre_guess = Spinner(default_size=100)
    centre_vary = Toggle(default_size=100, active=True)
    centre_min = Spinner(default_size=100)
    centre_max = Spinner(default_size=100)
    sigma_guess = Spinner(default_size=100)
    sigma_vary = Toggle(default_size=100, active=True)
    sigma_min = Spinner(default_size=100)
    sigma_max = Spinner(default_size=100)
    ampl_guess = Spinner(default_size=100)
    ampl_vary = Toggle(default_size=100, active=True)
    ampl_min = Spinner(default_size=100)
    ampl_max = Spinner(default_size=100)
    slope_guess = Spinner(default_size=100)
    slope_vary = Toggle(default_size=100, active=True)
    slope_min = Spinner(default_size=100)
    slope_max = Spinner(default_size=100)
    offset_guess = Spinner(default_size=100)
    offset_vary = Toggle(default_size=100, active=True)
    offset_min = Spinner(default_size=100)
    offset_max = Spinner(default_size=100)
    integ_from = Spinner(title="Integrate from:", default_size=145)
    integ_to = Spinner(title="to:", default_size=145)

    def fitparam_reset_button_callback():
        centre_guess.value = None
        centre_vary.active = True
        centre_min.value = None
        centre_max.value = None
        sigma_guess.value = None
        sigma_vary.active = True
        sigma_min.value = None
        sigma_max.value = None
        ampl_guess.value = None
        ampl_vary.active = True
        ampl_min.value = None
        ampl_max.value = None
        slope_guess.value = None
        slope_vary.active = True
        slope_min.value = None
        slope_max.value = None
        offset_guess.value = None
        offset_vary.active = True
        offset_min.value = None
        offset_max.value = None
        integ_from.value = None
        integ_to.value = None

    fitparam_reset_button = Button(label="Reset to defaults", default_size=145)
    fitparam_reset_button.on_click(fitparam_reset_button_callback)

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
            guess=[
                centre_guess.value,
                sigma_guess.value,
                ampl_guess.value,
                slope_guess.value,
                offset_guess.value,
            ],
            vary=[
                centre_vary.active,
                sigma_vary.active,
                ampl_vary.active,
                slope_vary.active,
                offset_vary.active,
            ],
            constraints_min=[
                centre_min.value,
                sigma_min.value,
                ampl_min.value,
                slope_min.value,
                offset_min.value,
            ],
            constraints_max=[
                centre_max.value,
                sigma_max.value,
                ampl_max.value,
                slope_max.value,
                offset_max.value,
            ],
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

    div_1 = Div(text="Guess:")
    div_2 = Div(text="Vary:")
    div_3 = Div(text="Min:")
    div_4 = Div(text="Max:")
    div_5 = Div(text="Gauss Centre:", margin=[5, 5, -5, 5])
    div_6 = Div(text="Gauss Sigma:", margin=[5, 5, -5, 5])
    div_7 = Div(text="Gauss Ampl.:", margin=[5, 5, -5, 5])
    div_8 = Div(text="Slope:", margin=[5, 5, -5, 5])
    div_9 = Div(text="Offset:", margin=[5, 5, -5, 5])
    fitpeak_controls = row(
        column(
            Spacer(height=36),
            div_1,
            Spacer(height=12),
            div_2,
            Spacer(height=12),
            div_3,
            Spacer(height=12),
            div_4,
        ),
        column(div_5, centre_guess, centre_vary, centre_min, centre_max),
        column(div_6, sigma_guess, sigma_vary, sigma_min, sigma_max),
        column(div_7, ampl_guess, ampl_vary, ampl_min, ampl_max),
        column(div_8, slope_guess, slope_vary, slope_min, slope_max),
        column(div_9, offset_guess, offset_vary, offset_min, offset_max),
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
