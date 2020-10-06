import base64
import io
import os
import tempfile

from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    Button,
    Circle,
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
    Plot,
    Select,
    Spacer,
    Spinner,
    TableColumn,
    TextAreaInput,
    TextInput,
    Toggle,
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

    proposal_textinput = TextInput(title="Enter proposal number:")
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def ccl_file_select_callback(_attr, _old, new):
        nonlocal det_data
        with open(new) as file:
            _, ext = os.path.splitext(new)
            det_data = pyzebra.parse_1D(file, ext)

        meas_list = list(det_data["Measurements"].keys())
        meas_table_source.data.update(measurement=meas_list, peaks=[0] * len(meas_list))
        meas_table_source.selected.indices = []
        meas_table_source.selected.indices = [0]

    ccl_file_select = Select(title="Available .ccl files")
    ccl_file_select.on_change("value", ccl_file_select_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            det_data = pyzebra.parse_1D(file, ext)

        meas_list = list(det_data["Measurements"].keys())
        meas_table_source.data.update(measurement=meas_list, peaks=[0] * len(meas_list))
        meas_table_source.selected.indices = []
        meas_table_source.selected.indices = [0]

    upload_button = FileInput(accept=".ccl")
    upload_button.on_change("value", upload_button_callback)

    def _update_table():
        num_of_peaks = [meas.get("num_of_peaks", 0) for meas in det_data["Measurements"].values()]
        meas_table_source.data.update(peaks=num_of_peaks)

    def _update_plot(ind):
        nonlocal peak_pos_textinput_lock
        peak_pos_textinput_lock = True

        meas = det_data["Measurements"][ind]
        y = meas["Counts"]
        x = list(range(len(y)))

        plot_line_source.data.update(x=x, y=y)

        num_of_peaks = meas.get("num_of_peaks")
        if num_of_peaks is not None and num_of_peaks > 0:
            peak_indexes = meas["peak_indexes"]
            if len(peak_indexes) == 1:
                peak_pos_textinput.value = str(peak_indexes[0])
            else:
                peak_pos_textinput.value = str(peak_indexes)

            plot_circle_source.data.update(x=peak_indexes, y=meas["peak_heights"])
            plot_line_smooth_source.data.update(x=x, y=meas["smooth_peaks"])
        else:
            peak_pos_textinput.value = None
            plot_circle_source.data.update(x=[], y=[])
            plot_line_smooth_source.data.update(x=[], y=[])

        peak_pos_textinput_lock = False

        fit = meas.get("fit")
        if fit is not None:
            plot_gauss_source.data.update(x=x, y=meas["fit"]["comps"]["gaussian"])
            plot_bkg_source.data.update(x=x, y=meas["fit"]["comps"]["background"])
            params = fit["result"].params
            fit_output_textinput.value = (
                "%s \n"
                "Gaussian: centre = %9.4f, sigma = %9.4f, area = %9.4f \n"
                "background: slope = %9.4f, intercept = %9.4f \n"
                "Int. area = %9.4f +/- %9.4f \n"
                "fit area = %9.4f +/- %9.4f \n"
                "ratio((fit-int)/fit) = %9.4f"
                % (
                    ind,
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
        else:
            plot_gauss_source.data.update(x=[], y=[])
            plot_bkg_source.data.update(x=[], y=[])
            fit_output_textinput.value = ""

    # Main plot
    plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=400,
        plot_width=700,
        toolbar_location=None,
    )

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_line_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_line_source, Line(x="x", y="y", line_color="steelblue"))

    plot_line_smooth_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(
        plot_line_smooth_source, Line(x="x", y="y", line_color="steelblue", line_dash="dashed")
    )

    plot_gauss_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_gauss_source, Line(x="x", y="y", line_color="red", line_dash="dashed"))

    plot_bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_bkg_source, Line(x="x", y="y", line_color="green", line_dash="dashed"))

    plot_circle_source = ColumnDataSource(dict(x=[], y=[]))
    plot.add_glyph(plot_circle_source, Circle(x="x", y="y"))

    # Measurement select
    def meas_table_callback(_attr, _old, new):
        if new:
            _update_plot(meas_table_source.data["measurement"][new[-1]])

    meas_table_source = ColumnDataSource(dict(measurement=[], peaks=[]))
    meas_table = DataTable(
        source=meas_table_source,
        columns=[
            TableColumn(field="measurement", title="Meas"),
            TableColumn(field="peaks", title="Peaks"),
        ],
        width=100,
        index_position=None,
    )

    meas_table_source.selected.on_change("indices", meas_table_callback)

    def peak_pos_textinput_callback(_attr, _old, new):
        if new is not None and not peak_pos_textinput_lock:
            sel_ind = meas_table_source.selected.indices[-1]
            meas_name = meas_table_source.data["measurement"][sel_ind]
            meas = det_data["Measurements"][meas_name]

            meas["num_of_peaks"] = 1
            meas["peak_indexes"] = [float(new)]
            meas["peak_heights"] = [0]
            _update_table()
            _update_plot(meas_name)

    peak_pos_textinput = TextInput(title="Peak position:")
    peak_pos_textinput.on_change("value", peak_pos_textinput_callback)

    peak_int_ratio_spinner = Spinner(
        title="Peak intensity ratio:", value=0.8, step=0.01, low=0, high=1, default_size=145
    )
    peak_prominence_spinner = Spinner(title="Peak prominence:", value=50, low=0, default_size=145)
    smooth_toggle = Toggle(label="Smooth curve")
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

    fitparam_reset_button = Button(label="Reset to defaults")
    fitparam_reset_button.on_click(fitparam_reset_button_callback)

    fit_output_textinput = TextAreaInput(title="Fit results:", width=450, height=400)

    def peakfind_all_button_callback():
        nonlocal det_data
        for meas in det_data["Measurements"]:
            det_data = pyzebra.ccl_findpeaks(
                det_data,
                meas,
                int_threshold=peak_int_ratio_spinner.value,
                prominence=peak_prominence_spinner.value,
                smooth=smooth_toggle.active,
                window_size=window_size_spinner.value,
                poly_order=poly_order_spinner.value,
            )

        _update_table()

        sel_ind = meas_table_source.selected.indices[-1]
        _update_plot(meas_table_source.data["measurement"][sel_ind])

    peakfind_all_button = Button(label="Peak Find All", button_type="primary")
    peakfind_all_button.on_click(peakfind_all_button_callback)

    def peakfind_button_callback():
        nonlocal det_data
        sel_ind = meas_table_source.selected.indices[-1]
        meas = meas_table_source.data["measurement"][sel_ind]
        det_data = pyzebra.ccl_findpeaks(
            det_data,
            meas,
            int_threshold=peak_int_ratio_spinner.value,
            prominence=peak_prominence_spinner.value,
            smooth=smooth_toggle.active,
            window_size=window_size_spinner.value,
            poly_order=poly_order_spinner.value,
        )

        _update_table()
        _update_plot(meas)

    peakfind_button = Button(label="Peak Find Current")
    peakfind_button.on_click(peakfind_button_callback)

    def fit_all_button_callback():
        nonlocal det_data
        for meas in det_data["Measurements"]:
            num_of_peaks = det_data["Measurements"][meas].get("num_of_peaks")
            if num_of_peaks is not None and num_of_peaks == 1:
                det_data = pyzebra.fitccl(
                    det_data,
                    meas,
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
                )

        sel_ind = meas_table_source.selected.indices[-1]
        _update_plot(meas_table_source.data["measurement"][sel_ind])

    fit_all_button = Button(label="Fit All", button_type="primary")
    fit_all_button.on_click(fit_all_button_callback)

    def fit_button_callback():
        nonlocal det_data
        sel_ind = meas_table_source.selected.indices[-1]
        meas = meas_table_source.data["measurement"][sel_ind]

        num_of_peaks = det_data["Measurements"][meas].get("num_of_peaks")
        if num_of_peaks is not None and num_of_peaks == 1:
            det_data = pyzebra.fitccl(
                det_data,
                meas,
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
            )

        _update_plot(meas)

    fit_button = Button(label="Fit Current")
    fit_button.on_click(fit_button_callback)

    preview_output_textinput = TextAreaInput(title="Export file preview:", width=450, height=400)

    def preview_output_button_callback():
        if det_data["meta"]["indices"] == "hkl":
            ext = ".comm"
        elif det_data["meta"]["indices"] == "real":
            ext = ".incomm"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp"
            pyzebra.export_comm(det_data, temp_file)

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
            pyzebra.export_comm(det_data, temp_file)

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
        peak_pos_textinput,
        row(peak_int_ratio_spinner, peak_prominence_spinner),
        smooth_toggle,
        row(window_size_spinner, poly_order_spinner),
        peakfind_button,
        peakfind_all_button,
    )

    div_1 = Div(text="Guess:")
    div_2 = Div(text="Vary:")
    div_3 = Div(text="Min:")
    div_4 = Div(text="Max:")
    div_5 = Div(text="Gauss Centre:")
    div_6 = Div(text="Gauss Sigma:")
    div_7 = Div(text="Gauss Ampl.:")
    div_8 = Div(text="Slope:")
    div_9 = Div(text="Offset:")
    fitpeak_controls = column(
        row(
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
        ),
        row(fitparam_reset_button),
        row(fit_button),
        row(fit_all_button),
    )

    export_layout = column(preview_output_textinput, row(preview_output_button, save_button))

    upload_div = Div(text="Or upload .ccl file:")
    tab_layout = column(
        row(proposal_textinput, ccl_file_select),
        row(column(Spacer(height=5), upload_div), upload_button),
        row(meas_table, plot, Spacer(width=30), fit_output_textinput, export_layout),
        row(findpeak_controls, Spacer(width=30), fitpeak_controls),
    )

    return Panel(child=tab_layout, title="ccl integrate")
