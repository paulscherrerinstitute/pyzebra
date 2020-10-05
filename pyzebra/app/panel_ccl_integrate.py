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
    Div,
    FileInput,
    Grid,
    Line,
    LinearAxis,
    Panel,
    Plot,
    Select,
    Spacer,
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
        meas_select.options = meas_list
        meas_select.value = None
        meas_select.value = meas_list[0]

    ccl_file_select = Select(title="Available .ccl files")
    ccl_file_select.on_change("value", ccl_file_select_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            det_data = pyzebra.parse_1D(file, ext)

        meas_list = list(det_data["Measurements"].keys())
        meas_select.options = meas_list
        meas_select.value = None
        meas_select.value = meas_list[0]

    upload_button = FileInput(accept=".ccl")
    upload_button.on_change("value", upload_button_callback)

    def _update_plot(ind):
        meas = det_data["Measurements"][ind]
        y = meas["Counts"]
        x = list(range(len(y)))

        plot_line_source.data.update(x=x, y=y)

        num_of_peaks = meas.get("num_of_peaks")
        if num_of_peaks is not None and num_of_peaks > 0:
            plot_circle_source.data.update(x=meas["peak_indexes"], y=meas["peak_heights"])
        else:
            plot_circle_source.data.update(x=[], y=[])

        fit = meas.get("fit")
        if fit is not None:
            plot_gauss_source.data.update(x=x, y=meas["fit"]["comps"]["gaussian"])
            plot_bkg_source.data.update(x=x, y=meas["fit"]["comps"]["background"])
            fit_output_textinput.value = str(fit["full_report"])
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

    plot_gauss_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_gauss_source, Line(x="x", y="y", line_color="red", line_dash="dashed"))

    plot_bkg_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_bkg_source, Line(x="x", y="y", line_color="green", line_dash="dashed"))

    plot_circle_source = ColumnDataSource(dict(x=[], y=[]))
    plot.add_glyph(plot_circle_source, Circle(x="x", y="y"))

    # Measurement select
    def meas_select_callback(_attr, _old, new):
        if new is not None:
            _update_plot(new)

    meas_select = Select()
    meas_select.on_change("value", meas_select_callback)

    smooth_toggle = Toggle(label="Smooth curve")

    fit_output_textinput = TextAreaInput(title="Fit results:", width=600, height=400)

    def process_button_callback():
        nonlocal det_data
        for meas in det_data["Measurements"]:
            det_data = pyzebra.ccl_findpeaks(det_data, meas, smooth=smooth_toggle.active)

            num_of_peaks = det_data["Measurements"][meas].get("num_of_peaks")
            if num_of_peaks is not None and num_of_peaks == 1:
                det_data = pyzebra.fitccl(
                    det_data,
                    meas,
                    guess=[None, None, None, None, None],
                    vary=[True, True, True, True, True],
                    constraints_min=[None, None, None, None, None],
                    constraints_max=[None, None, None, None, None],
                )

        _update_plot(meas_select.value)

    process_button = Button(label="Process All", button_type="primary")
    process_button.on_click(process_button_callback)

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

    save_button = Button(label="Export to .comm/.incomm file:")
    save_button.on_click(save_button_callback)
    save_button.js_on_click(CustomJS(args={"js_data": js_data}, code=javaScript))

    upload_div = Div(text="Or upload .ccl file:")
    tab_layout = column(
        row(proposal_textinput, ccl_file_select),
        row(column(Spacer(height=5), upload_div), upload_button, meas_select),
        row(plot, Spacer(width=30), fit_output_textinput),
        row(column(smooth_toggle, process_button, save_button)),
    )

    return Panel(child=tab_layout, title="ccl integrate")
