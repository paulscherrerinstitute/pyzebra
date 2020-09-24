import base64
import io
import os

from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    Button,
    Circle,
    ColumnDataSource,
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
    Toggle,
)

import pyzebra


def create():
    det_data = {}

    def upload_button_callback(_attr, _old, new):
        nonlocal det_data
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            det_data = pyzebra.parse_1D(file, ext)

        meas_list = list(det_data["Measurements"].keys())
        meas_select.options = meas_list
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
            plot_smooth_source.data.update(x=x, y=meas["smooth_peaks"])
        else:
            plot_circle_source.data.update(x=[], y=[])
            plot_smooth_source.data.update(x=[], y=[])

        fit = meas.get("fit")
        if fit is not None:
            fit_output_textinput.value = str(fit["full_report"])
        else:
            fit_output_textinput.value = ""

    # Main plot
    plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=400,
        plot_width=600,
        toolbar_location=None,
    )

    plot.add_layout(LinearAxis(axis_label="Counts"), place="left")
    plot.add_layout(LinearAxis(), place="below")

    plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    plot_line_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_line_source, Line(x="x", y="y", line_color="steelblue"))

    plot_smooth_source = ColumnDataSource(dict(x=[0], y=[0]))
    plot.add_glyph(plot_smooth_source, Line(x="x", y="y", line_color="red"))

    plot_circle_source = ColumnDataSource(dict(x=[], y=[]))
    plot.add_glyph(plot_circle_source, Circle(x="x", y="y"))

    # Measurement select
    def meas_select_callback(_attr, _old, new):
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

    upload_div = Div(text="Upload .ccl file:")
    tab_layout = column(
        row(column(Spacer(height=5), upload_div), upload_button, meas_select),
        row(plot, fit_output_textinput),
        row(smooth_toggle),
        row(process_button),
    )

    return Panel(child=tab_layout, title="1D Detector")
