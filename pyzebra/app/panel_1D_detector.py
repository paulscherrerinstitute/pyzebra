import base64
import io
import os

from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
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

    # Measurement select
    def meas_select_callback(_attr, _old, new):
        y = det_data["Measurements"][new]["Counts"]
        x = list(range(len(y)))

        plot_line_source.data.update(x=x, y=y)

    meas_select = Select()
    meas_select.on_change("value", meas_select_callback)

    upload_div = Div(text="Upload .ccl file:")
    tab_layout = column(row(column(Spacer(height=5), upload_div), upload_button, meas_select), plot)

    return Panel(child=tab_layout, title="1D Detector")
