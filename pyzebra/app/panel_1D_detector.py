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
    Spacer,
)

import pyzebra


def create():
    def upload_button_callback(_attr, _old, new):
        with io.StringIO(base64.b64decode(new).decode()) as file:
            _, ext = os.path.splitext(upload_button.filename)
            res = pyzebra.parse_1D(file, ext)
            for _, meas in res["Measurements"].items():
                y = meas["Counts"]
                x = list(range(len(y)))
                break

            plot_line_source.data.update(x=x, y=y)

    upload_button = FileInput(accept=".ccl")
    upload_button.on_change("value", upload_button_callback)

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

    upload_div = Div(text="Upload .ccl file:")
    tab_layout = column(row(column(Spacer(height=5), upload_div), upload_button), plot)

    return Panel(child=tab_layout, title="1D Detector")
