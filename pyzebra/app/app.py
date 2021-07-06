import logging
import sys
from io import StringIO

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Tabs, TextAreaInput

import panel_ccl_integrate
import panel_hdf_anatric
import panel_hdf_param_study
import panel_hdf_viewer
import panel_param_study
import panel_spind


doc = curdoc()

sys.stdout = StringIO()
stdout_textareainput = TextAreaInput(title="print output:", height=150)

bokeh_stream = StringIO()
bokeh_handler = logging.StreamHandler(bokeh_stream)
bokeh_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
bokeh_logger = logging.getLogger("bokeh")
bokeh_logger.addHandler(bokeh_handler)
bokeh_log_textareainput = TextAreaInput(title="server output:", height=150)

# Final layout
doc.add_root(
    column(
        Tabs(
            tabs=[
                panel_hdf_viewer.create(),
                panel_hdf_anatric.create(),
                panel_ccl_integrate.create(),
                panel_param_study.create(),
                panel_hdf_param_study.create(),
                panel_spind.create(),
            ]
        ),
        row(stdout_textareainput, bokeh_log_textareainput, sizing_mode="scale_both"),
    )
)


def update_stdout():
    stdout_textareainput.value = sys.stdout.getvalue()
    bokeh_log_textareainput.value = bokeh_stream.getvalue()


doc.add_periodic_callback(update_stdout, 1000)
