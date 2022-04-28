import logging
import sys
from io import StringIO

import pyzebra
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Panel, Tabs, TextAreaInput, TextInput

import panel_ccl_integrate
import panel_ccl_compare
import panel_hdf_anatric
import panel_hdf_param_study
import panel_hdf_viewer
import panel_param_study
import panel_spind
import panel_ccl_prepare

doc = curdoc()

sys.stdout = StringIO()
stdout_textareainput = TextAreaInput(title="print output:", height=150)

bokeh_stream = StringIO()
bokeh_handler = logging.StreamHandler(bokeh_stream)
bokeh_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
bokeh_logger = logging.getLogger("bokeh")
bokeh_logger.addHandler(bokeh_handler)
bokeh_log_textareainput = TextAreaInput(title="server output:", height=150)

def proposal_textinput_callback(_attr, _old, _new):
    apply_button.disabled = False

proposal_textinput = TextInput(title="Proposal number:", name="")
proposal_textinput.on_change("value_input", proposal_textinput_callback)
doc.proposal_textinput = proposal_textinput

def apply_button_callback():
    proposal = proposal_textinput.value.strip()
    if proposal:
        try:
            proposal_path = pyzebra.find_proposal_path(proposal)
        except ValueError as e:
            print(e)
            return
        apply_button.disabled = True
    else:
        proposal_path = ""

    proposal_textinput.name = proposal_path

apply_button = Button(label="Apply", button_type="primary")
apply_button.on_click(apply_button_callback)

# Final layout
doc.add_root(
    column(
        Tabs(
            tabs=[
                Panel(child=column(proposal_textinput, apply_button), title="user config"),
                panel_hdf_viewer.create(),
                panel_hdf_anatric.create(),
                panel_ccl_prepare.create(),
                panel_ccl_integrate.create(),
                panel_ccl_compare.create(),
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
