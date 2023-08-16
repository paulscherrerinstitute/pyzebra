import argparse
import logging
import sys

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Panel, Tabs, TextAreaInput, TextInput

import pyzebra
from pyzebra.app import (
    panel_ccl_compare,
    panel_ccl_integrate,
    panel_ccl_prepare,
    panel_hdf_anatric,
    panel_hdf_param_study,
    panel_hdf_viewer,
    panel_param_study,
    panel_plot_data,
    panel_spind,
)

doc = curdoc()
doc.title = "pyzebra"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--anatric-path", type=str, default=pyzebra.ANATRIC_PATH, help="path to anatric executable"
)

parser.add_argument(
    "--sxtal-refgen-path",
    type=str,
    default=pyzebra.SXTAL_REFGEN_PATH,
    help="path to Sxtal_Refgen executable",
)

parser.add_argument("--spind-path", type=str, default=None, help="path to spind scripts folder")

args = parser.parse_args()

doc.anatric_path = args.anatric_path
doc.spind_path = args.spind_path
doc.sxtal_refgen_path = args.sxtal_refgen_path

# In app_hooks.py a StreamHandler was added to "bokeh" logger
bokeh_stream = logging.getLogger("bokeh").handlers[0].stream

log_textareainput = TextAreaInput(title="logging output:")
bokeh_log_textareainput = TextAreaInput(title="bokeh server output:")


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
                panel_plot_data.create(),
                panel_ccl_integrate.create(),
                panel_ccl_compare.create(),
                panel_param_study.create(),
                panel_hdf_param_study.create(),
                panel_spind.create(),
            ]
        ),
        row(log_textareainput, bokeh_log_textareainput, sizing_mode="scale_both"),
    )
)


def update_stdout():
    log_textareainput.value = sys.stdout.getvalue()
    bokeh_log_textareainput.value = bokeh_stream.getvalue()


doc.add_periodic_callback(update_stdout, 1000)
