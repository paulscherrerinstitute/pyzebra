import argparse

from bokeh.io import curdoc
from bokeh.models import Tabs

import panel_anatric
import panel_data_viewer

parser = argparse.ArgumentParser(
    prog="pyzebra", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

args = parser.parse_args()

doc = curdoc()
doc.title = "pyzebra"

# Final layout
tab_data_viewer = panel_data_viewer.create()
tab_anatric = panel_anatric.create()

doc.add_root(Tabs(tabs=[tab_data_viewer, tab_anatric]))
