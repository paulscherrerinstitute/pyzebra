import argparse

from bokeh.io import curdoc
from bokeh.models import Tabs

import panel_anatric
import panel_data_viewer

parser = argparse.ArgumentParser(
    prog="pyzebra", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--init-meta", metavar="PATH", type=str, default="", help="initial path to .cami file",
)

args = parser.parse_args()

doc = curdoc()
doc.title = "pyzebra"

# Final layout
tab_data_viewer = panel_data_viewer.create(args.init_meta)
tab_anatric = panel_anatric.create()

doc.add_root(Tabs(tabs=[tab_data_viewer, tab_anatric]))
