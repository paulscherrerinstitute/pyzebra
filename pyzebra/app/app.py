import argparse

from bokeh.io import curdoc
from bokeh.models import Tabs

import panel_hdf_anatric
import panel_hdf_viewer
import panel_ccl_integrate

parser = argparse.ArgumentParser(
    prog="pyzebra", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

args = parser.parse_args()

doc = curdoc()
doc.title = "pyzebra"

# Final layout
tab_hdf_viewer = panel_hdf_viewer.create()
tab_hdf_anatric = panel_hdf_anatric.create()
tab_ccl_integrate = panel_ccl_integrate.create()

doc.add_root(Tabs(tabs=[tab_hdf_viewer, tab_hdf_anatric, tab_ccl_integrate]))
