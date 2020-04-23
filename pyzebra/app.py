import argparse

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    BoxEditTool,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    Dropdown,
    Grid,
    HoverTool,
    Image,
    Line,
    LinearAxis,
    LinearColorMapper,
    PanTool,
    Plot,
    RadioButtonGroup,
    Range1d,
    Rect,
    ResetTool,
    SaveTool,
    Select,
    Spinner,
    TextAreaInput,
    TextInput,
    Title,
    Toggle,
    WheelZoomTool,
)
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611

import pyzebra

parser = argparse.ArgumentParser(
    prog="pyzebra", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--init-meta", metavar="PATH", type=str, default="", help="initial path to .cami file",
)

args = parser.parse_args()


IMAGE_W = 256
IMAGE_H = 128

doc = curdoc()
doc.title = "pyzebra"

global curent_h5_data, current_index, det_data


def update_image():
    current_image = curent_h5_data[current_index]
    proj_v_line_source.data.update(x=np.arange(0, IMAGE_W) + 0.5, y=np.mean(current_image, axis=0))
    proj_h_line_source.data.update(x=np.mean(current_image, axis=1), y=np.arange(0, IMAGE_H) + 0.5)

    image_source.data.update(
        h=[np.zeros((1, 1))], k=[np.zeros((1, 1))], l=[np.zeros((1, 1))],
    )
    image_source.data.update(image=[current_image])
    index_spinner.value = current_index

    if auto_toggle.active:
        im_max = int(np.max(current_image))
        im_min = int(np.min(current_image))

        display_min_spinner.value = im_min
        display_max_spinner.value = im_max

        image_glyph.color_mapper.low = im_min
        image_glyph.color_mapper.high = im_max


def calculate_hkl(setup_type="nb_bi"):
    h = np.empty(shape=(IMAGE_H, IMAGE_W))
    k = np.empty(shape=(IMAGE_H, IMAGE_W))
    l = np.empty(shape=(IMAGE_H, IMAGE_W))

    wave = det_data["wave"]
    ddist = det_data["ddist"]
    gammad = det_data["pol_angle"][current_index]
    om = det_data["rot_angle"][current_index]
    nud = det_data["tlt_angle"]
    ub = det_data["UB"]

    if setup_type == "nb_bi":
        ch = det_data["chi_angle"][current_index]
        ph = det_data["phi_angle"][current_index]
    elif setup_type == "nb":
        ch = 0
        ph = 0
    else:
        raise ValueError(f"Unknown setup type '{setup_type}'")

    for xi in np.arange(IMAGE_W):
        for yi in np.arange(IMAGE_H):
            h[yi, xi], k[yi, xi], l[yi, xi] = pyzebra.ang2hkl(
                wave, ddist, gammad, om, ch, ph, nud, ub, xi, yi
            )

    return h, k, l


def filelist_callback(_attr, _old, new):
    global curent_h5_data, current_index, det_data
    det_data = pyzebra.read_detector_data(new)
    data = det_data["data"]
    curent_h5_data = data
    current_index = 0
    update_image()

    # update overview plots
    overview_x = np.mean(data, axis=1)
    overview_y = np.mean(data, axis=2)

    overview_plot_x_image_source.data.update(
        image=[overview_x], dh=[overview_x.shape[0]], dw=[overview_x.shape[1]]
    )
    overview_plot_y_image_source.data.update(
        image=[overview_y], dh=[overview_y.shape[0]], dw=[overview_y.shape[1]]
    )


filelist = Dropdown()
filelist.on_change("value", filelist_callback)


def fileinput_callback(_attr, _old, new):
    h5meta_list = pyzebra.read_h5meta(new)
    file_list = h5meta_list["filelist"]
    filelist.menu = file_list


fileinput = TextInput()
fileinput.on_change("value", fileinput_callback)
if args.init_meta:
    fileinput.value = args.init_meta


def index_spinner_callback(_attr, _old, new):
    global current_index
    if 0 <= new < curent_h5_data.shape[0]:
        current_index = new
        update_image()


index_spinner = Spinner(value=0)
index_spinner.on_change("value", index_spinner_callback)

plot = Plot(
    x_range=Range1d(0, IMAGE_W, bounds=(0, IMAGE_W)),
    y_range=Range1d(0, IMAGE_H, bounds=(0, IMAGE_H)),
    plot_height=IMAGE_H * 3,
    plot_width=IMAGE_W * 3,
    toolbar_location="left",
)

# ---- tools
plot.toolbar.logo = None

# ---- axes
plot.add_layout(LinearAxis(), place="above")
plot.add_layout(LinearAxis(major_label_orientation="vertical"), place="right")

# ---- grid lines
plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
image_source = ColumnDataSource(
    dict(
        image=[np.zeros((IMAGE_H, IMAGE_W), dtype="float32")],
        h=[np.zeros((1, 1))],
        k=[np.zeros((1, 1))],
        l=[np.zeros((1, 1))],
        x=[0],
        y=[0],
        dw=[IMAGE_W],
        dh=[IMAGE_H],
    )
)

h_glyph = Image(image="h", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
k_glyph = Image(image="k", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
l_glyph = Image(image="l", x="x", y="y", dw="dw", dh="dh", global_alpha=0)

plot.add_glyph(image_source, h_glyph)
plot.add_glyph(image_source, k_glyph)
plot.add_glyph(image_source, l_glyph)

image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
image_renderer = plot.add_glyph(image_source, image_glyph, name="image_glyph")

# ---- projections
proj_v = Plot(
    x_range=plot.x_range,
    y_range=DataRange1d(),
    plot_height=200,
    plot_width=IMAGE_W * 3,
    toolbar_location=None,
)

proj_v.add_layout(LinearAxis(major_label_orientation="vertical"), place="right")
proj_v.add_layout(LinearAxis(major_label_text_font_size="0pt"), place="below")

proj_v.add_layout(Grid(dimension=0, ticker=BasicTicker()))
proj_v.add_layout(Grid(dimension=1, ticker=BasicTicker()))

proj_v_line_source = ColumnDataSource(dict(x=[], y=[]))
proj_v.add_glyph(proj_v_line_source, Line(x="x", y="y", line_color="steelblue"))

proj_h = Plot(
    x_range=DataRange1d(),
    y_range=plot.y_range,
    plot_height=IMAGE_H * 3,
    plot_width=200,
    toolbar_location=None,
)

proj_h.add_layout(LinearAxis(), place="above")
proj_h.add_layout(LinearAxis(major_label_text_font_size="0pt"), place="left")

proj_h.add_layout(Grid(dimension=0, ticker=BasicTicker()))
proj_h.add_layout(Grid(dimension=1, ticker=BasicTicker()))

proj_h_line_source = ColumnDataSource(dict(x=[], y=[]))
proj_h.add_glyph(proj_h_line_source, Line(x="x", y="y", line_color="steelblue"))

# add tools
hovertool = HoverTool(tooltips=[("intensity", "@image"), ("h", "@h"), ("k", "@k"), ("l", "@l")])

box_edit_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
box_edit_glyph = Rect(x="x", y="y", width="width", height="height", fill_alpha=0, line_color="red")
box_edit_renderer = plot.add_glyph(box_edit_source, box_edit_glyph)
boxedittool = BoxEditTool(renderers=[box_edit_renderer], num_objects=1)


def box_edit_callback(_attr, _old, new):
    if new["x"]:
        x_val = np.arange(curent_h5_data.shape[0])
        left = int(np.floor(new["x"][0]))
        right = int(np.ceil(new["x"][0] + new["width"][0]))
        bottom = int(np.floor(new["y"][0]))
        top = int(np.ceil(new["y"][0] + new["height"][0]))
        y_val = np.sum(curent_h5_data[:, bottom:top, left:right], axis=(1, 2))
    else:
        x_val = []
        y_val = []

    roi_avg_plot_line_source.data.update(x=x_val, y=y_val)


box_edit_source.on_change("data", box_edit_callback)

plot.add_tools(
    PanTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool(), hovertool, boxedittool,
)
plot.toolbar.active_scroll = plot.tools[1]


# shared frame range
frame_range = DataRange1d()
det_x_range = DataRange1d()
overview_plot_x = Plot(
    title=Title(text="Projections on X-axis"),
    x_range=det_x_range,
    y_range=frame_range,
    plot_height=400,
    plot_width=400,
    toolbar_location="left",
)

# ---- tools
overview_plot_x.toolbar.logo = None
overview_plot_x.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(maintain_focus=False), ResetTool(),
)

# ---- axes
overview_plot_x.add_layout(LinearAxis(axis_label="Coordinate X, pix"), place="below")
overview_plot_x.add_layout(
    LinearAxis(axis_label="Frame", major_label_orientation="vertical"), place="left"
)

# ---- grid lines
overview_plot_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
overview_plot_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
overview_plot_x_image_source = ColumnDataSource(
    dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[1], dh=[1])
)

overview_plot_x_image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
overview_plot_x_image_renderer = overview_plot_x.add_glyph(
    overview_plot_x_image_source, overview_plot_x_image_glyph, name="image_glyph"
)


det_y_range = DataRange1d()
overview_plot_y = Plot(
    title=Title(text="Projections on Y-axis"),
    x_range=det_y_range,
    y_range=frame_range,
    plot_height=400,
    plot_width=400,
    toolbar_location="left",
)

# ---- tools
overview_plot_y.toolbar.logo = None
overview_plot_y.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(maintain_focus=False), ResetTool(),
)

# ---- axes
overview_plot_y.add_layout(LinearAxis(axis_label="Coordinate Y, pix"), place="below")
overview_plot_y.add_layout(
    LinearAxis(axis_label="Frame", major_label_orientation="vertical"), place="left"
)

# ---- grid lines
overview_plot_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
overview_plot_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
overview_plot_y_image_source = ColumnDataSource(
    dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[1], dh=[1])
)

overview_plot_y_image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
overview_plot_y_image_renderer = overview_plot_y.add_glyph(
    overview_plot_y_image_source, overview_plot_y_image_glyph, name="image_glyph"
)


roi_avg_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=IMAGE_H * 3,
    plot_width=IMAGE_W * 3,
    toolbar_location="left",
)

# ---- tools
roi_avg_plot.toolbar.logo = None

# ---- axes
roi_avg_plot.add_layout(LinearAxis(), place="below")
roi_avg_plot.add_layout(LinearAxis(major_label_orientation="vertical"), place="left")

# ---- grid lines
roi_avg_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
roi_avg_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

roi_avg_plot_line_source = ColumnDataSource(dict(x=[], y=[]))
roi_avg_plot.add_glyph(roi_avg_plot_line_source, Line(x="x", y="y", line_color="steelblue"))


def prev_button_callback():
    global current_index
    if current_index > 0:
        current_index -= 1
    update_image()


prev_button = Button(label="Previous")
prev_button.on_click(prev_button_callback)


def next_button_callback():
    global current_index
    if current_index < curent_h5_data.shape[0] - 1:
        current_index += 1
    update_image()


next_button = Button(label="Next")
next_button.on_click(next_button_callback)


def animate():
    next_button_callback()


def animate_toggle_callback(active):
    global cb
    if active:
        cb = doc.add_periodic_callback(animate, 300)
    else:
        doc.remove_periodic_callback(cb)


animate_toggle = Toggle(label="Animate")
animate_toggle.on_click(animate_toggle_callback)


cmap_dict = {
    "gray": Greys256,
    "gray_reversed": Greys256[::-1],
    "plasma": Plasma256,
    "cividis": Cividis256,
}


def colormap_callback(_attr, _old, new):
    image_glyph.color_mapper = LinearColorMapper(palette=cmap_dict[new])
    overview_plot_x_image_glyph.color_mapper = LinearColorMapper(palette=cmap_dict[new])
    overview_plot_y_image_glyph.color_mapper = LinearColorMapper(palette=cmap_dict[new])


colormap = Select(title="Colormap:", options=list(cmap_dict.keys()))
colormap.on_change("value", colormap_callback)
colormap.value = "plasma"


radio_button_group = RadioButtonGroup(labels=["nb", "nb_bi"], active=0)

STEP = 1

# ---- colormap auto toggle button
def auto_toggle_callback(state):
    if state:
        display_min_spinner.disabled = True
        display_max_spinner.disabled = True
    else:
        display_min_spinner.disabled = False
        display_max_spinner.disabled = False

    update_image()


auto_toggle = Toggle(label="Auto Range", active=True, button_type="default")
auto_toggle.on_click(auto_toggle_callback)


# ---- colormap display max value
def display_max_spinner_callback(_attr, _old_value, new_value):
    display_min_spinner.high = new_value - STEP
    image_glyph.color_mapper.high = new_value


display_max_spinner = Spinner(
    title="Maximal Display Value:", low=0 + STEP, value=1, step=STEP, disabled=auto_toggle.active,
)
display_max_spinner.on_change("value", display_max_spinner_callback)


# ---- colormap display min value
def display_min_spinner_callback(_attr, _old_value, new_value):
    display_max_spinner.low = new_value + STEP
    image_glyph.color_mapper.low = new_value


display_min_spinner = Spinner(
    title="Minimal Display Value:", high=1 - STEP, value=0, step=STEP, disabled=auto_toggle.active,
)
display_min_spinner.on_change("value", display_min_spinner_callback)


def hkl_button_callback():
    setup_type = "nb_bi" if radio_button_group.active else "nb"
    h, k, l = calculate_hkl(setup_type)
    image_source.data.update(h=[h], k=[k], l=[l])


hkl_button = Button(label="Calculate hkl (slow)")
hkl_button.on_click(hkl_button_callback)


selection_list = TextAreaInput()


def selection_button_callback():
    selection = [
        int(np.floor(det_x_range.start)),
        int(np.ceil(det_x_range.end)),
        int(np.floor(det_y_range.start)),
        int(np.ceil(det_y_range.end)),
        int(np.floor(frame_range.start)),
        int(np.ceil(frame_range.end)),
    ]

    if selection_list.value == "":
        selection_list.value = f"{selection}"
    else:
        selection_list.value = f"{selection_list.value},\n{selection}"


selection_button = Button(label="Add selection")
selection_button.on_click(selection_button_callback)


# Final layout
layout_image = gridplot([[proj_v, None], [plot, proj_h]], merge_tools=False)

animate_layout = column(index_spinner, next_button, prev_button, animate_toggle)
colormap_layout = column(colormap, auto_toggle, display_max_spinner, display_min_spinner)
hkl_layout = column(radio_button_group, hkl_button)

doc.add_root(
    row(
        column(fileinput, filelist, layout_image, row(colormap_layout, animate_layout, hkl_layout)),
        column(
            gridplot(
                [[overview_plot_x, overview_plot_y]],
                toolbar_options=dict(logo=None),
                merge_tools=True,
            ),
            row(selection_button, selection_list),
            roi_avg_plot,
        ),
    )
)
