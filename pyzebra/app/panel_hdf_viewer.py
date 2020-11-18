import base64
import io
import os

import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    BoxEditTool,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    Div,
    FileInput,
    Grid,
    HoverTool,
    Image,
    Line,
    LinearAxis,
    LinearColorMapper,
    Panel,
    PanTool,
    Plot,
    RadioButtonGroup,
    Range1d,
    Rect,
    ResetTool,
    Select,
    Spacer,
    Spinner,
    TextAreaInput,
    Title,
    Toggle,
    WheelZoomTool,
)
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611

import pyzebra

IMAGE_W = 256
IMAGE_H = 128


def create():
    det_data = {}
    roi_selection = {}

    def upload_button_callback(_attr, _old, new):
        with io.StringIO(base64.b64decode(new).decode()) as file:
            h5meta_list = pyzebra.parse_h5meta(file)
            file_list = h5meta_list["filelist"]
            filelist.options = [(entry, os.path.basename(entry)) for entry in file_list]
            filelist.value = file_list[0]

    upload_button = FileInput(accept=".cami")
    upload_button.on_change("value", upload_button_callback)

    def update_image(index=None):
        if index is None:
            index = index_spinner.value

        current_image = det_data["data"][index]
        proj_v_line_source.data.update(
            x=np.arange(0, IMAGE_W) + 0.5, y=np.mean(current_image, axis=0)
        )
        proj_h_line_source.data.update(
            x=np.mean(current_image, axis=1), y=np.arange(0, IMAGE_H) + 0.5
        )

        image_source.data.update(
            h=[np.zeros((1, 1))], k=[np.zeros((1, 1))], l=[np.zeros((1, 1))],
        )
        image_source.data.update(image=[current_image])

        if auto_toggle.active:
            im_min = np.min(current_image)
            im_max = np.max(current_image)

            display_min_spinner.value = im_min
            display_max_spinner.value = im_max

            image_glyph.color_mapper.low = im_min
            image_glyph.color_mapper.high = im_max

        if "magnetic_field" in det_data:
            magnetic_field_spinner.value = det_data["magnetic_field"][index]
        else:
            magnetic_field_spinner.value = None

        if "temperature" in det_data:
            temperature_spinner.value = det_data["temperature"][index]
        else:
            temperature_spinner.value = None

        gamma, nu = calculate_pol(det_data, index)
        omega = np.ones((IMAGE_H, IMAGE_W)) * det_data["rot_angle"][index]
        image_source.data.update(gamma=[gamma], nu=[nu], omega=[omega])

    def update_overview_plot():
        h5_data = det_data["data"]
        n_im, n_y, n_x = h5_data.shape
        overview_x = np.mean(h5_data, axis=1)
        overview_y = np.mean(h5_data, axis=2)

        overview_plot_x_image_source.data.update(image=[overview_x], dw=[n_x])
        overview_plot_y_image_source.data.update(image=[overview_y], dw=[n_y])

        if proj_auto_toggle.active:
            im_min = min(np.min(overview_x), np.min(overview_y))
            im_max = max(np.max(overview_x), np.max(overview_y))

            proj_display_min_spinner.value = im_min
            proj_display_max_spinner.value = im_max

            overview_plot_x_image_glyph.color_mapper.low = im_min
            overview_plot_y_image_glyph.color_mapper.low = im_min
            overview_plot_x_image_glyph.color_mapper.high = im_max
            overview_plot_y_image_glyph.color_mapper.high = im_max

        if frame_button_group.active == 0:  # Frame
            overview_plot_x.axis[1].axis_label = "Frame"
            overview_plot_y.axis[1].axis_label = "Frame"

            overview_plot_x_image_source.data.update(y=[0], dh=[n_im])
            overview_plot_y_image_source.data.update(y=[0], dh=[n_im])

        elif frame_button_group.active == 1:  # Omega
            overview_plot_x.axis[1].axis_label = "Omega"
            overview_plot_y.axis[1].axis_label = "Omega"

            om = det_data["rot_angle"]
            om_start = om[0]
            om_end = (om[-1] - om[0]) * n_im / (n_im - 1)
            overview_plot_x_image_source.data.update(y=[om_start], dh=[om_end])
            overview_plot_y_image_source.data.update(y=[om_start], dh=[om_end])

    def filelist_callback(_attr, _old, new):
        nonlocal det_data
        det_data = pyzebra.read_detector_data(new)

        index_spinner.value = 0
        index_spinner.high = det_data["data"].shape[0] - 1
        update_image(0)
        update_overview_plot()

    filelist = Select()
    filelist.on_change("value", filelist_callback)

    def index_spinner_callback(_attr, _old, new):
        update_image(new)

    index_spinner = Spinner(title="Image index:", value=0, low=0)
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
            gamma=[np.zeros((1, 1))],
            nu=[np.zeros((1, 1))],
            omega=[np.zeros((1, 1))],
            x=[0],
            y=[0],
            dw=[IMAGE_W],
            dh=[IMAGE_H],
        )
    )

    h_glyph = Image(image="h", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
    k_glyph = Image(image="k", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
    l_glyph = Image(image="l", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
    gamma_glyph = Image(image="gamma", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
    nu_glyph = Image(image="nu", x="x", y="y", dw="dw", dh="dh", global_alpha=0)
    omega_glyph = Image(image="omega", x="x", y="y", dw="dw", dh="dh", global_alpha=0)

    plot.add_glyph(image_source, h_glyph)
    plot.add_glyph(image_source, k_glyph)
    plot.add_glyph(image_source, l_glyph)
    plot.add_glyph(image_source, gamma_glyph)
    plot.add_glyph(image_source, nu_glyph)
    plot.add_glyph(image_source, omega_glyph)

    image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
    plot.add_glyph(image_source, image_glyph, name="image_glyph")

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
    hovertool = HoverTool(
        tooltips=[
            ("intensity", "@image"),
            ("gamma", "@gamma"),
            ("nu", "@nu"),
            ("omega", "@omega"),
            ("h", "@h"),
            ("k", "@k"),
            ("l", "@l"),
        ]
    )

    box_edit_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
    box_edit_glyph = Rect(
        x="x", y="y", width="width", height="height", fill_alpha=0, line_color="red"
    )
    box_edit_renderer = plot.add_glyph(box_edit_source, box_edit_glyph)
    boxedittool = BoxEditTool(renderers=[box_edit_renderer], num_objects=1)

    def box_edit_callback(_attr, _old, new):
        if new["x"]:
            h5_data = det_data["data"]
            x_val = np.arange(h5_data.shape[0])
            left = int(np.floor(new["x"][0]))
            right = int(np.ceil(new["x"][0] + new["width"][0]))
            bottom = int(np.floor(new["y"][0]))
            top = int(np.ceil(new["y"][0] + new["height"][0]))
            y_val = np.sum(h5_data[:, bottom:top, left:right], axis=(1, 2))
        else:
            x_val = []
            y_val = []

        roi_avg_plot_line_source.data.update(x=x_val, y=y_val)

    box_edit_source.on_change("data", box_edit_callback)

    wheelzoomtool = WheelZoomTool(maintain_focus=False)
    plot.add_tools(
        PanTool(), BoxZoomTool(), wheelzoomtool, ResetTool(), hovertool, boxedittool,
    )
    plot.toolbar.active_scroll = wheelzoomtool

    # shared frame range
    frame_range = DataRange1d()
    det_x_range = Range1d(0, IMAGE_W, bounds=(0, IMAGE_W))
    overview_plot_x = Plot(
        title=Title(text="Projections on X-axis"),
        x_range=det_x_range,
        y_range=frame_range,
        plot_height=500,
        plot_width=IMAGE_W * 3,
    )

    # ---- tools
    wheelzoomtool = WheelZoomTool(maintain_focus=False)
    overview_plot_x.toolbar.logo = None
    overview_plot_x.add_tools(
        PanTool(), BoxZoomTool(), wheelzoomtool, ResetTool(),
    )
    overview_plot_x.toolbar.active_scroll = wheelzoomtool

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
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_W], dh=[1])
    )

    overview_plot_x_image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
    overview_plot_x.add_glyph(
        overview_plot_x_image_source, overview_plot_x_image_glyph, name="image_glyph"
    )

    det_y_range = Range1d(0, IMAGE_H, bounds=(0, IMAGE_H))
    overview_plot_y = Plot(
        title=Title(text="Projections on Y-axis"),
        x_range=det_y_range,
        y_range=frame_range,
        plot_height=500,
        plot_width=IMAGE_H * 3,
    )

    # ---- tools
    wheelzoomtool = WheelZoomTool(maintain_focus=False)
    overview_plot_y.toolbar.logo = None
    overview_plot_y.add_tools(
        PanTool(), BoxZoomTool(), wheelzoomtool, ResetTool(),
    )
    overview_plot_y.toolbar.active_scroll = wheelzoomtool

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
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_H], dh=[1])
    )

    overview_plot_y_image_glyph = Image(image="image", x="x", y="y", dw="dw", dh="dh")
    overview_plot_y.add_glyph(
        overview_plot_y_image_source, overview_plot_y_image_glyph, name="image_glyph"
    )

    def frame_button_group_callback(_active):
        update_overview_plot()

    frame_button_group = RadioButtonGroup(labels=["Frames", "Omega"], active=0)
    frame_button_group.on_click(frame_button_group_callback)

    roi_avg_plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=200,
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

    colormap = Select(title="Colormap:", options=list(cmap_dict.keys()), default_size=145)
    colormap.on_change("value", colormap_callback)
    colormap.value = "plasma"

    radio_button_group = RadioButtonGroup(labels=["normal beam", "bisecting"], active=0)

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

    auto_toggle = Toggle(
        label="Main Auto Range", active=True, button_type="default", default_size=125
    )
    auto_toggle.on_click(auto_toggle_callback)

    # ---- colormap display max value
    def display_max_spinner_callback(_attr, _old_value, new_value):
        display_min_spinner.high = new_value - STEP
        image_glyph.color_mapper.high = new_value

    display_max_spinner = Spinner(
        title="Max Value:",
        low=0 + STEP,
        value=1,
        step=STEP,
        disabled=auto_toggle.active,
        default_size=80,
    )
    display_max_spinner.on_change("value", display_max_spinner_callback)

    # ---- colormap display min value
    def display_min_spinner_callback(_attr, _old_value, new_value):
        display_max_spinner.low = new_value + STEP
        image_glyph.color_mapper.low = new_value

    display_min_spinner = Spinner(
        title="Min Value:",
        low=0,
        high=1 - STEP,
        value=0,
        step=STEP,
        disabled=auto_toggle.active,
        default_size=80,
    )
    display_min_spinner.on_change("value", display_min_spinner_callback)

    PROJ_STEP = 0.1
    # ---- proj colormap auto toggle button
    def proj_auto_toggle_callback(state):
        if state:
            proj_display_min_spinner.disabled = True
            proj_display_max_spinner.disabled = True
        else:
            proj_display_min_spinner.disabled = False
            proj_display_max_spinner.disabled = False

        update_overview_plot()

    proj_auto_toggle = Toggle(
        label="Proj Auto Range", active=True, button_type="default", default_size=125
    )
    proj_auto_toggle.on_click(proj_auto_toggle_callback)

    # ---- proj colormap display max value
    def proj_display_max_spinner_callback(_attr, _old_value, new_value):
        proj_display_min_spinner.high = new_value - PROJ_STEP
        overview_plot_x_image_glyph.color_mapper.high = new_value
        overview_plot_y_image_glyph.color_mapper.high = new_value

    proj_display_max_spinner = Spinner(
        title="Max Value:",
        low=0 + PROJ_STEP,
        value=1,
        step=PROJ_STEP,
        disabled=proj_auto_toggle.active,
        default_size=80,
    )
    proj_display_max_spinner.on_change("value", proj_display_max_spinner_callback)

    # ---- proj colormap display min value
    def proj_display_min_spinner_callback(_attr, _old_value, new_value):
        proj_display_max_spinner.low = new_value + PROJ_STEP
        overview_plot_x_image_glyph.color_mapper.low = new_value
        overview_plot_y_image_glyph.color_mapper.low = new_value

    proj_display_min_spinner = Spinner(
        title="Min Value:",
        low=0,
        high=1 - PROJ_STEP,
        value=0,
        step=PROJ_STEP,
        disabled=proj_auto_toggle.active,
        default_size=80,
    )
    proj_display_min_spinner.on_change("value", proj_display_min_spinner_callback)

    def hkl_button_callback():
        index = index_spinner.value
        geometry = "bi" if radio_button_group.active else "nb"
        h, k, l = calculate_hkl(det_data, index, geometry)
        image_source.data.update(h=[h], k=[k], l=[l])

    hkl_button = Button(label="Calculate hkl (slow)")
    hkl_button.on_click(hkl_button_callback)

    selection_list = TextAreaInput(rows=7)

    def selection_button_callback():
        nonlocal roi_selection
        selection = [
            int(np.floor(det_x_range.start)),
            int(np.ceil(det_x_range.end)),
            int(np.floor(det_y_range.start)),
            int(np.ceil(det_y_range.end)),
            int(np.floor(frame_range.start)),
            int(np.ceil(frame_range.end)),
        ]

        filename_id = filelist.value[-8:-4]
        if filename_id in roi_selection:
            roi_selection[f"{filename_id}"].append(selection)
        else:
            roi_selection[f"{filename_id}"] = [selection]

        selection_list.value = str(roi_selection)

    selection_button = Button(label="Add selection")
    selection_button.on_click(selection_button_callback)

    magnetic_field_spinner = Spinner(
        title="Magnetic field:", format="0.00", width=145, disabled=True
    )
    temperature_spinner = Spinner(title="Temperature:", format="0.00", width=145, disabled=True)

    # Final layout
    layout_image = column(gridplot([[proj_v, None], [plot, proj_h]], merge_tools=False))
    colormap_layout = column(
        row(colormap),
        row(column(Spacer(height=19), auto_toggle), display_max_spinner, display_min_spinner),
        row(
            column(Spacer(height=19), proj_auto_toggle),
            proj_display_max_spinner,
            proj_display_min_spinner,
        ),
    )
    geometry_div = Div(text="Geometry:", margin=[5, 5, -5, 5])
    hkl_layout = column(column(geometry_div, radio_button_group), hkl_button)
    params_layout = row(magnetic_field_spinner, temperature_spinner)

    layout_controls = row(
        column(selection_button, selection_list),
        Spacer(width=20),
        column(frame_button_group, colormap_layout),
        Spacer(width=20),
        column(index_spinner, params_layout, hkl_layout),
    )

    layout_overview = column(
        gridplot(
            [[overview_plot_x, overview_plot_y]],
            toolbar_options=dict(logo=None),
            merge_tools=True,
            toolbar_location="left",
        ),
    )

    upload_div = Div(text="Upload .cami file:")
    tab_layout = row(
        column(
            row(column(Spacer(height=5), upload_div), upload_button, filelist),
            layout_overview,
            layout_controls,
        ),
        column(roi_avg_plot, layout_image),
    )

    return Panel(child=tab_layout, title="hdf viewer")


def calculate_hkl(det_data, index, geometry):
    h = np.empty(shape=(IMAGE_H, IMAGE_W))
    k = np.empty(shape=(IMAGE_H, IMAGE_W))
    l = np.empty(shape=(IMAGE_H, IMAGE_W))

    wave = det_data["wave"]
    ddist = det_data["ddist"]
    gammad = det_data["pol_angle"][index]
    om = det_data["rot_angle"][index]
    nud = det_data["tlt_angle"]
    ub = det_data["UB"]

    if geometry == "bi":
        ch = det_data["chi_angle"][index]
        ph = det_data["phi_angle"][index]
    elif geometry == "nb":
        ch = 0
        ph = 0
    else:
        raise ValueError(f"Unknown geometry type '{geometry}'")

    for xi in np.arange(IMAGE_W):
        for yi in np.arange(IMAGE_H):
            h[yi, xi], k[yi, xi], l[yi, xi] = pyzebra.ang2hkl(
                wave, ddist, gammad, om, ch, ph, nud, ub, xi, yi
            )

    return h, k, l


def calculate_pol(det_data, index):
    gamma = np.empty(shape=(IMAGE_H, IMAGE_W))
    nu = np.empty(shape=(IMAGE_H, IMAGE_W))

    ddist = det_data["ddist"]
    gammad = det_data["pol_angle"][index]
    nud = det_data["tlt_angle"]

    for xi in np.arange(IMAGE_W):
        for yi in np.arange(IMAGE_H):
            gamma[yi, xi], nu[yi, xi] = pyzebra.det2pol(ddist, gammad, nud, xi, yi)

    return gamma, nu
