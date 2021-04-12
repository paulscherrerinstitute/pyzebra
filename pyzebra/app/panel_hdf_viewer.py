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
    CheckboxGroup,
    ColumnDataSource,
    DataRange1d,
    Div,
    FileInput,
    Grid,
    MultiSelect,
    HoverTool,
    Image,
    Line,
    LinearAxis,
    LinearColorMapper,
    Panel,
    PanTool,
    Plot,
    Range1d,
    Rect,
    ResetTool,
    Select,
    Spacer,
    Spinner,
    TextAreaInput,
    TextInput,
    Title,
    WheelZoomTool,
)
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611

import pyzebra

IMAGE_W = 256
IMAGE_H = 128
IMAGE_PLOT_W = int(IMAGE_W * 2) + 52
IMAGE_PLOT_H = int(IMAGE_H * 2) + 27


def create():
    det_data = {}
    roi_selection = {}

    def proposal_textinput_callback(_attr, _old, new):
        proposal = new.strip()
        year = new[:4]
        proposal_path = f"/afs/psi.ch/project/sinqdata/{year}/zebra/{proposal}"
        file_list = []
        for file in os.listdir(proposal_path):
            if file.endswith(".hdf"):
                file_list.append((os.path.join(proposal_path, file), file))
        file_select.options = file_list

    proposal_textinput = TextInput(title="Proposal number:", width=210)
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def upload_button_callback(_attr, _old, new):
        with io.StringIO(base64.b64decode(new).decode()) as file:
            h5meta_list = pyzebra.parse_h5meta(file)
            file_list = h5meta_list["filelist"]
            file_select.options = [(entry, os.path.basename(entry)) for entry in file_list]

    upload_div = Div(text="or upload .cami file:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".cami", width=200)
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

        if main_auto_checkbox.active:
            im_min = np.min(current_image)
            im_max = np.max(current_image)

            display_min_spinner.value = im_min
            display_max_spinner.value = im_max

            image_glyph.color_mapper.low = im_min
            image_glyph.color_mapper.high = im_max

        if "mf" in det_data:
            mf_spinner.value = det_data["mf"][index]
        else:
            mf_spinner.value = None

        if "temp" in det_data:
            temp_spinner.value = det_data["temp"][index]
        else:
            temp_spinner.value = None

        gamma, nu = calculate_pol(det_data, index)
        omega = np.ones((IMAGE_H, IMAGE_W)) * det_data["omega"][index]
        image_source.data.update(gamma=[gamma], nu=[nu], omega=[omega])

    def update_overview_plot():
        h5_data = det_data["data"]
        n_im, n_y, n_x = h5_data.shape
        overview_x = np.mean(h5_data, axis=1)
        overview_y = np.mean(h5_data, axis=2)

        overview_plot_x_image_source.data.update(image=[overview_x], dw=[n_x], dh=[n_im])
        overview_plot_y_image_source.data.update(image=[overview_y], dw=[n_y], dh=[n_im])

        if proj_auto_checkbox.active:
            im_min = min(np.min(overview_x), np.min(overview_y))
            im_max = max(np.max(overview_x), np.max(overview_y))

            proj_display_min_spinner.value = im_min
            proj_display_max_spinner.value = im_max

            overview_plot_x_image_glyph.color_mapper.low = im_min
            overview_plot_y_image_glyph.color_mapper.low = im_min
            overview_plot_x_image_glyph.color_mapper.high = im_max
            overview_plot_y_image_glyph.color_mapper.high = im_max

        frame_range.start = 0
        frame_range.end = n_im
        frame_range.reset_start = 0
        frame_range.reset_end = n_im
        frame_range.bounds = (0, n_im)

        scan_motor = det_data["scan_motor"]
        overview_plot_y.axis[1].axis_label = f"Scanning motor, {scan_motor}"

        var = det_data[scan_motor]
        var_start = var[0]
        var_end = var[-1] + (var[-1] - var[0]) / (n_im - 1)

        scanning_motor_range.start = var_start
        scanning_motor_range.end = var_end
        scanning_motor_range.reset_start = var_start
        scanning_motor_range.reset_end = var_end
        scanning_motor_range.bounds = (var_start, var_end)

    def file_select_callback(_attr, old, new):
        nonlocal det_data
        if not new:
            # skip empty selections
            return

        # Avoid selection of multiple indicies (via Shift+Click or Ctrl+Click)
        if len(new) > 1:
            # drop selection to the previous one
            file_select.value = old
            return

        if len(old) > 1:
            # skip unnecessary update caused by selection drop
            return

        det_data = pyzebra.read_detector_data(new[0])

        index_spinner.value = 0
        index_spinner.high = det_data["data"].shape[0] - 1

        zebra_mode = det_data["zebra_mode"]
        if zebra_mode == "nb":
            geometry_textinput.value = "normal beam"
        else:  # zebra_mode == "bi"
            geometry_textinput.value = "bisecting"

        update_image(0)
        update_overview_plot()

    file_select = MultiSelect(title="Available .hdf files:", width=210, height=250)
    file_select.on_change("value", file_select_callback)

    def index_spinner_callback(_attr, _old, new):
        update_image(new)

    index_spinner = Spinner(title="Image index:", value=0, low=0, width=80)
    index_spinner.on_change("value", index_spinner_callback)

    plot = Plot(
        x_range=Range1d(0, IMAGE_W, bounds=(0, IMAGE_W)),
        y_range=Range1d(0, IMAGE_H, bounds=(0, IMAGE_H)),
        plot_height=IMAGE_PLOT_H,
        plot_width=IMAGE_PLOT_W,
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
        plot_height=150,
        plot_width=IMAGE_PLOT_W,
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
        plot_height=IMAGE_PLOT_H,
        plot_width=150,
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

    # shared frame ranges
    frame_range = Range1d(0, 1, bounds=(0, 1))
    scanning_motor_range = Range1d(0, 1, bounds=(0, 1))

    det_x_range = Range1d(0, IMAGE_W, bounds=(0, IMAGE_W))
    overview_plot_x = Plot(
        title=Title(text="Projections on X-axis"),
        x_range=det_x_range,
        y_range=frame_range,
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        plot_height=400,
        plot_width=IMAGE_PLOT_W - 3,
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
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        plot_height=400,
        plot_width=IMAGE_PLOT_H + 22,
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
        LinearAxis(
            y_range_name="scanning_motor",
            axis_label="Scanning motor",
            major_label_orientation="vertical",
        ),
        place="right",
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

    roi_avg_plot = Plot(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=150,
        plot_width=IMAGE_PLOT_W,
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

    colormap = Select(title="Colormap:", options=list(cmap_dict.keys()), width=210)
    colormap.on_change("value", colormap_callback)
    colormap.value = "plasma"

    STEP = 1

    def main_auto_checkbox_callback(state):
        if state:
            display_min_spinner.disabled = True
            display_max_spinner.disabled = True
        else:
            display_min_spinner.disabled = False
            display_max_spinner.disabled = False

        update_image()

    main_auto_checkbox = CheckboxGroup(
        labels=["Main Auto Range"], active=[0], width=145, margin=[10, 5, 0, 5]
    )
    main_auto_checkbox.on_click(main_auto_checkbox_callback)

    def display_max_spinner_callback(_attr, _old_value, new_value):
        display_min_spinner.high = new_value - STEP
        image_glyph.color_mapper.high = new_value

    display_max_spinner = Spinner(
        low=0 + STEP,
        value=1,
        step=STEP,
        disabled=bool(main_auto_checkbox.active),
        width=100,
        height=31,
    )
    display_max_spinner.on_change("value", display_max_spinner_callback)

    def display_min_spinner_callback(_attr, _old_value, new_value):
        display_max_spinner.low = new_value + STEP
        image_glyph.color_mapper.low = new_value

    display_min_spinner = Spinner(
        low=0,
        high=1 - STEP,
        value=0,
        step=STEP,
        disabled=bool(main_auto_checkbox.active),
        width=100,
        height=31,
    )
    display_min_spinner.on_change("value", display_min_spinner_callback)

    PROJ_STEP = 0.1

    def proj_auto_checkbox_callback(state):
        if state:
            proj_display_min_spinner.disabled = True
            proj_display_max_spinner.disabled = True
        else:
            proj_display_min_spinner.disabled = False
            proj_display_max_spinner.disabled = False

        update_overview_plot()

    proj_auto_checkbox = CheckboxGroup(
        labels=["Projections Auto Range"], active=[0], width=145, margin=[10, 5, 0, 5]
    )
    proj_auto_checkbox.on_click(proj_auto_checkbox_callback)

    def proj_display_max_spinner_callback(_attr, _old_value, new_value):
        proj_display_min_spinner.high = new_value - PROJ_STEP
        overview_plot_x_image_glyph.color_mapper.high = new_value
        overview_plot_y_image_glyph.color_mapper.high = new_value

    proj_display_max_spinner = Spinner(
        low=0 + PROJ_STEP,
        value=1,
        step=PROJ_STEP,
        disabled=bool(proj_auto_checkbox.active),
        width=100,
        height=31,
    )
    proj_display_max_spinner.on_change("value", proj_display_max_spinner_callback)

    def proj_display_min_spinner_callback(_attr, _old_value, new_value):
        proj_display_max_spinner.low = new_value + PROJ_STEP
        overview_plot_x_image_glyph.color_mapper.low = new_value
        overview_plot_y_image_glyph.color_mapper.low = new_value

    proj_display_min_spinner = Spinner(
        low=0,
        high=1 - PROJ_STEP,
        value=0,
        step=PROJ_STEP,
        disabled=bool(proj_auto_checkbox.active),
        width=100,
        height=31,
    )
    proj_display_min_spinner.on_change("value", proj_display_min_spinner_callback)

    def hkl_button_callback():
        index = index_spinner.value
        h, k, l = calculate_hkl(det_data, index)
        image_source.data.update(h=[h], k=[k], l=[l])

    hkl_button = Button(label="Calculate hkl (slow)", width=210)
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

        filename_id = file_select.value[0][-8:-4]
        if filename_id in roi_selection:
            roi_selection[f"{filename_id}"].append(selection)
        else:
            roi_selection[f"{filename_id}"] = [selection]

        selection_list.value = str(roi_selection)

    selection_button = Button(label="Add selection")
    selection_button.on_click(selection_button_callback)

    mf_spinner = Spinner(title="Magnetic field:", format="0.00", width=100, disabled=True)
    temp_spinner = Spinner(title="Temperature:", format="0.00", width=100, disabled=True)
    geometry_textinput = TextInput(title="Geometry:", width=120, disabled=True)

    # Final layout
    import_layout = column(proposal_textinput, upload_div, upload_button, file_select)
    layout_image = column(gridplot([[proj_v, None], [plot, proj_h]], merge_tools=False))
    colormap_layout = column(
        colormap,
        main_auto_checkbox,
        row(display_min_spinner, display_max_spinner),
        proj_auto_checkbox,
        row(proj_display_min_spinner, proj_display_max_spinner),
    )

    layout_controls = row(
        column(selection_button, selection_list),
        column(row(mf_spinner, temp_spinner), row(geometry_textinput, index_spinner), hkl_button),
    )

    layout_overview = column(
        gridplot(
            [[overview_plot_x, overview_plot_y]],
            toolbar_options=dict(logo=None),
            merge_tools=True,
            toolbar_location="left",
        ),
    )

    tab_layout = row(
        column(import_layout, colormap_layout),
        column(layout_overview, layout_controls),
        column(roi_avg_plot, layout_image),
    )

    return Panel(child=tab_layout, title="hdf viewer")


def calculate_hkl(det_data, index):
    h = np.empty(shape=(IMAGE_H, IMAGE_W))
    k = np.empty(shape=(IMAGE_H, IMAGE_W))
    l = np.empty(shape=(IMAGE_H, IMAGE_W))

    wave = det_data["wave"]
    ddist = det_data["ddist"]
    gammad = det_data["gamma"][index]
    om = det_data["omega"][index]
    nud = det_data["nu"]
    ub = det_data["ub"]
    geometry = det_data["zebra_mode"]

    if geometry == "bi":
        chi = det_data["chi"][index]
        phi = det_data["phi"][index]
    elif geometry == "nb":
        chi = 0
        phi = 0
    else:
        raise ValueError(f"Unknown geometry type '{geometry}'")

    for xi in np.arange(IMAGE_W):
        for yi in np.arange(IMAGE_H):
            h[yi, xi], k[yi, xi], l[yi, xi] = pyzebra.ang2hkl(
                wave, ddist, gammad, om, chi, phi, nud, ub, xi, yi
            )

    return h, k, l


def calculate_pol(det_data, index):
    gamma = np.empty(shape=(IMAGE_H, IMAGE_W))
    nu = np.empty(shape=(IMAGE_H, IMAGE_W))

    ddist = det_data["ddist"]
    gammad = det_data["gamma"][index]
    nud = det_data["nu"]

    for xi in np.arange(IMAGE_W):
        for yi in np.arange(IMAGE_H):
            gamma[yi, xi], nu[yi, xi] = pyzebra.det2pol(ddist, gammad, nud, xi, yi)

    return gamma, nu
