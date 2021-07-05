import base64
import io
import math
import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    BoxEditTool,
    BoxZoomTool,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    DataRange1d,
    DataTable,
    Div,
    FileInput,
    Grid,
    MultiSelect,
    NumberFormatter,
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
    Slider,
    Spacer,
    Spinner,
    TableColumn,
    TextInput,
    Title,
    WheelZoomTool,
)
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611
from scipy.optimize import curve_fit

import pyzebra

IMAGE_W = 256
IMAGE_H = 128
IMAGE_PLOT_W = int(IMAGE_W * 2) + 52
IMAGE_PLOT_H = int(IMAGE_H * 2) + 27


def create():
    doc = curdoc()
    det_data = {}
    cami_meta = {}

    num_formatter = NumberFormatter(format="0.00", nan_format="")

    def file_select_update_for_proposal():
        proposal = proposal_textinput.value.strip()
        if not proposal:
            return

        for zebra_proposals_path in pyzebra.ZEBRA_PROPOSALS_PATHS:
            proposal_path = os.path.join(zebra_proposals_path, proposal)
            if os.path.isdir(proposal_path):
                # found it
                break
        else:
            raise ValueError(f"Can not find data for proposal '{proposal}'.")

        file_list = []
        for file in os.listdir(proposal_path):
            if file.endswith(".hdf"):
                file_list.append((os.path.join(proposal_path, file), file))
        file_select.options = file_list

    doc.add_periodic_callback(file_select_update_for_proposal, 5000)

    def proposal_textinput_callback(_attr, _old, _new):
        nonlocal cami_meta
        cami_meta = {}
        file_select_update_for_proposal()

    proposal_textinput = TextInput(title="Proposal number:", width=210)
    proposal_textinput.on_change("value", proposal_textinput_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal cami_meta
        proposal_textinput.value = ""
        with io.StringIO(base64.b64decode(new).decode()) as file:
            cami_meta = pyzebra.parse_h5meta(file)
            file_list = cami_meta["filelist"]
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
            metadata_table_source.data.update(mf=[det_data["mf"][index]])
        else:
            metadata_table_source.data.update(mf=[None])

        if "temp" in det_data:
            metadata_table_source.data.update(temp=[det_data["temp"][index]])
        else:
            metadata_table_source.data.update(temp=[None])

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
        # handle both, ascending and descending sequences
        scanning_motor_range.bounds = (min(var_start, var_end), max(var_start, var_end))

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

        det_data = pyzebra.read_detector_data(new[0], cami_meta)

        index_spinner.value = 0
        index_spinner.high = det_data["data"].shape[0] - 1
        index_slider.end = det_data["data"].shape[0] - 1

        zebra_mode = det_data["zebra_mode"]
        if zebra_mode == "nb":
            metadata_table_source.data.update(geom=["normal beam"])
        else:  # zebra_mode == "bi"
            metadata_table_source.data.update(geom=["bisecting"])

        update_image(0)
        update_overview_plot()

    file_select = MultiSelect(title="Available .hdf files:", width=210, height=250)
    file_select.on_change("value", file_select_callback)

    def index_callback(_attr, _old, new):
        update_image(new)

    index_slider = Slider(value=0, start=0, end=1, show_value=False, width=400)

    index_spinner = Spinner(title="Image index:", value=0, low=0, width=100)
    index_spinner.on_change("value", index_callback)

    index_slider.js_link("value_throttled", index_spinner, "value")
    index_spinner.js_link("value", index_slider, "value")

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
        labels=["Frame Intensity Range"], active=[0], width=145, margin=[10, 5, 0, 5]
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
        labels=["Projections Intensity Range"], active=[0], width=145, margin=[10, 5, 0, 5]
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

    hkl_button = Button(label="Calculate hkl (slow)", width=145)
    hkl_button.on_click(hkl_button_callback)

    events_data = dict(
        wave=[],
        ddist=[],
        cell=[],
        frame=[],
        x_pos=[],
        y_pos=[],
        intensity=[],
        snr_cnts=[],
        gamma=[],
        omega=[],
        chi=[],
        phi=[],
        nu=[],
    )
    doc.events_data = events_data

    events_table_source = ColumnDataSource(events_data)
    events_table = DataTable(
        source=events_table_source,
        columns=[
            TableColumn(field="frame", title="Frame", formatter=num_formatter, width=70),
            TableColumn(field="x_pos", title="X", formatter=num_formatter, width=70),
            TableColumn(field="y_pos", title="Y", formatter=num_formatter, width=70),
            TableColumn(field="intensity", title="Intensity", formatter=num_formatter, width=70),
            TableColumn(field="gamma", title="Gamma", formatter=num_formatter, width=70),
            TableColumn(field="omega", title="Omega", formatter=num_formatter, width=70),
            TableColumn(field="chi", title="Chi", formatter=num_formatter, width=70),
            TableColumn(field="phi", title="Phi", formatter=num_formatter, width=70),
            TableColumn(field="nu", title="Nu", formatter=num_formatter, width=70),
        ],
        height=150,
        width=630,
        autosize_mode="none",
        index_position=None,
    )

    def add_event_button_callback():
        p0 = [1.0, 0.0, 1.0]
        maxfev = 100000

        wave = det_data["wave"]
        ddist = det_data["ddist"]
        cell = det_data["cell"]

        gamma = det_data["gamma"][0]
        omega = det_data["omega"][0]
        nu = det_data["nu"][0]
        chi = det_data["chi"][0]
        phi = det_data["phi"][0]

        scan_motor = det_data["scan_motor"]
        var_angle = det_data[scan_motor]

        x0 = int(np.floor(det_x_range.start))
        xN = int(np.ceil(det_x_range.end))
        y0 = int(np.floor(det_y_range.start))
        yN = int(np.ceil(det_y_range.end))
        fr0 = int(np.floor(frame_range.start))
        frN = int(np.ceil(frame_range.end))
        data_roi = det_data["data"][fr0:frN, y0:yN, x0:xN]

        cnts = np.sum(data_roi, axis=(1, 2))
        coeff, _ = curve_fit(gauss, range(len(cnts)), cnts, p0=p0, maxfev=maxfev)

        m = cnts.mean()
        sd = cnts.std()
        snr_cnts = np.where(sd == 0, 0, m / sd)

        frC = fr0 + coeff[1]
        var_F = var_angle[math.floor(frC)]
        var_C = var_angle[math.ceil(frC)]
        frStep = frC - math.floor(frC)
        var_step = var_C - var_F
        var_p = var_F + var_step * frStep

        if scan_motor == "gamma":
            gamma = var_p
        elif scan_motor == "omega":
            omega = var_p
        elif scan_motor == "nu":
            nu = var_p
        elif scan_motor == "chi":
            chi = var_p
        elif scan_motor == "phi":
            phi = var_p

        intensity = coeff[1] * abs(coeff[2] * var_step) * math.sqrt(2) * math.sqrt(np.pi)

        projX = np.sum(data_roi, axis=(0, 1))
        coeff, _ = curve_fit(gauss, range(len(projX)), projX, p0=p0, maxfev=maxfev)
        x_pos = x0 + coeff[1]

        projY = np.sum(data_roi, axis=(0, 2))
        coeff, _ = curve_fit(gauss, range(len(projY)), projY, p0=p0, maxfev=maxfev)
        y_pos = y0 + coeff[1]

        events_data["wave"].append(wave)
        events_data["ddist"].append(ddist)
        events_data["cell"].append(cell)
        events_data["frame"].append(frC)
        events_data["x_pos"].append(x_pos)
        events_data["y_pos"].append(y_pos)
        events_data["intensity"].append(intensity)
        events_data["snr_cnts"].append(snr_cnts)
        events_data["gamma"].append(gamma)
        events_data["omega"].append(omega)
        events_data["chi"].append(chi)
        events_data["phi"].append(phi)
        events_data["nu"].append(nu)

        events_table_source.data = events_data

    add_event_button = Button(label="Add spind event", width=145)
    add_event_button.on_click(add_event_button_callback)

    def remove_event_button_callback():
        ind2remove = events_table_source.selected.indices
        for value in events_data.values():
            for ind in reversed(ind2remove):
                del value[ind]

        events_table_source.data = events_data

    remove_event_button = Button(label="Remove spind event", width=145)
    remove_event_button.on_click(remove_event_button_callback)

    metadata_table_source = ColumnDataSource(dict(geom=[""], temp=[None], mf=[None]))
    metadata_table = DataTable(
        source=metadata_table_source,
        columns=[
            TableColumn(field="geom", title="Geometry", width=100),
            TableColumn(field="temp", title="Temperature", formatter=num_formatter, width=100),
            TableColumn(field="mf", title="Magnetic Field", formatter=num_formatter, width=100),
        ],
        width=300,
        height=50,
        autosize_mode="none",
        index_position=None,
    )

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

    layout_controls = column(
        row(metadata_table, index_spinner, column(Spacer(height=25), index_slider)),
        row(column(add_event_button, remove_event_button), events_table),
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
        column(roi_avg_plot, layout_image, hkl_button),
    )

    return Panel(child=tab_layout, title="hdf viewer")


def gauss(x, *p):
    """Defines Gaussian function
    Args:
        A - amplitude, mu - position of the center, sigma - width
    Returns:
        Gaussian function
    """
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


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
