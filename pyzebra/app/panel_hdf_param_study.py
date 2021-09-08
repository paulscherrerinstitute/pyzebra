import base64
import io
import math
import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
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
    NumberEditor,
    NumberFormatter,
    Image,
    LinearAxis,
    LinearColorMapper,
    Panel,
    PanTool,
    Plot,
    Range1d,
    ResetTool,
    Scatter,
    Select,
    Spinner,
    TableColumn,
    Tabs,
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
    zebra_data = []
    det_data = {}
    cami_meta = {}

    num_formatter = NumberFormatter(format="0.00", nan_format="")

    def file_select_update():
        if data_source.value == "proposal number":
            proposal_path = proposal_textinput.name
            if proposal_path:
                file_list = []
                for file in os.listdir(proposal_path):
                    if file.endswith(".hdf"):
                        file_list.append((os.path.join(proposal_path, file), file))
                file_select.options = file_list
            else:
                file_select.options = []

        else:  # "cami file"
            if not cami_meta:
                file_select.options = []
                return

            file_list = cami_meta["filelist"]
            file_select.options = [(entry, os.path.basename(entry)) for entry in file_list]

    def data_source_callback(_attr, _old, _new):
        file_select_update()

    data_source = Select(
        title="Data Source:",
        value="proposal number",
        options=["proposal number", "cami file"],
        width=210,
    )
    data_source.on_change("value", data_source_callback)

    doc.add_periodic_callback(file_select_update, 5000)

    def proposal_textinput_callback(_attr, _old, _new):
        file_select_update()

    proposal_textinput = doc.proposal_textinput
    proposal_textinput.on_change("name", proposal_textinput_callback)

    def upload_button_callback(_attr, _old, new):
        nonlocal cami_meta
        with io.StringIO(base64.b64decode(new).decode()) as file:
            cami_meta = pyzebra.parse_h5meta(file)
        data_source.value = "cami file"
        file_select_update()

    upload_div = Div(text="or upload .cami file:", margin=(5, 5, 0, 5))
    upload_button = FileInput(accept=".cami", width=200)
    upload_button.on_change("value", upload_button_callback)

    file_select = MultiSelect(title="Available .hdf files:", width=210, height=320)

    def _init_datatable():
        file_list = []
        for scan in zebra_data:
            file_list.append(os.path.basename(scan["original_filename"]))

        scan_table_source.data.update(
            file=file_list,
            param=[None] * len(zebra_data),
            frame=[None] * len(zebra_data),
            x_pos=[None] * len(zebra_data),
            y_pos=[None] * len(zebra_data),
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        param_select.value = "user defined"

    def _update_table():
        frame = []
        x_pos = []
        y_pos = []
        for scan in zebra_data:
            if "fit" in scan:
                framei = scan["fit"]["frame"]
                x_posi = scan["fit"]["x_pos"]
                y_posi = scan["fit"]["y_pos"]
            else:
                framei = x_posi = y_posi = None

            frame.append(framei)
            x_pos.append(x_posi)
            y_pos.append(y_posi)

        scan_table_source.data.update(frame=frame, x_pos=x_pos, y_pos=y_pos)

    def file_open_button_callback():
        nonlocal zebra_data
        zebra_data = []
        for f_name in file_select.value:
            zebra_data.append(pyzebra.read_detector_data(f_name))

        _init_datatable()

    file_open_button = Button(label="Open New", width=100)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        for f_name in file_select.value:
            zebra_data.append(pyzebra.read_detector_data(f_name))

        _init_datatable()

    file_append_button = Button(label="Append", width=100)
    file_append_button.on_click(file_append_button_callback)

    # Scan select
    def scan_table_select_callback(_attr, old, new):
        nonlocal det_data

        if not new:
            # skip empty selections
            return

        # Avoid selection of multiple indicies (via Shift+Click or Ctrl+Click)
        if len(new) > 1:
            # drop selection to the previous one
            scan_table_source.selected.indices = old
            return

        if len(old) > 1:
            # skip unnecessary update caused by selection drop
            return

        det_data = zebra_data[new[0]]

        zebra_mode = det_data["zebra_mode"]
        if zebra_mode == "nb":
            metadata_table_source.data.update(geom=["normal beam"])
        else:  # zebra_mode == "bi"
            metadata_table_source.data.update(geom=["bisecting"])

        if "mf" in det_data:
            metadata_table_source.data.update(mf=[det_data["mf"][0]])
        else:
            metadata_table_source.data.update(mf=[None])

        if "temp" in det_data:
            metadata_table_source.data.update(temp=[det_data["temp"][0]])
        else:
            metadata_table_source.data.update(temp=[None])

        update_overview_plot()

    def scan_table_source_callback(_attr, _old, _new):
        pass

    scan_table_source = ColumnDataSource(dict(file=[], param=[], frame=[], x_pos=[], y_pos=[]))
    scan_table_source.selected.on_change("indices", scan_table_select_callback)
    scan_table_source.on_change("data", scan_table_source_callback)

    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="file", title="file", width=150),
            TableColumn(
                field="param",
                title="param",
                formatter=num_formatter,
                editor=NumberEditor(),
                width=50,
            ),
            TableColumn(field="frame", title="Frame", formatter=num_formatter, width=70),
            TableColumn(field="x_pos", title="X", formatter=num_formatter, width=70),
            TableColumn(field="y_pos", title="Y", formatter=num_formatter, width=70),
        ],
        width=470,  # +60 because of the index column
        height=420,
        editable=True,
        autosize_mode="none",
    )

    def param_select_callback(_attr, _old, new):
        if new == "user defined":
            param = [None] * len(zebra_data)
        else:
            # TODO: which value to take?
            param = [scan[new][0] for scan in zebra_data]

        scan_table_source.data["param"] = param
        _update_param_plot()

    param_select = Select(
        title="Parameter:",
        options=["user defined", "temp", "mf", "h", "k", "l"],
        value="user defined",
        width=145,
    )
    param_select.on_change("value", param_select_callback)

    def update_overview_plot():
        h5_data = det_data["data"]
        n_im, n_y, n_x = h5_data.shape
        overview_x = np.mean(h5_data, axis=1)
        overview_y = np.mean(h5_data, axis=2)

        # normalize for simpler colormapping
        overview_max_val = max(np.max(overview_x), np.max(overview_y))
        overview_x = 1000 * overview_x / overview_max_val
        overview_y = 1000 * overview_y / overview_max_val

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

    cmap_dict = {
        "gray": Greys256,
        "gray_reversed": Greys256[::-1],
        "plasma": Plasma256,
        "cividis": Cividis256,
    }

    def colormap_callback(_attr, _old, new):
        overview_plot_x_image_glyph.color_mapper = LinearColorMapper(palette=cmap_dict[new])
        overview_plot_y_image_glyph.color_mapper = LinearColorMapper(palette=cmap_dict[new])

    colormap = Select(title="Colormap:", options=list(cmap_dict.keys()), width=210)
    colormap.on_change("value", colormap_callback)
    colormap.value = "plasma"

    PROJ_STEP = 1

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

    def fit_event(scan):
        p0 = [1.0, 0.0, 1.0]
        maxfev = 100000

        # wave = scan["wave"]
        # ddist = scan["ddist"]
        # cell = scan["cell"]

        # gamma = scan["gamma"][0]
        # omega = scan["omega"][0]
        # nu = scan["nu"][0]
        # chi = scan["chi"][0]
        # phi = scan["phi"][0]

        scan_motor = scan["scan_motor"]
        var_angle = scan[scan_motor]

        x0 = int(np.floor(det_x_range.start))
        xN = int(np.ceil(det_x_range.end))
        y0 = int(np.floor(det_y_range.start))
        yN = int(np.ceil(det_y_range.end))
        fr0 = int(np.floor(frame_range.start))
        frN = int(np.ceil(frame_range.end))
        data_roi = scan["data"][fr0:frN, y0:yN, x0:xN]

        cnts = np.sum(data_roi, axis=(1, 2))
        coeff, _ = curve_fit(gauss, range(len(cnts)), cnts, p0=p0, maxfev=maxfev)

        # m = cnts.mean()
        # sd = cnts.std()
        # snr_cnts = np.where(sd == 0, 0, m / sd)

        frC = fr0 + coeff[1]
        var_F = var_angle[math.floor(frC)]
        var_C = var_angle[math.ceil(frC)]
        # frStep = frC - math.floor(frC)
        var_step = var_C - var_F
        # var_p = var_F + var_step * frStep

        # if scan_motor == "gamma":
        #     gamma = var_p
        # elif scan_motor == "omega":
        #     omega = var_p
        # elif scan_motor == "nu":
        #     nu = var_p
        # elif scan_motor == "chi":
        #     chi = var_p
        # elif scan_motor == "phi":
        #     phi = var_p

        intensity = coeff[1] * abs(coeff[2] * var_step) * math.sqrt(2) * math.sqrt(np.pi)

        projX = np.sum(data_roi, axis=(0, 1))
        coeff, _ = curve_fit(gauss, range(len(projX)), projX, p0=p0, maxfev=maxfev)
        x_pos = x0 + coeff[1]

        projY = np.sum(data_roi, axis=(0, 2))
        coeff, _ = curve_fit(gauss, range(len(projY)), projY, p0=p0, maxfev=maxfev)
        y_pos = y0 + coeff[1]

        scan["fit"] = {"frame": frC, "x_pos": x_pos, "y_pos": y_pos, "intensity": intensity}

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

    def _update_param_plot():
        x = []
        y = []
        fit_param = fit_param_select.value
        for s, p in zip(zebra_data, scan_table_source.data["param"]):
            if "fit" in s and fit_param:
                x.append(p)
                y.append(s["fit"][fit_param])
        param_plot_scatter_source.data.update(x=x, y=y)

    # Parameter plot
    param_plot = Plot(x_range=DataRange1d(), y_range=DataRange1d(), plot_height=400, plot_width=700)

    param_plot.add_layout(LinearAxis(axis_label="Fit parameter"), place="left")
    param_plot.add_layout(LinearAxis(axis_label="Parameter"), place="below")

    param_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
    param_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

    param_plot_scatter_source = ColumnDataSource(dict(x=[], y=[]))
    param_plot.add_glyph(param_plot_scatter_source, Scatter(x="x", y="y"))

    param_plot.add_tools(PanTool(), WheelZoomTool(), ResetTool())
    param_plot.toolbar.logo = None

    def fit_param_select_callback(_attr, _old, _new):
        _update_param_plot()

    fit_param_select = Select(title="Fit parameter", options=[], width=145)
    fit_param_select.on_change("value", fit_param_select_callback)

    def proc_all_button_callback():
        for scan in zebra_data:
            fit_event(scan)

        _update_table()

        for scan in zebra_data:
            if "fit" in scan:
                options = list(scan["fit"].keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

        _update_param_plot()

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
        fit_event(det_data)

        _update_table()

        for scan in zebra_data:
            if "fit" in scan:
                options = list(scan["fit"].keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

        _update_param_plot()

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    layout_controls = row(
        colormap,
        column(proj_auto_checkbox, row(proj_display_min_spinner, proj_display_max_spinner)),
        proc_button,
        proc_all_button,
    )

    layout_overview = column(
        gridplot(
            [[overview_plot_x, overview_plot_y]],
            toolbar_options=dict(logo=None),
            merge_tools=True,
            toolbar_location="left",
        ),
        layout_controls,
    )

    # Plot tabs
    plots = Tabs(
        tabs=[
            Panel(child=layout_overview, title="single scan"),
            Panel(child=column(param_plot, row(fit_param_select)), title="parameter plot"),
        ]
    )

    # Final layout
    import_layout = column(
        data_source,
        upload_div,
        upload_button,
        file_select,
        row(file_open_button, file_append_button),
    )

    scan_layout = column(scan_table, row(param_select, metadata_table))

    tab_layout = column(row(import_layout, scan_layout, plots))

    return Panel(child=tab_layout, title="hdf param study")


def gauss(x, *p):
    """Defines Gaussian function
    Args:
        A - amplitude, mu - position of the center, sigma - width
    Returns:
        Gaussian function
    """
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
