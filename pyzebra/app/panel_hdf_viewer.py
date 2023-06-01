import base64
import io
import os

import numpy as np
from bokeh.events import MouseEnter
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BoxEditTool,
    Button,
    CellEditor,
    CheckboxGroup,
    ColumnDataSource,
    DataTable,
    Div,
    FileInput,
    HoverTool,
    LinearAxis,
    LinearColorMapper,
    LogColorMapper,
    MultiSelect,
    NumberFormatter,
    Panel,
    RadioGroup,
    Range1d,
    Select,
    Slider,
    Spacer,
    Spinner,
    TableColumn,
    Tabs,
)
from bokeh.plotting import figure

import pyzebra

IMAGE_W = 256
IMAGE_H = 128
IMAGE_PLOT_W = int(IMAGE_W * 2.4) + 52
IMAGE_PLOT_H = int(IMAGE_H * 2.4) + 27


def create():
    doc = curdoc()
    dataset = []
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

    def upload_cami_button_callback(_attr, _old, new):
        nonlocal cami_meta
        with io.StringIO(base64.b64decode(new).decode()) as file:
            cami_meta = pyzebra.parse_h5meta(file)
        data_source.value = "cami file"
        file_select_update()

    upload_cami_div = Div(text="or upload .cami file:", margin=(5, 5, 0, 5))
    upload_cami_button = FileInput(accept=".cami", width=200)
    upload_cami_button.on_change("value", upload_cami_button_callback)

    def upload_hdf_button_callback(_attr, _old, new):
        nonlocal dataset
        try:
            scan = pyzebra.read_detector_data(io.BytesIO(base64.b64decode(new)), None)
        except KeyError:
            print("Could not read data from the file.")
            return

        dataset = [scan]
        last_im_index = scan["counts"].shape[0] - 1

        index_spinner.value = 0
        index_spinner.high = last_im_index
        if last_im_index == 0:
            index_slider.disabled = True
        else:
            index_slider.disabled = False
            index_slider.end = last_im_index

        zebra_mode = scan["zebra_mode"]
        if zebra_mode == "nb":
            metadata_table_source.data.update(geom=["normal beam"])
        else:  # zebra_mode == "bi"
            metadata_table_source.data.update(geom=["bisecting"])

        _init_datatable()

    upload_hdf_div = Div(text="or upload .hdf file:", margin=(5, 5, 0, 5))
    upload_hdf_button = FileInput(accept=".hdf", width=200)
    upload_hdf_button.on_change("value", upload_hdf_button_callback)

    def file_open_button_callback():
        nonlocal dataset
        new_data = []
        cm = cami_meta if data_source.value == "cami file" else None
        for f_path in file_select.value:
            f_name = os.path.basename(f_path)
            try:
                file_data = [pyzebra.read_detector_data(f_path, cm)]
            except:
                print(f"Error loading {f_name}")
                continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)

            if not new_data:  # first file
                new_data = file_data
            else:
                pyzebra.merge_datasets(new_data, file_data)

        if new_data:
            dataset = new_data
            _init_datatable()

    file_open_button = Button(label="Open New", width=100)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        file_data = []
        for f_path in file_select.value:
            f_name = os.path.basename(f_path)
            try:
                file_data = [pyzebra.read_detector_data(f_path, None)]
            except:
                print(f"Error loading {f_name}")
                continue

            pyzebra.normalize_dataset(file_data, monitor_spinner.value)
            pyzebra.merge_datasets(dataset, file_data)

        if file_data:
            _init_datatable()

    file_append_button = Button(label="Append", width=100)
    file_append_button.on_click(file_append_button_callback)

    def _init_datatable():
        scan_list = [s["idx"] for s in dataset]
        export = [s["export"] for s in dataset]

        twotheta = [np.median(s["twotheta"]) if "twotheta" in s else None for s in dataset]
        gamma = [np.median(s["gamma"]) if "gamma" in s else None for s in dataset]
        omega = [np.median(s["omega"]) if "omega" in s else None for s in dataset]
        chi = [np.median(s["chi"]) if "chi" in s else None for s in dataset]
        phi = [np.median(s["phi"]) if "phi" in s else None for s in dataset]
        nu = [np.median(s["nu"]) if "nu" in s else None for s in dataset]

        scan_table_source.data.update(
            scan=scan_list,
            fit=[0] * len(scan_list),
            export=export,
            twotheta=twotheta,
            gamma=gamma,
            omega=omega,
            chi=chi,
            phi=phi,
            nu=nu,
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        merge_options = [(str(i), f"{i} ({idx})") for i, idx in enumerate(scan_list)]
        merge_from_select.options = merge_options
        merge_from_select.value = merge_options[0][0]

    def scan_table_select_callback(_attr, old, new):
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

        scan = _get_selected_scan()
        last_im_index = scan["counts"].shape[0] - 1

        index_spinner.value = 0
        index_spinner.high = last_im_index
        if last_im_index == 0:
            index_slider.disabled = True
        else:
            index_slider.disabled = False
            index_slider.end = last_im_index

        zebra_mode = scan["zebra_mode"]
        if zebra_mode == "nb":
            metadata_table_source.data.update(geom=["normal beam"])
        else:  # zebra_mode == "bi"
            metadata_table_source.data.update(geom=["bisecting"])

        _update_image()
        _update_proj_plots()

    def scan_table_source_callback(_attr, _old, new):
        # unfortunately, we don't know if the change comes from data update or user input
        # also `old` and `new` are the same for non-scalars
        for scan, export in zip(dataset, new["export"]):
            scan["export"] = export

    scan_table_source = ColumnDataSource(
        dict(scan=[], fit=[], export=[], twotheta=[], gamma=[], omega=[], chi=[], phi=[], nu=[])
    )
    scan_table_source.on_change("data", scan_table_source_callback)
    scan_table_source.selected.on_change("indices", scan_table_select_callback)

    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="scan", title="Scan", editor=CellEditor(), width=50),
            TableColumn(field="fit", title="Fit", editor=CellEditor(), width=50),
            TableColumn(field="export", title="Export", editor=CellEditor(), width=50),
            TableColumn(field="twotheta", title="2theta", editor=CellEditor(), width=50),
            TableColumn(field="gamma", title="gamma", editor=CellEditor(), width=50),
            TableColumn(field="omega", title="omega", editor=CellEditor(), width=50),
            TableColumn(field="chi", title="chi", editor=CellEditor(), width=50),
            TableColumn(field="phi", title="phi", editor=CellEditor(), width=50),
            TableColumn(field="nu", title="nu", editor=CellEditor(), width=50),
        ],
        width=310,  # +60 because of the index column, but excluding twotheta onwards
        height=350,
        autosize_mode="none",
        editable=True,
    )

    def _get_selected_scan():
        return dataset[scan_table_source.selected.indices[0]]

    def _update_table():
        export = [scan["export"] for scan in dataset]
        scan_table_source.data.update(export=export)

    def monitor_spinner_callback(_attr, _old, new):
        if dataset:
            pyzebra.normalize_dataset(dataset, new)
            _update_image()
            _update_proj_plots()

    monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
    monitor_spinner.on_change("value", monitor_spinner_callback)

    merge_from_select = Select(title="scan:", width=145)

    def merge_button_callback():
        scan_into = _get_selected_scan()
        scan_from = dataset[int(merge_from_select.value)]

        if scan_into is scan_from:
            print("WARNING: Selected scans for merging are identical")
            return

        pyzebra.merge_h5_scans(scan_into, scan_from)
        _update_table()
        _update_image()
        _update_proj_plots()

    merge_button = Button(label="Merge into current", width=145)
    merge_button.on_click(merge_button_callback)

    def restore_button_callback():
        pyzebra.restore_scan(_get_selected_scan())
        _update_table()
        _update_image()
        _update_proj_plots()

    restore_button = Button(label="Restore scan", width=145)
    restore_button.on_click(restore_button_callback)

    def _update_image(index=None):
        if index is None:
            index = index_spinner.value

        scan = _get_selected_scan()
        current_image = scan["counts"][index]
        proj_v_line_source.data.update(
            x=np.arange(0, IMAGE_W) + 0.5, y=np.mean(current_image, axis=0)
        )
        proj_h_line_source.data.update(
            x=np.mean(current_image, axis=1), y=np.arange(0, IMAGE_H) + 0.5
        )

        image_source.data.update(h=[np.zeros((1, 1))], k=[np.zeros((1, 1))], l=[np.zeros((1, 1))])
        image_source.data.update(image=[current_image])

        if main_auto_checkbox.active:
            im_min = np.min(current_image)
            im_max = np.max(current_image)

            display_min_spinner.value = im_min
            display_max_spinner.value = im_max

        if "mf" in scan:
            metadata_table_source.data.update(mf=[scan["mf"][index]])
        else:
            metadata_table_source.data.update(mf=[None])

        if "temp" in scan:
            metadata_table_source.data.update(temp=[scan["temp"][index]])
        else:
            metadata_table_source.data.update(temp=[None])

        gamma, nu = calculate_pol(scan, index)
        omega = np.ones((IMAGE_H, IMAGE_W)) * scan["omega"][index]
        image_source.data.update(gamma=[gamma], nu=[nu], omega=[omega])

        # update detector center angles
        det_c_x = int(IMAGE_W / 2)
        det_c_y = int(IMAGE_H / 2)
        if scan["zebra_mode"] == "nb":
            gamma_c = gamma[det_c_y, det_c_x]
            nu_c = nu[det_c_y, det_c_x]
            omega_c = omega[det_c_y, det_c_x]
            chi_c = scan["chi"][index]
            phi_c = scan["phi"][index]

        else:  # zebra_mode == "bi"
            wave = scan["wave"]
            ddist = scan["ddist"]
            gammad = scan["gamma"][index]
            om = scan["omega"][index]
            ch = scan["chi"][index]
            ph = scan["phi"][index]
            nud = scan["nu"]

            nu_c = 0
            chi_c, phi_c, gamma_c, omega_c = pyzebra.ang_proc(
                wave, ddist, gammad, om, ch, ph, nud, det_c_x, det_c_y
            )

        detcenter_table_source.data.update(
            gamma=[gamma_c], nu=[nu_c], omega=[omega_c], chi=[chi_c], phi=[phi_c]
        )

    def _update_proj_plots():
        scan = _get_selected_scan()
        counts = scan["counts"]
        n_im, n_y, n_x = counts.shape
        im_proj_x = np.mean(counts, axis=1)
        im_proj_y = np.mean(counts, axis=2)

        # normalize for simpler colormapping
        im_proj_max_val = max(np.max(im_proj_x), np.max(im_proj_y))
        im_proj_x = 1000 * im_proj_x / im_proj_max_val
        im_proj_y = 1000 * im_proj_y / im_proj_max_val

        proj_x_image_source.data.update(image=[im_proj_x], dw=[n_x], dh=[n_im])
        proj_y_image_source.data.update(image=[im_proj_y], dw=[n_y], dh=[n_im])

        if proj_auto_checkbox.active:
            im_min = min(np.min(im_proj_x), np.min(im_proj_y))
            im_max = max(np.max(im_proj_x), np.max(im_proj_y))

            proj_display_min_spinner.value = im_min
            proj_display_max_spinner.value = im_max

        frame_range.start = 0
        frame_range.end = n_im
        frame_range.reset_start = 0
        frame_range.reset_end = n_im
        frame_range.bounds = (0, n_im)

        scan_motor = scan["scan_motor"]
        proj_y_plot.axis[1].axis_label = f"Scanning motor, {scan_motor}"

        var = scan[scan_motor]
        var_start = var[0]
        var_end = var[-1] + (var[-1] - var[0]) / (n_im - 1) if n_im != 1 else var_start + 1

        scanning_motor_range.start = var_start
        scanning_motor_range.end = var_end
        scanning_motor_range.reset_start = var_start
        scanning_motor_range.reset_end = var_end
        # handle both, ascending and descending sequences
        scanning_motor_range.bounds = (min(var_start, var_end), max(var_start, var_end))

        gamma = image_source.data["gamma"][0]
        gamma_start = gamma[0, 0]
        gamma_end = gamma[0, -1]

        gamma_range.start = gamma_start
        gamma_range.end = gamma_end
        gamma_range.reset_start = gamma_start
        gamma_range.reset_end = gamma_end
        gamma_range.bounds = (min(gamma_start, gamma_end), max(gamma_start, gamma_end))

        nu = image_source.data["nu"][0]
        nu_start = nu[0, 0]
        nu_end = nu[-1, 0]

        nu_range.start = nu_start
        nu_range.end = nu_end
        nu_range.reset_start = nu_start
        nu_range.reset_end = nu_end
        nu_range.bounds = (min(nu_start, nu_end), max(nu_start, nu_end))

    file_select = MultiSelect(title="Available .hdf files:", width=210, height=250)

    def index_callback(_attr, _old, new):
        _update_image(new)

    index_slider = Slider(value=0, start=0, end=1, show_value=False, width=400)

    index_spinner = Spinner(title="Image index:", value=0, low=0, width=100)
    index_spinner.on_change("value", index_callback)

    index_slider.js_link("value_throttled", index_spinner, "value")
    index_spinner.js_link("value", index_slider, "value")

    # image viewer figure
    plot = figure(
        x_range=Range1d(0, IMAGE_W, bounds=(0, IMAGE_W)),
        y_range=Range1d(0, IMAGE_H, bounds=(0, IMAGE_H)),
        x_axis_location="above",
        y_axis_location="right",
        plot_height=IMAGE_PLOT_H,
        plot_width=IMAGE_PLOT_W,
        toolbar_location="left",
        tools="pan,box_zoom,wheel_zoom,reset",
        active_scroll="wheel_zoom",
    )

    plot.yaxis.major_label_orientation = "vertical"
    plot.toolbar.tools[2].maintain_focus = False
    plot.toolbar.logo = None

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

    lin_color_mapper = LinearColorMapper(low=0, high=1)
    log_color_mapper = LogColorMapper(low=0, high=1)
    plot_image = plot.image(source=image_source, color_mapper=lin_color_mapper)
    plot.image(source=image_source, image="h", global_alpha=0)
    plot.image(source=image_source, image="k", global_alpha=0)
    plot.image(source=image_source, image="l", global_alpha=0)
    plot.image(source=image_source, image="gamma", global_alpha=0)
    plot.image(source=image_source, image="nu", global_alpha=0)
    plot.image(source=image_source, image="omega", global_alpha=0)

    # calculate hkl-indices of first mouse entry
    def mouse_enter_callback(_event):
        if dataset and np.array_equal(image_source.data["h"][0], np.zeros((1, 1))):
            scan = _get_selected_scan()
            index = index_spinner.value
            h, k, l = calculate_hkl(scan, index)
            image_source.data.update(h=[h], k=[k], l=[l])

    plot.on_event(MouseEnter, mouse_enter_callback)

    # Single frame projection plots
    proj_v = figure(
        x_range=plot.x_range,
        y_axis_location="right",
        plot_height=150,
        plot_width=IMAGE_PLOT_W,
        tools="",
        toolbar_location=None,
    )

    proj_v.yaxis.major_label_orientation = "vertical"
    proj_v.xaxis.major_label_text_font_size = "0pt"

    proj_v_line_source = ColumnDataSource(dict(x=[], y=[]))
    proj_v.line(source=proj_v_line_source, line_color="steelblue")

    proj_h = figure(
        x_axis_location="above",
        y_range=plot.y_range,
        plot_height=IMAGE_PLOT_H,
        plot_width=150,
        tools="",
        toolbar_location=None,
    )

    proj_h.yaxis.major_label_text_font_size = "0pt"

    proj_h_line_source = ColumnDataSource(dict(x=[], y=[]))
    proj_h.line(source=proj_h_line_source, line_color="steelblue")

    # extra tools
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
    box_edit_renderer = plot.rect(source=box_edit_source, fill_alpha=0, line_color="red")
    boxedittool = BoxEditTool(renderers=[box_edit_renderer], num_objects=1)

    def box_edit_callback(_attr, _old, new):
        if new["x"]:
            scan = _get_selected_scan()
            counts = scan["counts"]
            x_val = np.arange(counts.shape[0])
            left = int(np.floor(new["x"][0]))
            right = int(np.ceil(new["x"][0] + new["width"][0]))
            bottom = int(np.floor(new["y"][0]))
            top = int(np.ceil(new["y"][0] + new["height"][0]))
            y_val = np.sum(counts[:, bottom:top, left:right], axis=(1, 2))
        else:
            x_val = []
            y_val = []

        roi_avg_plot_line_source.data.update(x=x_val, y=y_val)

    box_edit_source.on_change("data", box_edit_callback)

    plot.add_tools(hovertool, boxedittool)

    # Overview projection plots
    # shared frame ranges
    frame_range = Range1d(0, 1, bounds=(0, 1))
    scanning_motor_range = Range1d(0, 1, bounds=(0, 1))
    lin_color_mapper_proj = LinearColorMapper(low=0, high=1)
    log_color_mapper_proj = LogColorMapper(low=0, high=1)

    det_x_range = Range1d(0, IMAGE_W, bounds=(0, IMAGE_W))
    gamma_range = Range1d(0, 1, bounds=(0, 1))
    proj_x_plot = figure(
        title="Projections on X-axis",
        x_axis_label="Coordinate X, pix",
        y_axis_label="Frame",
        x_range=det_x_range,
        y_range=frame_range,
        extra_x_ranges={"gamma": gamma_range},
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        plot_height=540,
        plot_width=IMAGE_PLOT_W - 3,
        tools="pan,box_zoom,wheel_zoom,reset",
        active_scroll="wheel_zoom",
    )

    proj_x_plot.yaxis.major_label_orientation = "vertical"
    proj_x_plot.toolbar.tools[2].maintain_focus = False

    proj_x_plot.add_layout(LinearAxis(x_range_name="gamma", axis_label="Gamma, deg"), place="above")

    proj_x_image_source = ColumnDataSource(
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_W], dh=[1])
    )

    proj_x_image = proj_x_plot.image(source=proj_x_image_source, color_mapper=lin_color_mapper_proj)

    det_y_range = Range1d(0, IMAGE_H, bounds=(0, IMAGE_H))
    nu_range = Range1d(0, 1, bounds=(0, 1))
    proj_y_plot = figure(
        title="Projections on Y-axis",
        x_axis_label="Coordinate Y, pix",
        y_axis_label="Scanning motor",
        y_axis_location="right",
        x_range=det_y_range,
        y_range=frame_range,
        extra_x_ranges={"nu": nu_range},
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        plot_height=540,
        plot_width=IMAGE_PLOT_H + 22,
        tools="pan,box_zoom,wheel_zoom,reset",
        active_scroll="wheel_zoom",
    )

    proj_y_plot.yaxis.y_range_name = "scanning_motor"
    proj_y_plot.yaxis.major_label_orientation = "vertical"
    proj_y_plot.toolbar.tools[2].maintain_focus = False

    proj_y_plot.add_layout(LinearAxis(x_range_name="nu", axis_label="Nu, deg"), place="above")

    proj_y_image_source = ColumnDataSource(
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_H], dh=[1])
    )

    proj_y_image = proj_y_plot.image(source=proj_y_image_source, color_mapper=lin_color_mapper_proj)

    # ROI slice plot
    roi_avg_plot = figure(plot_height=150, plot_width=IMAGE_PLOT_W, tools="", toolbar_location=None)

    roi_avg_plot_line_source = ColumnDataSource(dict(x=[], y=[]))
    roi_avg_plot.line(source=roi_avg_plot_line_source, line_color="steelblue")

    def colormap_select_callback(_attr, _old, new):
        lin_color_mapper.palette = new
        log_color_mapper.palette = new
        lin_color_mapper_proj.palette = new
        log_color_mapper_proj.palette = new

    colormap_select = Select(
        title="Colormap:",
        options=[("Greys256", "greys"), ("Plasma256", "plasma"), ("Cividis256", "cividis")],
        width=100,
    )
    colormap_select.on_change("value", colormap_select_callback)
    colormap_select.value = "Plasma256"

    def colormap_scale_rg_callback(selection):
        if selection == 0:  # Linear
            plot_image.glyph.color_mapper = lin_color_mapper
            proj_x_image.glyph.color_mapper = lin_color_mapper_proj
            proj_y_image.glyph.color_mapper = lin_color_mapper_proj

        else:  # Logarithmic
            if (
                display_min_spinner.value > 0
                and display_max_spinner.value > 0
                and proj_display_min_spinner.value > 0
                and proj_display_max_spinner.value > 0
            ):
                plot_image.glyph.color_mapper = log_color_mapper
                proj_x_image.glyph.color_mapper = log_color_mapper_proj
                proj_y_image.glyph.color_mapper = log_color_mapper_proj
            else:
                colormap_scale_rg.active = 0

    colormap_scale_rg = RadioGroup(labels=["Linear", "Logarithmic"], active=0, width=100)
    colormap_scale_rg.on_click(colormap_scale_rg_callback)

    def main_auto_checkbox_callback(state):
        if state:
            display_min_spinner.disabled = True
            display_max_spinner.disabled = True
        else:
            display_min_spinner.disabled = False
            display_max_spinner.disabled = False

        _update_image()

    main_auto_checkbox = CheckboxGroup(
        labels=["Frame Intensity Range"], active=[0], width=145, margin=[10, 5, 0, 5]
    )
    main_auto_checkbox.on_click(main_auto_checkbox_callback)

    def display_max_spinner_callback(_attr, _old, new):
        lin_color_mapper.high = new
        log_color_mapper.high = new
        # TODO: without this _update_image() log color mapper display is delayed
        _update_image()

    display_max_spinner = Spinner(value=1, disabled=bool(main_auto_checkbox.active), width=100)
    display_max_spinner.on_change("value", display_max_spinner_callback)

    def display_min_spinner_callback(_attr, _old, new):
        lin_color_mapper.low = new
        log_color_mapper.low = new
        _update_image()

    display_min_spinner = Spinner(value=0, disabled=bool(main_auto_checkbox.active), width=100)
    display_min_spinner.on_change("value", display_min_spinner_callback)

    def proj_auto_checkbox_callback(state):
        if state:
            proj_display_min_spinner.disabled = True
            proj_display_max_spinner.disabled = True
        else:
            proj_display_min_spinner.disabled = False
            proj_display_max_spinner.disabled = False

        _update_proj_plots()

    proj_auto_checkbox = CheckboxGroup(
        labels=["Projections Intensity Range"], active=[0], width=145, margin=[10, 5, 0, 5]
    )
    proj_auto_checkbox.on_click(proj_auto_checkbox_callback)

    def proj_display_max_spinner_callback(_attr, _old, new):
        lin_color_mapper_proj.high = new
        log_color_mapper_proj.high = new
        _update_proj_plots()

    proj_display_max_spinner = Spinner(value=1, disabled=bool(proj_auto_checkbox.active), width=100)
    proj_display_max_spinner.on_change("value", proj_display_max_spinner_callback)

    def proj_display_min_spinner_callback(_attr, _old, new):
        lin_color_mapper_proj.low = new
        log_color_mapper_proj.low = new
        _update_proj_plots()

    proj_display_min_spinner = Spinner(value=0, disabled=bool(proj_auto_checkbox.active), width=100)
    proj_display_min_spinner.on_change("value", proj_display_min_spinner_callback)

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

    detcenter_table_source = ColumnDataSource(dict(gamma=[], omega=[], chi=[], phi=[], nu=[]))
    detcenter_table = DataTable(
        source=detcenter_table_source,
        columns=[
            TableColumn(field="gamma", title="Gamma", formatter=num_formatter, width=70),
            TableColumn(field="omega", title="Omega", formatter=num_formatter, width=70),
            TableColumn(field="chi", title="Chi", formatter=num_formatter, width=70),
            TableColumn(field="phi", title="Phi", formatter=num_formatter, width=70),
            TableColumn(field="nu", title="Nu", formatter=num_formatter, width=70),
        ],
        height=150,
        width=350,
        autosize_mode="none",
        index_position=None,
    )

    def add_event_button_callback():
        scan = _get_selected_scan()
        pyzebra.fit_event(
            scan,
            int(np.floor(frame_range.start)),
            int(np.ceil(frame_range.end)),
            int(np.floor(det_y_range.start)),
            int(np.ceil(det_y_range.end)),
            int(np.floor(det_x_range.start)),
            int(np.ceil(det_x_range.end)),
        )

        wave = scan["wave"]
        ddist = scan["ddist"]
        cell = scan["cell"]

        gamma = scan["gamma"][0]
        omega = scan["omega"][0]
        nu = scan["nu"]
        chi = scan["chi"][0]
        phi = scan["phi"][0]

        scan_motor = scan["scan_motor"]
        var_angle = scan[scan_motor]

        snr_cnts = scan["fit"]["snr"]
        frC = scan["fit"]["frame"]

        var_F = var_angle[int(np.floor(frC))]
        var_C = var_angle[int(np.ceil(frC))]
        frStep = frC - np.floor(frC)
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

        intensity = scan["fit"]["intensity"]
        x_pos = scan["fit"]["x_pos"]
        y_pos = scan["fit"]["y_pos"]

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

    add_event_button = Button(label="Add peak center", width=145)
    add_event_button.on_click(add_event_button_callback)

    def remove_event_button_callback():
        ind2remove = events_table_source.selected.indices
        for value in events_data.values():
            for ind in reversed(ind2remove):
                del value[ind]

        events_table_source.data = events_data

    remove_event_button = Button(label="Remove peak center", width=145)
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
    peak_tables = Tabs(
        tabs=[
            Panel(child=events_table, title="Actual peak center"),
            Panel(child=detcenter_table, title="Peak in the detector center"),
        ]
    )

    import_layout = column(
        data_source,
        upload_cami_div,
        upload_cami_button,
        upload_hdf_div,
        upload_hdf_button,
        file_select,
        row(file_open_button, file_append_button),
    )

    layout_image = column(gridplot([[proj_v, None], [plot, proj_h]], merge_tools=False))
    colormap_layout = column(
        row(colormap_select, column(Spacer(height=15), colormap_scale_rg)),
        main_auto_checkbox,
        row(display_min_spinner, display_max_spinner),
        proj_auto_checkbox,
        row(proj_display_min_spinner, proj_display_max_spinner),
    )

    layout_controls = column(
        row(metadata_table, index_spinner, column(Spacer(height=25), index_slider)),
        row(column(add_event_button, remove_event_button), peak_tables),
    )

    layout_proj = column(
        gridplot(
            [[proj_x_plot, proj_y_plot]], toolbar_options={"logo": None}, toolbar_location="right"
        )
    )

    scan_layout = column(
        scan_table,
        row(monitor_spinner, column(Spacer(height=19), restore_button)),
        row(column(Spacer(height=19), merge_button), merge_from_select),
    )

    tab_layout = row(
        column(import_layout, colormap_layout),
        column(row(scan_layout, layout_proj), layout_controls),
        column(roi_avg_plot, layout_image),
    )

    return Panel(child=tab_layout, title="hdf viewer")


def calculate_hkl(scan, index):
    h = np.empty(shape=(IMAGE_H, IMAGE_W))
    k = np.empty(shape=(IMAGE_H, IMAGE_W))
    l = np.empty(shape=(IMAGE_H, IMAGE_W))

    wave = scan["wave"]
    ddist = scan["ddist"]
    gammad = scan["gamma"][index]
    om = scan["omega"][index]
    nud = scan["nu"]
    ub_inv = np.linalg.inv(scan["ub"])
    geometry = scan["zebra_mode"]

    if geometry == "bi":
        chi = scan["chi"][index]
        phi = scan["phi"][index]
    elif geometry == "nb":
        chi = 0
        phi = 0
    else:
        raise ValueError(f"Unknown geometry type '{geometry}'")

    h, k, l = pyzebra.ang2hkl_det(wave, ddist, gammad, om, chi, phi, nud, ub_inv)

    return h, k, l


def calculate_pol(scan, index):
    ddist = scan["ddist"]
    gammad = scan["gamma"][index]
    nud = scan["nu"]
    yi, xi = np.ogrid[:IMAGE_H, :IMAGE_W]
    gamma, nu = pyzebra.det2pol(ddist, gammad, nud, xi, yi)

    return gamma, nu
