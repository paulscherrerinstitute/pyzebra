import base64
import io
import os

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    Button,
    CellEditor,
    CheckboxGroup,
    ColumnDataSource,
    DataTable,
    Div,
    FileInput,
    LinearColorMapper,
    MultiSelect,
    NumberEditor,
    NumberFormatter,
    Panel,
    Range1d,
    Select,
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
        for scan in dataset:
            file_list.append(os.path.basename(scan["original_filename"]))

        scan_table_source.data.update(
            file=file_list,
            param=[None] * len(dataset),
            frame=[None] * len(dataset),
            x_pos=[None] * len(dataset),
            y_pos=[None] * len(dataset),
        )
        scan_table_source.selected.indices = []
        scan_table_source.selected.indices = [0]

        param_select.value = "user defined"

    def _update_table():
        frame = []
        x_pos = []
        y_pos = []
        for scan in dataset:
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

    def _file_open():
        new_data = []
        for f_name in file_select.value:
            try:
                new_data.append(pyzebra.read_detector_data(f_name))
            except KeyError:
                print("Could not read data from the file.")
                return

        dataset.extend(new_data)

        _init_datatable()

    def file_open_button_callback():
        nonlocal dataset
        dataset = []
        _file_open()

    file_open_button = Button(label="Open New", width=100)
    file_open_button.on_click(file_open_button_callback)

    def file_append_button_callback():
        _file_open()

    file_append_button = Button(label="Append", width=100)
    file_append_button.on_click(file_append_button_callback)

    # Scan select
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

        scan = dataset[new[0]]

        zebra_mode = scan["zebra_mode"]
        if zebra_mode == "nb":
            metadata_table_source.data.update(geom=["normal beam"])
        else:  # zebra_mode == "bi"
            metadata_table_source.data.update(geom=["bisecting"])

        if "mf" in scan:
            metadata_table_source.data.update(mf=[scan["mf"][0]])
        else:
            metadata_table_source.data.update(mf=[None])

        if "temp" in scan:
            metadata_table_source.data.update(temp=[scan["temp"][0]])
        else:
            metadata_table_source.data.update(temp=[None])

        _update_proj_plots()

    def scan_table_source_callback(_attr, _old, _new):
        pass

    scan_table_source = ColumnDataSource(dict(file=[], param=[], frame=[], x_pos=[], y_pos=[]))
    scan_table_source.selected.on_change("indices", scan_table_select_callback)
    scan_table_source.on_change("data", scan_table_source_callback)

    scan_table = DataTable(
        source=scan_table_source,
        columns=[
            TableColumn(field="file", title="file", editor=CellEditor(), width=150),
            TableColumn(
                field="param",
                title="param",
                formatter=num_formatter,
                editor=NumberEditor(),
                width=50,
            ),
            TableColumn(
                field="frame", title="Frame", formatter=num_formatter, editor=CellEditor(), width=70
            ),
            TableColumn(
                field="x_pos", title="X", formatter=num_formatter, editor=CellEditor(), width=70
            ),
            TableColumn(
                field="y_pos", title="Y", formatter=num_formatter, editor=CellEditor(), width=70
            ),
        ],
        width=470,  # +60 because of the index column
        height=420,
        editable=True,
        autosize_mode="none",
    )

    def _get_selected_scan():
        return dataset[scan_table_source.selected.indices[0]]

    def param_select_callback(_attr, _old, new):
        if new == "user defined":
            param = [None] * len(dataset)
        else:
            # TODO: which value to take?
            param = [scan[new][0] for scan in dataset]

        scan_table_source.data["param"] = param
        _update_param_plot()

    param_select = Select(
        title="Parameter:",
        options=["user defined", "temp", "mf", "h", "k", "l"],
        value="user defined",
        width=145,
    )
    param_select.on_change("value", param_select_callback)

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
    color_mapper_proj = LinearColorMapper()

    det_x_range = Range1d(0, IMAGE_W, bounds=(0, IMAGE_W))
    proj_x_plot = figure(
        title="Projections on X-axis",
        x_axis_label="Coordinate X, pix",
        y_axis_label="Frame",
        x_range=det_x_range,
        y_range=frame_range,
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        height=540,
        width=IMAGE_PLOT_W - 3,
        tools="pan,box_zoom,wheel_zoom,reset",
        active_scroll="wheel_zoom",
    )

    proj_x_plot.yaxis.major_label_orientation = "vertical"
    proj_x_plot.toolbar.tools[2].maintain_focus = False

    proj_x_image_source = ColumnDataSource(
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_W], dh=[1])
    )

    proj_x_plot.image(source=proj_x_image_source, color_mapper=color_mapper_proj)

    det_y_range = Range1d(0, IMAGE_H, bounds=(0, IMAGE_H))
    proj_y_plot = figure(
        title="Projections on Y-axis",
        x_axis_label="Coordinate Y, pix",
        y_axis_label="Scanning motor",
        y_axis_location="right",
        x_range=det_y_range,
        y_range=frame_range,
        extra_y_ranges={"scanning_motor": scanning_motor_range},
        height=540,
        width=IMAGE_PLOT_H + 22,
        tools="pan,box_zoom,wheel_zoom,reset",
        active_scroll="wheel_zoom",
    )

    proj_y_plot.yaxis.y_range_name = "scanning_motor"
    proj_y_plot.yaxis.major_label_orientation = "vertical"
    proj_y_plot.toolbar.tools[2].maintain_focus = False

    proj_y_image_source = ColumnDataSource(
        dict(image=[np.zeros((1, 1), dtype="float32")], x=[0], y=[0], dw=[IMAGE_H], dh=[1])
    )

    proj_y_plot.image(source=proj_y_image_source, color_mapper=color_mapper_proj)

    def colormap_select_callback(_attr, _old, new):
        color_mapper_proj.palette = new

    colormap_select = Select(
        title="Colormap:",
        options=[("Greys256", "greys"), ("Plasma256", "plasma"), ("Cividis256", "cividis")],
        width=210,
    )
    colormap_select.on_change("value", colormap_select_callback)
    colormap_select.value = "Plasma256"

    def proj_auto_checkbox_callback(_attr, _old, new):
        if 0 in new:
            proj_display_min_spinner.disabled = True
            proj_display_max_spinner.disabled = True
        else:
            proj_display_min_spinner.disabled = False
            proj_display_max_spinner.disabled = False

        _update_proj_plots()

    proj_auto_checkbox = CheckboxGroup(
        labels=["Projections Intensity Range"], active=[0], width=145, margin=[10, 5, 0, 5]
    )
    proj_auto_checkbox.on_change("active", proj_auto_checkbox_callback)

    def proj_display_max_spinner_callback(_attr, _old, new):
        color_mapper_proj.high = new

    proj_display_max_spinner = Spinner(
        value=1, disabled=bool(proj_auto_checkbox.active), mode="int", width=100
    )
    proj_display_max_spinner.on_change("value", proj_display_max_spinner_callback)

    def proj_display_min_spinner_callback(_attr, _old, new):
        color_mapper_proj.low = new

    proj_display_min_spinner = Spinner(
        value=0, disabled=bool(proj_auto_checkbox.active), mode="int", width=100
    )
    proj_display_min_spinner.on_change("value", proj_display_min_spinner_callback)

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
        for s, p in zip(dataset, scan_table_source.data["param"]):
            if "fit" in s and fit_param:
                x.append(p)
                y.append(s["fit"][fit_param])
        param_scatter_source.data.update(x=x, y=y)

    # Parameter plot
    param_plot = figure(
        x_axis_label="Parameter",
        y_axis_label="Fit parameter",
        height=400,
        width=700,
        tools="pan,wheel_zoom,reset",
    )

    param_scatter_source = ColumnDataSource(dict(x=[], y=[]))
    param_plot.circle(source=param_scatter_source)

    param_plot.toolbar.logo = None

    def fit_param_select_callback(_attr, _old, _new):
        _update_param_plot()

    fit_param_select = Select(title="Fit parameter", options=[], width=145)
    fit_param_select.on_change("value", fit_param_select_callback)

    def proc_all_button_callback():
        for scan in dataset:
            pyzebra.fit_event(
                scan,
                int(np.floor(frame_range.start)),
                int(np.ceil(frame_range.end)),
                int(np.floor(det_y_range.start)),
                int(np.ceil(det_y_range.end)),
                int(np.floor(det_x_range.start)),
                int(np.ceil(det_x_range.end)),
            )

        _update_table()

        for scan in dataset:
            if "fit" in scan:
                options = list(scan["fit"].keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

        _update_param_plot()

    proc_all_button = Button(label="Process All", button_type="primary", width=145)
    proc_all_button.on_click(proc_all_button_callback)

    def proc_button_callback():
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

        _update_table()

        for scan in dataset:
            if "fit" in scan:
                options = list(scan["fit"].keys())
                fit_param_select.options = options
                fit_param_select.value = options[0]
                break

        _update_param_plot()

    proc_button = Button(label="Process Current", width=145)
    proc_button.on_click(proc_button_callback)

    layout_controls = row(
        colormap_select,
        column(proj_auto_checkbox, row(proj_display_min_spinner, proj_display_max_spinner)),
        proc_button,
        proc_all_button,
    )

    layout_proj = column(
        gridplot(
            [[proj_x_plot, proj_y_plot]], toolbar_options={"logo": None}, toolbar_location="right"
        ),
        layout_controls,
    )

    # Plot tabs
    plots = Tabs(
        tabs=[
            Panel(child=layout_proj, title="single scan"),
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
