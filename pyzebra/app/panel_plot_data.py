import base64
import io
import os

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    DataRange1d,
    Div,
    FileInput,
    LinearColorMapper,
    LogColorMapper,
    NumericInput,
    Panel,
    RadioGroup,
    Select,
    Spacer,
    Spinner,
    TextInput,
)
from bokeh.plotting import figure
from scipy import interpolate

import pyzebra
from pyzebra import app
from pyzebra.app.panel_hdf_viewer import calculate_hkl


def create():
    _update_slice = None
    measured_data_div = Div(text="Measured <b>HDF</b> data:")
    measured_data = FileInput(accept=".hdf", multiple=True, width=200)

    upload_hkl_div = Div(text="Open hkl/mhkl data:")
    upload_hkl_fi = FileInput(accept=".hkl,.mhkl", multiple=True, width=200)

    def _prepare_plotting():
        flag_ub = bool(redef_ub_cb.active)
        flag_lattice = bool(redef_lattice_cb.active)

        # Define horizontal direction of plotting plane, vertical direction will be calculated
        # automatically
        x_dir = list(map(float, hkl_in_plane_x.value.split()))

        # Define direction orthogonal to plotting plane. Together with orth_cut, this parameter also
        # defines the position of the cut, ie cut will be taken at orth_dir = [x,y,z]*orth_cut +- delta,
        # where delta is max distance a data point can have from cut in rlu units
        orth_dir = list(map(float, hkl_normal.value.split()))

        # Load data files
        md_fnames = measured_data.filename
        md_fdata = measured_data.value

        for ind, (fname, fdata) in enumerate(zip(md_fnames, md_fdata)):
            # Read data
            try:
                det_data = pyzebra.read_detector_data(io.BytesIO(base64.b64decode(fdata)))
            except:
                print(f"Error loading {fname}")
                return None

            if ind == 0:
                if not flag_ub:
                    redef_ub_ti.value = " ".join(map(str, det_data["ub"].ravel()))
                if not flag_lattice:
                    redef_lattice_ti.value = " ".join(map(str, det_data["cell"]))

            num_slices = np.shape(det_data["counts"])[0]

            # Change parameter
            if flag_ub:
                ub = list(map(float, redef_ub_ti.value.strip().split()))
                det_data["ub"] = np.array(ub).reshape(3, 3)

            # Convert h k l for all images in file
            h_temp = np.empty(np.shape(det_data["counts"]))
            k_temp = np.empty(np.shape(det_data["counts"]))
            l_temp = np.empty(np.shape(det_data["counts"]))
            for i in range(num_slices):
                h_temp[i], k_temp[i], l_temp[i] = calculate_hkl(det_data, i)

            # Append to matrix
            if ind == 0:
                h = h_temp
                k = k_temp
                l = l_temp
                I_matrix = det_data["counts"]
            else:
                h = np.append(h, h_temp, axis=0)
                k = np.append(k, k_temp, axis=0)
                l = np.append(l, l_temp, axis=0)
                I_matrix = np.append(I_matrix, det_data["counts"], axis=0)

        if flag_lattice:
            vals = list(map(float, redef_lattice_ti.value.strip().split()))
            lattice = np.array(vals)
        else:
            lattice = det_data["cell"]

        # Define matrix for converting to cartesian coordinates and back
        alpha = lattice[3] * np.pi / 180.0
        beta = lattice[4] * np.pi / 180.0
        gamma = lattice[5] * np.pi / 180.0

        # reciprocal angle parameters
        beta_star = np.arccos(
            (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))
        )
        gamma_star = np.arccos(
            (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
        )

        # conversion matrix:
        M = np.array(
            [
                [1, 1 * np.cos(gamma_star), 1 * np.cos(beta_star)],
                [0, 1 * np.sin(gamma_star), -np.sin(beta_star) * np.cos(alpha)],
                [0, 0, 1 * np.sin(beta_star) * np.sin(alpha)],
            ]
        )

        # Get last lattice vector
        y_dir = np.cross(x_dir, orth_dir)  # Second axes of plotting plane

        # Rescale such that smallest element of y-dir vector is 1
        y_dir2 = y_dir[y_dir != 0]
        min_val = np.min(np.abs(y_dir2))
        y_dir = y_dir / min_val

        # Possibly flip direction of ydir:
        if y_dir[np.argmax(abs(y_dir))] < 0:
            y_dir = -y_dir

        # Display the resulting y_dir
        hkl_in_plane_y.value = " ".join([f"{val:.1f}" for val in y_dir])

        # # Save length of lattice vectors
        # x_length = np.linalg.norm(x_dir)
        # y_length = np.linalg.norm(y_dir)

        # # Save str for labels
        # xlabel_str = " ".join(map(str, x_dir))
        # ylabel_str = " ".join(map(str, y_dir))

        # Normalize lattice vectors
        y_dir = y_dir / np.linalg.norm(y_dir)
        x_dir = x_dir / np.linalg.norm(x_dir)
        orth_dir = orth_dir / np.linalg.norm(orth_dir)

        # Calculate cartesian equivalents of lattice vectors
        x_c = np.matmul(M, x_dir)
        y_c = np.matmul(M, y_dir)
        o_c = np.matmul(M, orth_dir)

        # Calulcate vertical direction in plotting plame
        y_vert = np.cross(x_c, o_c)  # verical direction in plotting plane
        if y_vert[np.argmax(abs(y_vert))] < 0:
            y_vert = -y_vert
        y_vert = y_vert / np.linalg.norm(y_vert)

        # Normalize all directions
        y_c = y_c / np.linalg.norm(y_c)
        x_c = x_c / np.linalg.norm(x_c)
        o_c = o_c / np.linalg.norm(o_c)

        # Convert all hkls to cartesian
        hkl = [[h, k, l]]
        hkl = np.transpose(hkl)
        hkl_c = np.matmul(M, hkl)

        # Prepare hkl/mhkl data
        hkl_coord = []
        for j, fname in enumerate(upload_hkl_fi.filename):
            with io.StringIO(base64.b64decode(upload_hkl_fi.value[j]).decode()) as file:
                _, ext = os.path.splitext(fname)
                try:
                    fdata = pyzebra.parse_hkl(file, ext)
                except:
                    print(f"Error loading {fname}")
                    return

            for ind in range(len(fdata["counts"])):
                # Recognize k_flag_vec
                hkl = np.array([fdata["h"][ind], fdata["k"][ind], fdata["l"][ind]])

                # Save data
                hkl_coord.append(hkl)

        def _update_slice():
            # Where should cut be along orthogonal direction (Mutliplication factor onto orth_dir)
            orth_cut = hkl_cut.value

            # Width of cut
            delta = hkl_delta.value

            # Calculate distance of all points to plane
            Q = np.array(o_c) * orth_cut
            N = o_c / np.sqrt(np.sum(o_c**2))
            v = np.empty(np.shape(hkl_c))
            v[:, :, :, :, 0] = hkl_c[:, :, :, :, 0] - Q
            dist = np.abs(np.dot(N, v))
            dist = np.squeeze(dist)
            dist = np.transpose(dist)

            # Find points within acceptable distance of plane defined by o_c
            ind = np.where(abs(dist) < delta)
            if ind[0].size == 0:
                image_source.data.update(image=[np.zeros((1, 1))])
                return

            # Project points onto axes
            x = np.dot(x_c / np.sqrt(np.sum(x_c**2)), hkl_c)
            y = np.dot(y_c / np.sqrt(np.sum(y_c**2)), hkl_c)

            # take care of dimensions
            x = np.squeeze(x)
            x = np.transpose(x)
            y = np.squeeze(y)
            y = np.transpose(y)

            # Get slices:
            x_slice = x[ind]
            y_slice = y[ind]
            I_slice = I_matrix[ind]

            # Meshgrid limits for plotting
            if auto_range_cb.active:
                min_x = np.min(x_slice)
                max_x = np.max(x_slice)
                min_y = np.min(y_slice)
                max_y = np.max(y_slice)
                xrange_min_ni.value = min_x
                xrange_max_ni.value = max_x
                yrange_min_ni.value = min_y
                yrange_max_ni.value = max_y
            else:
                min_x = xrange_min_ni.value
                max_x = xrange_max_ni.value
                min_y = yrange_min_ni.value
                max_y = yrange_max_ni.value

            delta_x = xrange_step_ni.value
            delta_y = yrange_step_ni.value

            # Create interpolated mesh grid for plotting
            grid_x, grid_y = np.mgrid[min_x:max_x:delta_x, min_y:max_y:delta_y]
            I = interpolate.griddata((x_slice, y_slice), I_slice, (grid_x, grid_y))

            # Update plot
            display_min_ni.value = 0
            display_max_ni.value = np.max(I_slice) * 0.25
            image_source.data.update(
                image=[I.T], x=[min_x], dw=[max_x - min_x], y=[min_y], dh=[max_y - min_y]
            )

            scan_x, scan_y = [], []
            for j in range(len(hkl_coord)):
                # Get middle hkl from list
                hklm = M @ hkl_coord[j]

                # Decide if point is in the cut
                proj = np.dot(hklm, o_c)
                if abs(proj - orth_cut) >= delta:
                    continue

                # Project onto axes
                hklmx = np.dot(hklm, x_c)
                hklmy = np.dot(hklm, y_vert)

                # Plot middle point of scan
                scan_x.append(hklmx)
                scan_y.append(hklmy)

            scatter_source.data.update(x=scan_x, y=scan_y)

        return _update_slice

    def plot_file_callback():
        nonlocal _update_slice
        _update_slice = _prepare_plotting()
        _update_slice()

    plot_file = Button(label="Plot selected file(s)", button_type="primary", width=200)
    plot_file.on_click(plot_file_callback)

    plot = figure(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        height=550 + 27,
        width=550 + 117,
        tools="pan,wheel_zoom,reset",
    )
    plot.toolbar.logo = None

    lin_color_mapper = LinearColorMapper(nan_color=(0, 0, 0, 0), low=0, high=1)
    log_color_mapper = LogColorMapper(nan_color=(0, 0, 0, 0), low=0, high=1)
    image_source = ColumnDataSource(dict(image=[np.zeros((1, 1))], x=[0], y=[0], dw=[1], dh=[1]))
    plot_image = plot.image(source=image_source, color_mapper=lin_color_mapper)

    lin_color_bar = ColorBar(color_mapper=lin_color_mapper, width=15)
    log_color_bar = ColorBar(color_mapper=log_color_mapper, width=15, visible=False)
    plot.add_layout(lin_color_bar, "right")
    plot.add_layout(log_color_bar, "right")

    scatter_source = ColumnDataSource(dict(x=[], y=[]))
    plot.scatter(source=scatter_source, size=4, fill_color="green", line_color="green")

    hkl_div = Div(text="HKL:", margin=(5, 5, 0, 5))
    hkl_normal = TextInput(title="normal", value="0 0 1", width=70)

    def hkl_cut_callback(_attr, _old, _new):
        if _update_slice is not None:
            _update_slice()

    hkl_cut = Spinner(title="cut", value=0, step=0.1, width=70)
    hkl_cut.on_change("value_throttled", hkl_cut_callback)

    hkl_delta = NumericInput(title="delta", value=0.1, mode="float", width=70)
    hkl_in_plane_x = TextInput(title="in-plane X", value="1 0 0", width=70)
    hkl_in_plane_y = TextInput(title="in-plane Y", value="", width=100, disabled=True)

    def redef_lattice_cb_callback(_attr, _old, new):
        if 0 in new:
            redef_lattice_ti.disabled = False
        else:
            redef_lattice_ti.disabled = True

    redef_lattice_cb = CheckboxGroup(labels=["Redefine lattice:"], width=110)
    redef_lattice_cb.on_change("active", redef_lattice_cb_callback)
    redef_lattice_ti = TextInput(width=490, disabled=True)

    def redef_ub_cb_callback(_attr, _old, new):
        if 0 in new:
            redef_ub_ti.disabled = False
        else:
            redef_ub_ti.disabled = True

    redef_ub_cb = CheckboxGroup(labels=["Redefine UB:"], width=110)
    redef_ub_cb.on_change("active", redef_ub_cb_callback)
    redef_ub_ti = TextInput(width=490, disabled=True)

    def colormap_select_callback(_attr, _old, new):
        lin_color_mapper.palette = new
        log_color_mapper.palette = new

    colormap_select = Select(
        title="Colormap:",
        options=[("Greys256", "greys"), ("Plasma256", "plasma"), ("Cividis256", "cividis")],
        width=100,
    )
    colormap_select.on_change("value", colormap_select_callback)
    colormap_select.value = "Plasma256"

    def display_min_ni_callback(_attr, _old, new):
        lin_color_mapper.low = new
        log_color_mapper.low = new

    display_min_ni = NumericInput(title="Intensity min:", value=0, mode="float", width=70)
    display_min_ni.on_change("value", display_min_ni_callback)

    def display_max_ni_callback(_attr, _old, new):
        lin_color_mapper.high = new
        log_color_mapper.high = new

    display_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    display_max_ni.on_change("value", display_max_ni_callback)

    def colormap_scale_rg_callback(_attr, _old, new):
        if new == 0:  # Linear
            plot_image.glyph.color_mapper = lin_color_mapper
            lin_color_bar.visible = True
            log_color_bar.visible = False

        else:  # Logarithmic
            if display_min_ni.value > 0 and display_max_ni.value > 0:
                plot_image.glyph.color_mapper = log_color_mapper
                lin_color_bar.visible = False
                log_color_bar.visible = True
            else:
                colormap_scale_rg.active = 0

    colormap_scale_rg = RadioGroup(labels=["Linear", "Logarithmic"], active=0, width=100)
    colormap_scale_rg.on_change("active", colormap_scale_rg_callback)

    xrange_min_ni = NumericInput(title="x range min:", value=0, mode="float", width=70)
    xrange_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    xrange_step_ni = NumericInput(title="x mesh:", value=0.01, mode="float", width=70)

    yrange_min_ni = NumericInput(title="y range min:", value=0, mode="float", width=70)
    yrange_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    yrange_step_ni = NumericInput(title="y mesh:", value=0.01, mode="float", width=70)

    def auto_range_cb_callback(_attr, _old, new):
        if 0 in new:
            xrange_min_ni.disabled = True
            xrange_max_ni.disabled = True
            yrange_min_ni.disabled = True
            yrange_max_ni.disabled = True
        else:
            xrange_min_ni.disabled = False
            xrange_max_ni.disabled = False
            yrange_min_ni.disabled = False
            yrange_max_ni.disabled = False

    auto_range_cb = CheckboxGroup(labels=["Auto range:"], width=110)
    auto_range_cb.on_change("active", auto_range_cb_callback)
    auto_range_cb.active = [0]

    column1_layout = column(
        row(
            column(row(measured_data_div, measured_data), row(upload_hkl_div, upload_hkl_fi)),
            plot_file,
        ),
        row(
            plot,
            column(
                hkl_div,
                row(hkl_normal, hkl_cut, hkl_delta),
                row(hkl_in_plane_x, hkl_in_plane_y),
                row(colormap_select, column(Spacer(height=15), colormap_scale_rg)),
                row(display_min_ni, display_max_ni),
                row(column(Spacer(height=19), auto_range_cb)),
                row(xrange_min_ni, xrange_max_ni),
                row(yrange_min_ni, yrange_max_ni),
                row(xrange_step_ni, yrange_step_ni),
            ),
        ),
        row(column(Spacer(height=7), redef_lattice_cb), redef_lattice_ti),
        row(column(Spacer(height=7), redef_ub_cb), redef_ub_ti),
    )
    column2_layout = app.PlotHKL().layout

    tab_layout = row(column1_layout, Spacer(width=50), column2_layout)

    return Panel(child=tab_layout, title="plot data")
