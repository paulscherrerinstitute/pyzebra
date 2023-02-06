import base64
import io

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    DataRange1d,
    Div,
    FileInput,
    LinearColorMapper,
    NumericInput,
    Panel,
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
    measured_data_div = Div(text="Measured data:")
    measured_data = FileInput(accept=".hdf", multiple=True, width=200)

    def plot_file_callback():
        flag_ub = bool(redef_ub_cb.active)
        flag_lattice = bool(redef_lattice_cb.active)

        # Define horizontal direction of plotting plane, vertical direction will be calculated
        # automatically
        x_dir = list(map(float, hkl_in_plane_x.value.split()))

        # Define direction orthogonal to plotting plane. Together with orth_cut, this parameter also
        # defines the position of the cut, ie cut will be taken at orth_dir = [x,y,z]*orth_cut +- delta,
        # where delta is max distance a data point can have from cut in rlu units
        orth_dir = list(map(float, hkl_normal.value.split()))

        # Where should cut be along orthogonal direction (Mutliplication factor onto orth_dir)
        orth_cut = hkl_cut.value

        # Width of cut
        delta = hkl_delta.value

        # Load data files
        md_fnames = measured_data.filename
        md_fdata = measured_data.value

        for ind, (fname, fdata) in enumerate(zip(md_fnames, md_fdata)):
            # Read data
            try:
                det_data = pyzebra.read_detector_data(io.BytesIO(base64.b64decode(fdata)))
            except:
                print(f"Error loading {fname}")
                return

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

        # Convert all hkls to cartesian
        hkl = [[h, k, l]]
        hkl = np.transpose(hkl)
        hkl_c = np.matmul(M, hkl)

        # Convert directions to cartesian:
        x_c = np.matmul(M, x_dir)
        o_c = np.matmul(M, orth_dir)

        # Calculate y-direction in plot (orthogonal to x-direction and out-of-plane direction)
        y_c = np.cross(x_c, o_c)

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

    plot_file = Button(label="Plot selected file(s)", button_type="primary", width=200)
    plot_file.on_click(plot_file_callback)

    plot = figure(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        plot_height=450,
        plot_width=450 + 32,
        tools="pan,wheel_zoom,reset",
    )
    plot.toolbar.logo = None

    color_mapper = LinearColorMapper(nan_color=(0, 0, 0, 0))
    image_source = ColumnDataSource(dict(image=[np.zeros((1, 1))], x=[0], y=[0], dw=[1], dh=[1]))
    plot.image(source=image_source, color_mapper=color_mapper)

    hkl_div = Div(text="HKL:", margin=(5, 5, 0, 5))
    hkl_normal = TextInput(title="normal", value="0 0 1", width=70)
    hkl_cut = Spinner(title="cut", value=0, step=0.1, width=70)
    hkl_delta = NumericInput(title="delta", value=0.1, mode="float", width=70)
    hkl_in_plane_x = TextInput(title="in-plane X", value="1 0 0", width=70)

    def redef_lattice_cb_callback(_attr, _old, new):
        if new:
            redef_lattice_ti.disabled = False
        else:
            redef_lattice_ti.disabled = True

    redef_lattice_cb = CheckboxGroup(labels=["Redefine lattice:"], width=110)
    redef_lattice_cb.on_change("active", redef_lattice_cb_callback)
    redef_lattice_ti = TextInput(width=490, disabled=True)

    def redef_ub_cb_callback(_attr, _old, new):
        if new:
            redef_ub_ti.disabled = False
        else:
            redef_ub_ti.disabled = True

    redef_ub_cb = CheckboxGroup(labels=["Redefine UB:"], width=110)
    redef_ub_cb.on_change("active", redef_ub_cb_callback)
    redef_ub_ti = TextInput(width=490, disabled=True)

    def colormap_select_callback(_attr, _old, new):
        color_mapper.palette = new

    colormap_select = Select(
        title="Colormap:",
        options=[("Greys256", "greys"), ("Plasma256", "plasma"), ("Cividis256", "cividis")],
        width=100,
    )
    colormap_select.on_change("value", colormap_select_callback)
    colormap_select.value = "Plasma256"

    def display_min_ni_callback(_attr, _old, new):
        color_mapper.low = new

    display_min_ni = NumericInput(title="Intensity min:", value=0, mode="float", width=70)
    display_min_ni.on_change("value", display_min_ni_callback)

    def display_max_ni_callback(_attr, _old, new):
        color_mapper.high = new

    display_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    display_max_ni.on_change("value", display_max_ni_callback)

    xrange_min_ni = NumericInput(title="x range min:", value=0, mode="float", width=70)
    xrange_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    xrange_step_ni = NumericInput(title="x mesh:", value=0.01, mode="float", width=70)

    yrange_min_ni = NumericInput(title="y range min:", value=0, mode="float", width=70)
    yrange_max_ni = NumericInput(title="max:", value=1, mode="float", width=70)
    yrange_step_ni = NumericInput(title="y mesh:", value=0.01, mode="float", width=70)

    def auto_range_cb_callback(_attr, _old, new):
        if new:
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

    range_layout = row(
        column(Spacer(height=19), auto_range_cb),
        xrange_min_ni,
        xrange_max_ni,
        yrange_min_ni,
        yrange_max_ni,
        Spacer(width=27),
        xrange_step_ni,
        yrange_step_ni,
    )
    cm_layout = row(colormap_select, display_min_ni, display_max_ni)
    column1_layout = column(
        row(measured_data_div, measured_data, plot_file),
        plot,
        column(
            hkl_div,
            row(
                hkl_normal,
                hkl_cut,
                hkl_delta,
                Spacer(width=10),
                hkl_in_plane_x,
                Spacer(width=50),
                cm_layout,
            ),
            row(column(Spacer(height=7), redef_lattice_cb), redef_lattice_ti),
            row(column(Spacer(height=7), redef_ub_cb), redef_ub_ti),
            range_layout,
        ),
    )
    column2_layout = app.PlotHKL().layout

    hdf_div = Div(text="<b>HDF DATA</b>")
    ccl_div = Div(text="<b>CCL DATA</b>")
    tab_layout = row(
        column(hdf_div, column1_layout), Spacer(width=70), column(ccl_div, column2_layout)
    )

    return Panel(child=tab_layout, title="plot data")
