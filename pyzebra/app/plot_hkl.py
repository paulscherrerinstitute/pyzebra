import base64
import io
import os

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Arrow,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    FileInput,
    HoverTool,
    Legend,
    LegendItem,
    NormalHead,
    NumericInput,
    RadioGroup,
    Spinner,
    TextAreaInput,
    TextInput,
)
from bokeh.palettes import Dark2
from bokeh.plotting import figure
from scipy.integrate import simpson, trapezoid

import pyzebra


class PlotHKL:
    def __init__(self):
        _update_slice = None
        measured_data_div = Div(text="Measured <b>CCL</b> data:")
        measured_data = FileInput(accept=".ccl", multiple=True, width=200)

        upload_hkl_div = Div(text="Open hkl/mhkl data:")
        upload_hkl_fi = FileInput(accept=".hkl,.mhkl", multiple=True, width=200)

        min_grid_x = -10
        max_grid_x = 10
        min_grid_y = -10
        max_grid_y = 10
        cmap = Dark2[8]
        syms = ["circle", "inverted_triangle", "square", "diamond", "star", "triangle"]

        def _prepare_plotting():
            orth_dir = list(map(float, hkl_normal.value.split()))
            x_dir = list(map(float, hkl_in_plane_x.value.split()))

            k = np.array(k_vectors.value.split()).astype(float).reshape(-1, 3)
            tol_k = tol_k_ni.value

            # multiplier for resolution function (in case of samples with large mosaicity)
            res_mult = res_mult_ni.value

            md_fnames = measured_data.filename
            md_fdata = measured_data.value

            # Load first data file, read angles and define matrices to perform conversion to cartesian
            # coordinates and back
            with io.StringIO(base64.b64decode(md_fdata[0]).decode()) as file:
                _, ext = os.path.splitext(md_fnames[0])
                try:
                    file_data = pyzebra.parse_1D(file, ext)
                except:
                    print(f"Error loading {md_fnames[0]}")
                    return None

            alpha = file_data[0]["alpha_cell"] * np.pi / 180.0
            beta = file_data[0]["beta_cell"] * np.pi / 180.0
            gamma = file_data[0]["gamma_cell"] * np.pi / 180.0

            # reciprocal angle parameters
            beta_star = np.arccos(
                (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))
            )
            gamma_star = np.arccos(
                (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
            )

            # conversion matrix
            M = np.array(
                [
                    [1, np.cos(gamma_star), np.cos(beta_star)],
                    [0, np.sin(gamma_star), -np.sin(beta_star) * np.cos(alpha)],
                    [0, 0, np.sin(beta_star) * np.sin(alpha)],
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

            # Save length of lattice vectors
            x_length = np.linalg.norm(x_dir)
            y_length = np.linalg.norm(y_dir)

            # Save str for labels
            xlabel_str = " ".join(map(str, x_dir))
            ylabel_str = " ".join(map(str, y_dir))

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

            # Read all data
            hkl_coord = []
            intensity_vec = []
            k_flag_vec = []
            file_flag_vec = []
            res_vec = []
            res_N = 10

            for j, md_fname in enumerate(md_fnames):
                with io.StringIO(base64.b64decode(md_fdata[j]).decode()) as file:
                    _, ext = os.path.splitext(md_fname)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {md_fname}")
                        return None

                pyzebra.normalize_dataset(file_data)

                # Loop throguh all data
                for scan in file_data:
                    om = scan["omega"]
                    gammad = scan["twotheta"]
                    chi = scan["chi"]
                    phi = scan["phi"]
                    nud = 0  # 1d detector
                    ub_inv = np.linalg.inv(scan["ub"])
                    counts = scan["counts"]
                    wave = scan["wavelength"]

                    # Calculate resolution in degrees
                    expr = np.tan(gammad / 2 * np.pi / 180)
                    fwhm = np.sqrt(0.4639 * expr**2 - 0.4452 * expr + 0.1506) * res_mult
                    res = 4 * np.pi / wave * np.sin(fwhm * np.pi / 180)

                    # Get first and final hkl
                    hkl1 = pyzebra.ang2hkl_1d(wave, gammad, om[0], chi, phi, nud, ub_inv)
                    hkl2 = pyzebra.ang2hkl_1d(wave, gammad, om[-1], chi, phi, nud, ub_inv)

                    # Get hkl at best intensity
                    hkl_m = pyzebra.ang2hkl_1d(
                        wave, gammad, om[np.argmax(counts)], chi, phi, nud, ub_inv
                    )

                    # Estimate intensity for marker size scaling
                    y_bkg = [counts[0], counts[-1]]
                    x_bkg = [om[0], om[-1]]
                    c = int(simpson(counts, x=om) - trapezoid(y_bkg, x=x_bkg))

                    # Recognize k_flag_vec
                    reduced_hkl_m = np.minimum(1 - hkl_m % 1, hkl_m % 1)
                    for ind, _k in enumerate(k):
                        if all(np.abs(reduced_hkl_m - _k) < tol_k):
                            k_flag_vec.append(ind)
                            break
                    else:
                        # not required
                        continue

                    # Save data
                    hkl_coord.append([hkl1, hkl2, hkl_m])
                    intensity_vec.append(c)
                    file_flag_vec.append(j)
                    res_vec.append(res)

            x_spacing = np.dot(M @ x_dir, x_c) * x_length
            y_spacing = np.dot(M @ y_dir, y_vert) * y_length
            y_spacingx = np.dot(M @ y_dir, x_c) * y_length

            # Plot coordinate system
            arrow1.x_end = x_spacing
            arrow1.y_end = 0
            arrow2.x_end = y_spacingx
            arrow2.y_end = y_spacing

            # Add labels
            kvect_source.data.update(
                x=[x_spacing / 4, -0.1],
                y=[x_spacing / 4 - 0.5, y_spacing / 2],
                text=[xlabel_str, ylabel_str],
            )

            # Plot grid lines
            xs, ys = [], []
            xs_minor, ys_minor = [], []
            for yy in np.arange(min_grid_y, max_grid_y, 1):
                # Calculate end and start point
                hkl1 = min_grid_x * x_dir + yy * y_dir
                hkl2 = max_grid_x * x_dir + yy * y_dir
                hkl1 = M @ hkl1
                hkl2 = M @ hkl2

                # Project points onto axes
                x1 = np.dot(x_c, hkl1) * x_length
                y1 = np.dot(y_vert, hkl1) * y_length
                x2 = np.dot(x_c, hkl2) * x_length
                y2 = np.dot(y_vert, hkl2) * y_length

                xs.append([x1, x2])
                ys.append([y1, y2])

            for xx in np.arange(min_grid_x, max_grid_x, 1):
                # Calculate end and start point
                hkl1 = xx * x_dir + min_grid_y * y_dir
                hkl2 = xx * x_dir + max_grid_y * y_dir
                hkl1 = M @ hkl1
                hkl2 = M @ hkl2

                # Project points onto axes
                x1 = np.dot(x_c, hkl1) * x_length
                y1 = np.dot(y_vert, hkl1) * y_length
                x2 = np.dot(x_c, hkl2) * x_length
                y2 = np.dot(y_vert, hkl2) * y_length

                xs.append([x1, x2])
                ys.append([y1, y2])

            for yy in np.arange(min_grid_y, max_grid_y, 0.5):
                # Calculate end and start point
                hkl1 = min_grid_x * x_dir + yy * y_dir
                hkl2 = max_grid_x * x_dir + yy * y_dir
                hkl1 = M @ hkl1
                hkl2 = M @ hkl2

                # Project points onto axes
                x1 = np.dot(x_c, hkl1) * x_length
                y1 = np.dot(y_vert, hkl1) * y_length
                x2 = np.dot(x_c, hkl2) * x_length
                y2 = np.dot(y_vert, hkl2) * y_length

                xs_minor.append([x1, x2])
                ys_minor.append([y1, y2])

            for xx in np.arange(min_grid_x, max_grid_x, 0.5):
                # Calculate end and start point
                hkl1 = xx * x_dir + min_grid_y * y_dir
                hkl2 = xx * x_dir + max_grid_y * y_dir
                hkl1 = M @ hkl1
                hkl2 = M @ hkl2

                # Project points onto axes
                x1 = np.dot(x_c, hkl1) * x_length
                y1 = np.dot(y_vert, hkl1) * y_length
                x2 = np.dot(x_c, hkl2) * x_length
                y2 = np.dot(y_vert, hkl2) * y_length

                xs_minor.append([x1, x2])
                ys_minor.append([y1, y2])

            grid_source.data.update(xs=xs, ys=ys)
            minor_grid_source.data.update(xs=xs_minor, ys=ys_minor)

            # Prepare hkl/mhkl data
            hkl_coord2 = []
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
                    hkl_coord2.append(hkl)

            def _update_slice():
                cut_tol = hkl_delta.value
                cut_or = hkl_cut.value

                # different symbols based on file number
                file_flag = 0 in disting_opt_cb.active
                # scale marker size according to intensity
                intensity_flag = 1 in disting_opt_cb.active
                # use color to mark different propagation vectors
                prop_legend_flag = 2 in disting_opt_cb.active
                # use resolution ellipsis
                res_flag = disting_opt_rb.active

                el_x, el_y, el_w, el_h, el_c = [], [], [], [], []
                scan_xs, scan_ys, scan_x, scan_y = [], [], [], []
                scan_m, scan_s, scan_c, scan_l, scan_hkl = [], [], [], [], []
                for j in range(len(hkl_coord)):
                    # Get middle hkl from list
                    hklm = M @ hkl_coord[j][2]

                    # Decide if point is in the cut
                    proj = np.dot(hklm, o_c)
                    if abs(proj - cut_or) >= cut_tol:
                        continue

                    hkl1 = M @ hkl_coord[j][0]
                    hkl2 = M @ hkl_coord[j][1]

                    # Project onto axes
                    hkl1x = np.dot(hkl1, x_c)
                    hkl1y = np.dot(hkl1, y_vert)
                    hkl2x = np.dot(hkl2, x_c)
                    hkl2y = np.dot(hkl2, y_vert)
                    hklmx = np.dot(hklm, x_c)
                    hklmy = np.dot(hklm, y_vert)

                    if intensity_flag:
                        markersize = max(1, int(intensity_vec[j] / max(intensity_vec) * 20))
                    else:
                        markersize = 4

                    if file_flag:
                        plot_symbol = syms[file_flag_vec[j]]
                    else:
                        plot_symbol = "circle"

                    if prop_legend_flag:
                        col_value = cmap[k_flag_vec[j]]
                    else:
                        col_value = "black"

                    if res_flag:
                        # Generate series of circles along scan line
                        res = res_vec[j]
                        el_x.extend(np.linspace(hkl1x, hkl2x, num=res_N))
                        el_y.extend(np.linspace(hkl1y, hkl2y, num=res_N))
                        el_w.extend([res / 2] * res_N)
                        el_h.extend([res / 2] * res_N)
                        el_c.extend([col_value] * res_N)
                    else:
                        # Plot scan line
                        scan_xs.append([hkl1x, hkl2x])
                        scan_ys.append([hkl1y, hkl2y])

                        # Plot middle point of scan
                        scan_x.append(hklmx)
                        scan_y.append(hklmy)
                        scan_m.append(plot_symbol)
                        scan_s.append(markersize)

                        # Color and legend label
                        scan_c.append(col_value)
                        scan_l.append(md_fnames[file_flag_vec[j]])
                        scan_hkl.append(hkl_coord[j][2])

                ellipse_source.data.update(x=el_x, y=el_y, width=el_w, height=el_h, c=el_c)
                scan_source.data.update(
                    xs=scan_xs,
                    ys=scan_ys,
                    x=scan_x,
                    y=scan_y,
                    m=scan_m,
                    s=scan_s,
                    c=scan_c,
                    l=scan_l,
                    hkl=scan_hkl,
                )

                # Legend items for different file entries (symbol)
                legend_items = []
                if not res_flag and file_flag:
                    labels, inds = np.unique(scan_source.data["l"], return_index=True)
                    for label, ind in zip(labels, inds):
                        legend_items.append(LegendItem(label=label, renderers=[scatter], index=ind))

                # Legend items for propagation vector (color)
                if prop_legend_flag:
                    if res_flag:
                        source, render = ellipse_source, ellipse
                    else:
                        source, render = scan_source, mline

                    labels, inds = np.unique(source.data["c"], return_index=True)
                    for label, ind in zip(labels, inds):
                        label = f"k={k[cmap.index(label)]}"
                        legend_items.append(LegendItem(label=label, renderers=[render], index=ind))

                plot.legend.items = legend_items

                scan_x2, scan_y2, scan_hkl2 = [], [], []
                for j in range(len(hkl_coord2)):
                    # Get middle hkl from list
                    hklm = M @ hkl_coord2[j]

                    # Decide if point is in the cut
                    proj = np.dot(hklm, o_c)
                    if abs(proj - cut_or) >= cut_tol:
                        continue

                    # Project onto axes
                    hklmx = np.dot(hklm, x_c)
                    hklmy = np.dot(hklm, y_vert)

                    scan_x2.append(hklmx)
                    scan_y2.append(hklmy)
                    scan_hkl2.append(hkl_coord2[j])

                scatter_source2.data.update(x=scan_x2, y=scan_y2, hkl=scan_hkl2)

            return _update_slice

        def plot_file_callback():
            nonlocal _update_slice
            _update_slice = _prepare_plotting()
            _update_slice()

        plot_file = Button(label="Plot selected file(s)", button_type="primary", width=200)
        plot_file.on_click(plot_file_callback)

        plot = figure(plot_height=550, plot_width=550 + 32, tools="pan,wheel_zoom,reset")
        plot.toolbar.logo = None

        plot.xaxis.visible = False
        plot.xgrid.visible = False
        plot.yaxis.visible = False
        plot.ygrid.visible = False

        arrow1 = Arrow(x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10))
        plot.add_layout(arrow1)
        arrow2 = Arrow(x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10))
        plot.add_layout(arrow2)

        kvect_source = ColumnDataSource(dict(x=[], y=[], text=[]))
        plot.text(source=kvect_source)

        grid_source = ColumnDataSource(dict(xs=[], ys=[]))
        plot.multi_line(source=grid_source, line_color="gray")

        minor_grid_source = ColumnDataSource(dict(xs=[], ys=[]))
        plot.multi_line(source=minor_grid_source, line_color="gray", line_dash="dotted")

        ellipse_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[], c=[]))
        ellipse = plot.ellipse(source=ellipse_source, fill_color="c", line_color="c")

        scan_source = ColumnDataSource(
            dict(xs=[], ys=[], x=[], y=[], m=[], s=[], c=[], l=[], hkl=[])
        )
        mline = plot.multi_line(source=scan_source, line_color="c")
        scatter = plot.scatter(
            source=scan_source, marker="m", size="s", fill_color="c", line_color="c"
        )

        scatter_source2 = ColumnDataSource(dict(x=[], y=[], hkl=[]))
        scatter2 = plot.scatter(
            source=scatter_source2, size=4, fill_color="green", line_color="green"
        )

        plot.x_range.renderers = [ellipse, mline, scatter, scatter2]
        plot.y_range.renderers = [ellipse, mline, scatter, scatter2]

        plot.add_layout(Legend(items=[], location="top_left", click_policy="hide"))

        plot.add_tools(HoverTool(renderers=[scatter, scatter2], tooltips=[("hkl", "@hkl")]))

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

        disting_opt_div = Div(text="Distinguish options:", margin=(5, 5, 0, 5))
        disting_opt_cb = CheckboxGroup(
            labels=["files (symbols)", "intensities (size)", "k vectors nucl/magn (colors)"],
            active=[0, 1, 2],
            width=200,
        )
        disting_opt_rb = RadioGroup(
            labels=["scan direction", "resolution ellipsoid"], active=0, width=200
        )

        k_vectors = TextAreaInput(
            title="k vectors:", value="0.0 0.0 0.0\n0.5 0.0 0.0\n0.5 0.5 0.0", width=150
        )
        res_mult_ni = NumericInput(title="Resolution mult:", value=10, mode="int", width=100)
        tol_k_ni = NumericInput(title="k tolerance:", value=0.01, mode="float", width=100)

        def show_legend_cb_callback(_attr, _old, new):
            plot.legend.visible = bool(new)

        show_legend_cb = CheckboxGroup(labels=["Show legend"], active=[0])
        show_legend_cb.on_change("active", show_legend_cb_callback)

        layout = column(
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
                    k_vectors,
                    row(tol_k_ni, res_mult_ni),
                    disting_opt_div,
                    disting_opt_cb,
                    disting_opt_rb,
                    show_legend_cb,
                ),
            ),
        )
        self.layout = layout
