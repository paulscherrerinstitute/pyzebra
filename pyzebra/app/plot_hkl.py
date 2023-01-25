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
    Legend,
    LegendItem,
    NormalHead,
    NumericInput,
    RadioGroup,
    Range1d,
    Spacer,
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
        measured_data_div = Div(text="Measured data:")
        measured_data = FileInput(accept=".ccl", multiple=True, width=200)

        min_grid_x = -10
        max_grid_x = 10
        min_grid_y = -5
        max_grid_y = 5
        cmap = Dark2[8]
        syms = ["circle", "inverted_triangle", "square", "diamond", "star", "triangle"]

        def plot_file_callback():
            orth_dir = list(map(float, hkl_normal.value.split()))
            cut_tol = hkl_delta.value
            cut_or = hkl_cut.value
            x_dir = list(map(float, hkl_in_plane_x.value.split()))
            y_dir = list(map(float, hkl_in_plane_y.value.split()))

            k = np.array(k_vectors.value.split()).astype(float).reshape(-1, 3)
            tol_k = tol_k_ni.value

            # Plotting options
            grid_flag = 1
            grid_minor_flag = 1
            grid_div = 2  # Number of minor division lines per unit

            # different symbols based on file number
            file_flag = 0 in disting_opt_cb.active
            # scale marker size according to intensity
            intensity_flag = 1 in disting_opt_cb.active
            # use color to mark different propagation vectors
            prop_legend_flag = 2 in disting_opt_cb.active
            # use resolution ellipsis
            res_flag = disting_opt_rb.active
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
                    return

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

            # Calculate in-plane y-direction
            x_c = M @ x_dir
            y_c = M @ y_dir
            o_c = M @ orth_dir

            # Normalize all directions
            y_c = y_c / np.linalg.norm(y_c)
            x_c = x_c / np.linalg.norm(x_c)
            o_c = o_c / np.linalg.norm(o_c)

            # Read all data
            hkl_coord = []
            intensity_vec = []
            k_flag_vec = []
            file_flag_vec = []
            res_vec_x = []
            res_vec_y = []
            res_N = 10

            for j, md_fname in enumerate(md_fnames):
                with io.StringIO(base64.b64decode(md_fdata[j]).decode()) as file:
                    _, ext = os.path.splitext(md_fname)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {md_fname}")
                        return

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
                    res = (
                        np.sqrt(0.4639 * expr**2 - 0.4452 * expr + 0.1506) * res_mult
                    )  # res in deg

                    # convert to resolution in hkl along scan line
                    ang2hkl_1d = pyzebra.ang2hkl_1d
                    res_x = []
                    res_y = []
                    for _om in np.linspace(om[0], om[-1], num=res_N):
                        expr1 = ang2hkl_1d(wave, gammad, _om + res / 2, chi, phi, nud, ub_inv)
                        expr2 = ang2hkl_1d(wave, gammad, _om - res / 2, chi, phi, nud, ub_inv)
                        hkl_temp = M @ (np.abs(expr1 - expr2) / 2)
                        res_x.append(hkl_temp[0])
                        res_y.append(hkl_temp[1])

                    # Get first and final hkl
                    hkl1 = ang2hkl_1d(wave, gammad, om[0], chi, phi, nud, ub_inv)
                    hkl2 = ang2hkl_1d(wave, gammad, om[-1], chi, phi, nud, ub_inv)

                    # Get hkl at best intensity
                    hkl_m = ang2hkl_1d(wave, gammad, om[np.argmax(counts)], chi, phi, nud, ub_inv)

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
                    res_vec_x.append(res_x)
                    res_vec_y.append(res_y)

            plot.x_range.start = plot.x_range.reset_start = -2
            plot.x_range.end = plot.x_range.reset_end = 5
            plot.y_range.start = plot.y_range.reset_start = -4
            plot.y_range.end = plot.y_range.reset_end = 3.5

            # Plot grid lines
            xs, ys = [], []
            xs_minor, ys_minor = [], []
            if grid_flag:
                for yy in np.arange(min_grid_y, max_grid_y, 1):
                    hkl1 = M @ [0, yy, 0]
                    xs.append([min_grid_y, max_grid_y])
                    ys.append([hkl1[1], hkl1[1]])

                for xx in np.arange(min_grid_x, max_grid_x, 1):
                    hkl1 = M @ [xx, min_grid_x, 0]
                    hkl2 = M @ [xx, max_grid_x, 0]
                    xs.append([hkl1[0], hkl2[0]])
                    ys.append([hkl1[1], hkl2[1]])

                if grid_minor_flag:
                    for yy in np.arange(min_grid_y, max_grid_y, 1 / grid_div):
                        hkl1 = M @ [0, yy, 0]
                        xs_minor.append([min_grid_y, max_grid_y])
                        ys_minor.append([hkl1[1], hkl1[1]])

                    for xx in np.arange(min_grid_x, max_grid_x, 1 / grid_div):
                        hkl1 = M @ [xx, min_grid_x, 0]
                        hkl2 = M @ [xx, max_grid_x, 0]
                        xs_minor.append([hkl1[0], hkl2[0]])
                        ys_minor.append([hkl1[1], hkl2[1]])

            grid_source.data.update(xs=xs, ys=ys)
            minor_grid_source.data.update(xs=xs_minor, ys=ys_minor)

            el_x, el_y, el_w, el_h, el_c = [], [], [], [], []
            scan_xs, scan_ys, scan_x, scan_y = [], [], [], []
            scan_m, scan_s, scan_c, scan_l = [], [], [], []
            for j in range(len(hkl_coord)):
                # Get middle hkl from list
                hklm = M @ hkl_coord[j][2]

                # Decide if point is in the cut
                proj = np.dot(hklm, o_c)
                if abs(proj - cut_or) >= cut_tol:
                    continue

                hkl1 = M @ hkl_coord[j][0]
                hkl2 = M @ hkl_coord[j][1]

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
                    # Generate series of ellipses along scan line
                    el_x.extend(np.linspace(hkl1[0], hkl2[0], num=res_N))
                    el_y.extend(np.linspace(hkl1[1], hkl2[1], num=res_N))
                    el_w.extend(np.array(res_vec_x[j]) * 2)
                    el_h.extend(np.array(res_vec_y[j]) * 2)
                    el_c.extend([col_value] * res_N)
                else:
                    # Plot scan line
                    scan_xs.append([hkl1[0], hkl2[0]])
                    scan_ys.append([hkl1[1], hkl2[1]])

                    # Plot middle point of scan
                    scan_x.append(hklm[0])
                    scan_y.append(hklm[1])
                    scan_m.append(plot_symbol)
                    scan_s.append(markersize)

                    # Color and legend label
                    scan_c.append(col_value)
                    scan_l.append(md_fnames[file_flag_vec[j]])

            ellipse_source.data.update(x=el_x, y=el_y, width=el_w, height=el_h, c=el_c)
            scan_source.data.update(
                xs=scan_xs, ys=scan_ys, x=scan_x, y=scan_y, m=scan_m, s=scan_s, c=scan_c, l=scan_l
            )

            arrow1.visible = True
            arrow1.x_end = x_c[0]
            arrow1.y_end = x_c[1]
            arrow2.visible = True
            arrow2.x_end = y_c[0]
            arrow2.y_end = y_c[1]

            kvect_source.data.update(
                x=[x_c[0] / 2, y_c[0] / 2 - 0.1], y=[x_c[1] - 0.1, y_c[1] / 2], text=["h", "k"]
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

        plot_file = Button(label="Plot selected file(s)", button_type="primary", width=200)
        plot_file.on_click(plot_file_callback)

        plot = figure(
            x_range=Range1d(),
            y_range=Range1d(),
            plot_height=450,
            plot_width=600,
            tools="pan,wheel_zoom,reset",
        )
        plot.toolbar.logo = None

        arrow1 = Arrow(
            x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10), visible=False
        )
        plot.add_layout(arrow1)
        arrow2 = Arrow(
            x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10), visible=False
        )
        plot.add_layout(arrow2)

        kvect_source = ColumnDataSource(dict(x=[], y=[], text=[]))
        plot.text(source=kvect_source)

        grid_source = ColumnDataSource(dict(xs=[], ys=[]))
        plot.multi_line(source=grid_source, line_color="gray")

        minor_grid_source = ColumnDataSource(dict(xs=[], ys=[]))
        plot.multi_line(source=minor_grid_source, line_color="gray", line_dash="dotted")

        ellipse_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[], c=[]))
        ellipse = plot.ellipse(source=ellipse_source, fill_color="c", line_color="c")

        scan_source = ColumnDataSource(dict(xs=[], ys=[], x=[], y=[], m=[], s=[], c=[], l=[]))
        mline = plot.multi_line(source=scan_source, line_color="c")
        scatter = plot.scatter(
            source=scan_source, marker="m", size="s", fill_color="c", line_color="c"
        )

        plot.add_layout(Legend(items=[], location="top_left", click_policy="hide"))

        hkl_div = Div(text="HKL:", margin=(5, 5, 0, 5))
        hkl_normal = TextInput(title="normal", value="0 0 1", width=70)
        hkl_cut = Spinner(title="cut", value=0, step=0.1, width=70)
        hkl_delta = NumericInput(title="delta", value=0.1, mode="float", width=70)
        hkl_in_plane_x = TextInput(title="in-plane X", value="1 0 0", width=70)
        hkl_in_plane_y = TextInput(title="in-plane Y", value="0 1 0", width=70)

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

        hkl_layout = column(
            hkl_div,
            row(hkl_normal, hkl_cut, hkl_delta, Spacer(width=10), hkl_in_plane_x, hkl_in_plane_y),
        )
        disting_layout = column(disting_opt_div, row(disting_opt_cb, disting_opt_rb))

        layout = column(
            row(measured_data_div, measured_data, plot_file),
            plot,
            row(hkl_layout, k_vectors),
            row(disting_layout, column(tol_k_ni, res_mult_ni)),
        )
        self.layout = layout
