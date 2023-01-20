import base64
import io
import os
import subprocess
import tempfile

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
    MultiSelect,
    NormalHead,
    NumericInput,
    Panel,
    RadioGroup,
    Range1d,
    Select,
    Spacer,
    Spinner,
    TextAreaInput,
    TextInput,
)
from bokeh.palettes import Dark2
from bokeh.plotting import figure

import pyzebra
from pyzebra import app

ANG_CHUNK_DEFAULTS = {"2theta": 30, "gamma": 30, "omega": 30, "chi": 35, "phi": 35, "nu": 10}
SORT_OPT_BI = ["2theta", "chi", "phi", "omega"]
SORT_OPT_NB = ["gamma", "nu", "omega"]


def create():
    ang_lims = {}
    cif_data = {}
    params = {}
    res_files = {}
    app_dlfiles = app.DownloadFiles(n_files=1)

    anglim_div = Div(text="Angular min/max limits:", margin=(5, 5, 0, 5))
    sttgamma_ti = TextInput(title="stt/gamma", width=100)
    omega_ti = TextInput(title="omega", width=100)
    chinu_ti = TextInput(title="chi/nu", width=100)
    phi_ti = TextInput(title="phi", width=100)

    def _update_ang_lims(ang_lims):
        sttgamma_ti.value = " ".join(ang_lims["gamma"][:2])
        omega_ti.value = " ".join(ang_lims["omega"][:2])
        if ang_lims["geom"] == "nb":
            chinu_ti.value = " ".join(ang_lims["nu"][:2])
            phi_ti.value = ""
        else:  # ang_lims["geom"] == "bi"
            chinu_ti.value = " ".join(ang_lims["chi"][:2])
            phi_ti.value = " ".join(ang_lims["phi"][:2])

    def _update_params(params):
        if "WAVE" in params:
            wavelen_input.value = params["WAVE"]
        if "SPGR" in params:
            cryst_space_group.value = params["SPGR"]
        if "CELL" in params:
            cryst_cell.value = params["CELL"]
        if "UBMAT" in params:
            ub_matrix.value = " ".join(params["UBMAT"])
        if "HLIM" in params:
            ranges_hkl.value = params["HLIM"]
        if "SRANG" in params:
            ranges_srang.value = params["SRANG"]
        if "lattiCE" in params:
            magstruct_lattice.value = params["lattiCE"]
        if "kvect" in params:
            magstruct_kvec.value = params["kvect"]

    def open_geom_callback(_attr, _old, new):
        nonlocal ang_lims
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            ang_lims = pyzebra.read_geom_file(fileobj)
        _update_ang_lims(ang_lims)

    open_geom_div = Div(text="Open GEOM:")
    open_geom = FileInput(accept=".geom", width=200)
    open_geom.on_change("value", open_geom_callback)

    def open_cfl_callback(_attr, _old, new):
        nonlocal params
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            params = pyzebra.read_cfl_file(fileobj)
            _update_params(params)

    open_cfl_div = Div(text="Open CFL:")
    open_cfl = FileInput(accept=".cfl", width=200)
    open_cfl.on_change("value", open_cfl_callback)

    def open_cif_callback(_attr, _old, new):
        nonlocal cif_data
        with io.StringIO(base64.b64decode(new).decode()) as fileobj:
            cif_data = pyzebra.read_cif_file(fileobj)
            _update_params(cif_data)

    open_cif_div = Div(text="Open CIF:")
    open_cif = FileInput(accept=".cif", width=200)
    open_cif.on_change("value", open_cif_callback)

    wavelen_div = Div(text="Wavelength:", margin=(5, 5, 0, 5))
    wavelen_input = TextInput(title="value", width=70)

    def wavelen_select_callback(_attr, _old, new):
        if new:
            wavelen_input.value = new
        else:
            wavelen_input.value = ""

    wavelen_select = Select(
        title="preset", options=["", "0.788", "1.178", "1.383", "2.305"], width=70
    )
    wavelen_select.on_change("value", wavelen_select_callback)

    cryst_div = Div(text="Crystal structure:", margin=(5, 5, 0, 5))
    cryst_space_group = TextInput(title="space group", width=100)
    cryst_cell = TextInput(title="cell", width=250)

    def ub_matrix_calc_callback():
        params = dict()
        params["SPGR"] = cryst_space_group.value
        params["CELL"] = cryst_cell.value
        ub = pyzebra.calc_ub_matrix(params)
        ub_matrix.value = " ".join(ub)

    ub_matrix_calc = Button(label="UB matrix:", button_type="primary", width=100)
    ub_matrix_calc.on_click(ub_matrix_calc_callback)

    ub_matrix = TextInput(title="\u200B", width=600)

    ranges_div = Div(text="Ranges:", margin=(5, 5, 0, 5))
    ranges_hkl = TextInput(title="HKL", value="-25 25 -25 25 -25 25", width=250)
    ranges_srang = TextInput(title="sin(​θ​)/λ", value="0.0 0.7", width=100)

    magstruct_div = Div(text="Magnetic structure:", margin=(5, 5, 0, 5))
    magstruct_lattice = TextInput(title="lattice", width=100)
    magstruct_kvec = TextAreaInput(title="k vector", width=150)

    def sorting0_callback(_attr, _old, new):
        sorting_0_dt.value = ANG_CHUNK_DEFAULTS[new]

    def sorting1_callback(_attr, _old, new):
        sorting_1_dt.value = ANG_CHUNK_DEFAULTS[new]

    def sorting2_callback(_attr, _old, new):
        sorting_2_dt.value = ANG_CHUNK_DEFAULTS[new]

    sorting_0 = Select(title="1st", width=100)
    sorting_0.on_change("value", sorting0_callback)
    sorting_0_dt = NumericInput(title="Δ", width=70)
    sorting_1 = Select(title="2nd", width=100)
    sorting_1.on_change("value", sorting1_callback)
    sorting_1_dt = NumericInput(title="Δ", width=70)
    sorting_2 = Select(title="3rd", width=100)
    sorting_2.on_change("value", sorting2_callback)
    sorting_2_dt = NumericInput(title="Δ", width=70)

    def geom_radiogroup_callback(_attr, _old, new):
        nonlocal ang_lims, params
        if new == 0:
            geom_file = pyzebra.get_zebraBI_default_geom_file()
            sort_opt = SORT_OPT_BI
        else:
            geom_file = pyzebra.get_zebraNB_default_geom_file()
            sort_opt = SORT_OPT_NB
        cfl_file = pyzebra.get_zebra_default_cfl_file()

        ang_lims = pyzebra.read_geom_file(geom_file)
        _update_ang_lims(ang_lims)
        params = pyzebra.read_cfl_file(cfl_file)
        _update_params(params)

        sorting_0.options = sorting_1.options = sorting_2.options = sort_opt
        sorting_0.value = sort_opt[0]
        sorting_1.value = sort_opt[1]
        sorting_2.value = sort_opt[2]

    geom_radiogroup_div = Div(text="Geometry:", margin=(5, 5, 0, 5))
    geom_radiogroup = RadioGroup(labels=["bisecting", "normal beam"], width=150)
    geom_radiogroup.on_change("active", geom_radiogroup_callback)
    geom_radiogroup.active = 0

    def go_button_callback():
        ang_lims["gamma"][0], ang_lims["gamma"][1] = sttgamma_ti.value.strip().split()
        ang_lims["omega"][0], ang_lims["omega"][1] = omega_ti.value.strip().split()
        if ang_lims["geom"] == "nb":
            ang_lims["nu"][0], ang_lims["nu"][1] = chinu_ti.value.strip().split()
        else:  # ang_lims["geom"] == "bi"
            ang_lims["chi"][0], ang_lims["chi"][1] = chinu_ti.value.strip().split()
            ang_lims["phi"][0], ang_lims["phi"][1] = phi_ti.value.strip().split()

        if cif_data:
            params.update(cif_data)

        params["WAVE"] = wavelen_input.value
        params["SPGR"] = cryst_space_group.value
        params["CELL"] = cryst_cell.value
        params["UBMAT"] = ub_matrix.value.split()
        params["HLIM"] = ranges_hkl.value
        params["SRANG"] = ranges_srang.value
        params["lattiCE"] = magstruct_lattice.value
        kvects = magstruct_kvec.value.split("\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            geom_path = os.path.join(temp_dir, "zebra.geom")
            if open_geom.value:
                geom_template = io.StringIO(base64.b64decode(open_geom.value).decode())
            else:
                geom_template = None
            pyzebra.export_geom_file(geom_path, ang_lims, geom_template)

            print(f"Content of {geom_path}:")
            with open(geom_path) as f:
                print(f.read())

            priority = [sorting_0.value, sorting_1.value, sorting_2.value]
            chunks = [sorting_0_dt.value, sorting_1_dt.value, sorting_2_dt.value]
            if geom_radiogroup.active == 0:
                sort_hkl_file = pyzebra.sort_hkl_file_bi
                priority.extend(set(SORT_OPT_BI) - set(priority))
            else:
                sort_hkl_file = pyzebra.sort_hkl_file_nb

            # run sxtal_refgen for each kvect provided
            for i, kvect in enumerate(kvects, start=1):
                params["kvect"] = kvect
                if open_cfl.filename:
                    base_fname = f"{os.path.splitext(open_cfl.filename)[0]}_{i}"
                else:
                    base_fname = f"zebra_{i}"

                cfl_path = os.path.join(temp_dir, base_fname + ".cfl")
                if open_cfl.value:
                    cfl_template = io.StringIO(base64.b64decode(open_cfl.value).decode())
                else:
                    cfl_template = None
                pyzebra.export_cfl_file(cfl_path, params, cfl_template)

                print(f"Content of {cfl_path}:")
                with open(cfl_path) as f:
                    print(f.read())

                comp_proc = subprocess.run(
                    [pyzebra.SXTAL_REFGEN_PATH, cfl_path],
                    cwd=temp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print(" ".join(comp_proc.args))
                print(comp_proc.stdout)

                if i == 1:  # all hkl files are identical, so keep only one
                    hkl_fname = base_fname + ".hkl"
                    hkl_fpath = os.path.join(temp_dir, hkl_fname)
                    with open(hkl_fpath) as f:
                        res_files[hkl_fname] = f.read()

                    hkl_fname_sorted = base_fname + "_sorted.hkl"
                    hkl_fpath_sorted = os.path.join(temp_dir, hkl_fname_sorted)
                    sort_hkl_file(hkl_fpath, hkl_fpath_sorted, priority, chunks)
                    with open(hkl_fpath_sorted) as f:
                        res_files[hkl_fname_sorted] = f.read()

                mhkl_fname = base_fname + ".mhkl"
                mhkl_fpath = os.path.join(temp_dir, mhkl_fname)
                with open(mhkl_fpath) as f:
                    res_files[mhkl_fname] = f.read()

                mhkl_fname_sorted = base_fname + "_sorted.mhkl"
                mhkl_fpath_sorted = os.path.join(temp_dir, hkl_fname_sorted)
                sort_hkl_file(mhkl_fpath, mhkl_fpath_sorted, priority, chunks)
                with open(mhkl_fpath_sorted) as f:
                    res_files[mhkl_fname_sorted] = f.read()

            created_lists.options = list(res_files)

    go_button = Button(label="GO", button_type="primary", width=50)
    go_button.on_click(go_button_callback)

    def created_lists_callback(_attr, _old, new):
        sel_file = new[0]
        file_text = res_files[sel_file]
        preview_lists.value = file_text
        app_dlfiles.set_contents([file_text])
        app_dlfiles.set_names([sel_file])

    created_lists = MultiSelect(title="Created lists:", width=200, height=150)
    created_lists.on_change("value", created_lists_callback)
    preview_lists = TextAreaInput(title="Preview selected list:", width=600, height=150)

    plot_list = Button(label="Plot selected list", button_type="primary", width=200, disabled=True)

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
        tol_k = 0.1

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

            # Loop throguh all data
            for scan in file_data:
                om = scan["omega"]
                gammad = scan["twotheta"]
                chi = scan["chi"]
                phi = scan["phi"]
                nud = 0  # 1d detector
                ub = scan["ub"]
                counts = scan["counts"]
                mon = scan["monitor"]
                wave = scan["wavelength"]

                # Calculate resolution in degrees
                expr = np.tan(gammad / 2 * np.pi / 180)
                res = np.sqrt(0.4639 * expr**2 - 0.4452 * expr + 0.1506) * res_mult  # res in deg

                # convert to resolution in hkl along scan line
                ang2hkl_1d = pyzebra.ang2hkl_1d
                res_x = []
                res_y = []
                for _om in np.linspace(om[0], om[-1], num=res_N):
                    expr1 = ang2hkl_1d(wave, gammad, _om + res / 2, chi, phi, nud, ub)
                    expr2 = ang2hkl_1d(wave, gammad, _om - res / 2, chi, phi, nud, ub)
                    hkl_temp = M @ (np.abs(expr1 - expr2) / 2)
                    res_x.append(hkl_temp[0])
                    res_y.append(hkl_temp[1])

                # Get first and final hkl
                hkl1 = ang2hkl_1d(wave, gammad, om[0], chi, phi, nud, ub)
                hkl2 = ang2hkl_1d(wave, gammad, om[-1], chi, phi, nud, ub)

                # Get hkl at best intensity
                hkl_m = ang2hkl_1d(wave, gammad, om[np.argmax(counts)], chi, phi, nud, ub)

                # Estimate intensity for marker size scaling
                y1 = counts[0]
                y2 = counts[-1]
                x1 = om[0]
                x2 = om[-1]
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1
                intensity_exp = np.sum(counts - (a * om + b))
                c = int(intensity_exp / mon * 10000)

                # Recognize k_flag_vec
                min_hkl_m = np.minimum(1 - hkl_m % 1, hkl_m % 1)
                for j2, _k in enumerate(k):
                    if all(np.abs(min_hkl_m - _k) < tol_k):
                        k_flag_vec.append(j2)
                        break
                else:
                    k_flag_vec.append(len(k))

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

    arrow1 = Arrow(x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10), visible=False)
    plot.add_layout(arrow1)
    arrow2 = Arrow(x_start=0, y_start=0, x_end=0, y_end=0, end=NormalHead(size=10), visible=False)
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
    scatter = plot.scatter(source=scan_source, marker="m", size="s", fill_color="c", line_color="c")

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

    fileinput_layout = row(open_cfl_div, open_cfl, open_cif_div, open_cif, open_geom_div, open_geom)

    geom_layout = column(geom_radiogroup_div, geom_radiogroup)
    wavelen_layout = column(wavelen_div, row(wavelen_select, wavelen_input))
    anglim_layout = column(anglim_div, row(sttgamma_ti, omega_ti, chinu_ti, phi_ti))
    cryst_layout = column(cryst_div, row(cryst_space_group, cryst_cell))
    ubmat_layout = row(column(Spacer(height=18), ub_matrix_calc), ub_matrix)
    ranges_layout = column(ranges_div, row(ranges_hkl, ranges_srang))
    magstruct_layout = column(magstruct_div, row(magstruct_lattice, magstruct_kvec))
    sorting_layout = row(
        sorting_0,
        sorting_0_dt,
        Spacer(width=30),
        sorting_1,
        sorting_1_dt,
        Spacer(width=30),
        sorting_2,
        sorting_2_dt,
    )

    column1_layout = column(
        fileinput_layout,
        Spacer(height=10),
        row(geom_layout, wavelen_layout, Spacer(width=50), anglim_layout),
        cryst_layout,
        ubmat_layout,
        row(ranges_layout, Spacer(width=50), magstruct_layout),
        row(sorting_layout, Spacer(width=30), column(Spacer(height=18), go_button)),
        row(created_lists, preview_lists),
        row(app_dlfiles.button, plot_list),
    )

    hkl_layout = column(
        hkl_div,
        row(hkl_normal, hkl_cut, hkl_delta, Spacer(width=10), hkl_in_plane_x, hkl_in_plane_y),
    )
    disting_layout = column(disting_opt_div, row(disting_opt_cb, disting_opt_rb))

    column2_layout = column(
        row(measured_data_div, measured_data, plot_file),
        plot,
        row(hkl_layout, k_vectors),
        row(disting_layout, res_mult_ni),
    )

    tab_layout = row(column1_layout, column2_layout)

    return Panel(child=tab_layout, title="ccl prepare")
