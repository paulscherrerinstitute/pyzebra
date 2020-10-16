import numpy as np


def correction(value, lorentz=True, zebra_mode="--", ang1=0, ang2=0):
    if lorentz is False:
        return value
    else:
        if zebra_mode == "bi":
            corr_value = np.abs(value * np.sin(ang1))
            return corr_value
        elif zebra_mode == "nb":
            corr_value = np.abs(value * np.sin(ang1) * np.cos(ang2))
            return corr_value


def export_comm(data, path, lorentz=False):
    """exports data in the *.comm format
    :param lorentz: perform Lorentz correction
    :param path: path to file + name
    :arg data - data to export, is dict after peak fitting

    """
    zebra_mode = data["meta"]["zebra_mode"]
    align = ">"
    if data["meta"]["indices"] == "hkl":
        extension = ".comm"
        padding = [6, 4, 10, 8]
    elif data["meta"]["indices"] == "real":
        extension = ".incomm"
        padding = [4, 6, 10, 8]

    with open(str(path + extension), "w") as out_file:
        for key, meas in data["meas"].items():
            if "fit" not in meas:
                print("Measurement skipped - no fit value for:", key)
                continue
            meas_number_str = f"{key:{align}{padding[0]}}"
            h_str = f'{int(meas["h_index"]):{padding[1]}}'
            k_str = f'{int(meas["k_index"]):{padding[1]}}'
            l_str = f'{int(meas["l_index"]):{padding[1]}}'
            if data["meta"]["area_method"] == "fit":
                area = float(meas["fit"]["fit_area"].n)
                sigma_str = (
                    f'{"{:8.2f}".format(float(meas["fit"]["fit_area"].s)):{align}{padding[2]}}'
                )
            elif data["meta"]["area_method"] == "integ":
                area = float(meas["fit"]["int_area"].n)
                sigma_str = (
                    f'{"{:8.2f}".format(float(meas["fit"]["int_area"].s)):{align}{padding[2]}}'
                )

            if zebra_mode == "bi":
                area = correction(area, lorentz, zebra_mode, meas["twotheta_angle"])
                int_str = f'{"{:8.2f}".format(area):{align}{padding[2]}}'
                angle_str1 = f'{meas["twotheta_angle"]:{padding[3]}}'
                angle_str2 = f'{meas["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{meas["chi_angle"]:{padding[3]}}'
                angle_str4 = f'{meas["phi_angle"]:{padding[3]}}'
            elif zebra_mode == "nb":
                area = correction(area, lorentz, zebra_mode, meas["gamma_angle"], meas["nu_angle"])
                int_str = f'{"{:8.2f}".format(area):{align}{padding[2]}}'
                angle_str1 = f'{meas["gamma_angle"]:{padding[3]}}'
                angle_str2 = f'{meas["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{meas["nu_angle"]:{padding[3]}}'
                angle_str4 = f'{meas["unkwn_angle"]:{padding[3]}}'

            line = (
                meas_number_str
                + h_str
                + l_str
                + k_str
                + int_str
                + sigma_str
                + angle_str1
                + angle_str2
                + angle_str3
                + angle_str4
                + "\n"
            )
            out_file.write(line)
