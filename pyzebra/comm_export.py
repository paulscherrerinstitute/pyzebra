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

    align = ">"
    if data["meta"]["indices"] == "hkl":
        extension = ".comm"
        padding = [6, 4, 10, 8]
    elif data["meta"]["indices"] == "real":
        extension = ".incomm"
        padding = [4, 6, 10, 8]

    with open(str(path + extension), "w") as out_file:
        for meas_num, meas in data["Measurements"].items():
            print(meas_num)
            if meas.get("fit") is None:
                print("Measurement skipped - no fit value for:", meas_num)
                continue

            meas_number_str = f"{meas_num[1:]:{align}{padding[0]}}"
            h_str = f'{int(meas["h_index"]):{padding[1]}}'
            k_str = f'{int(meas["k_index"]):{padding[1]}}'
            l_str = f'{int(meas["l_index"]):{padding[1]}}'
            if meas["fit"]["export_fit"] is True:
                area = float(meas["fit"]["g_amp"][0]) + float(meas["fit"]["l_amp"][0])
            else:
                area = float(meas["fit"]["int_area"]) - float(meas["fit"]["int_background"][0])

            if data["meta"]["zebra_mode"] == "bi":
                int_str = f'{"{:8.2f}".format(correction(area, lorentz, data["meta"]["zebra_mode"], meas["twotheta_angle"])):{align}{padding[2]}}'
                angle_str1 = f'{meas["twotheta_angle"]:{padding[3]}}'
                angle_str2 = f'{meas["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{meas["chi_angle"]:{padding[3]}}'
                angle_str4 = f'{meas["phi_angle"]:{padding[3]}}'
            elif data["meta"]["zebra_mode"] == "nb":
                int_str = f'{"{:8.2f}".format(correction(area, lorentz, data["meta"]["zebra_mode"], meas["gamma_angle"],meas["nu_angle"])):{align}{padding[2]}}'
                angle_str1 = f'{meas["gamma_angle"]:{padding[3]}}'
                angle_str2 = f'{meas["omega_angle"]:{padding[3]}}'
                angle_str3 = f'{meas["nu_angle"]:{padding[3]}}'
                angle_str4 = f'{meas["unkwn_angle"]:{padding[3]}}'

            sigma_str = f'{"{:8.2f}".format(float(meas["fit"]["g_width"][0])):{align}{padding[2]}}'
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
