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
        for keys in data["Measurements"]:
            print(keys)
            try:
                meas_number_str = f"{keys[1:]:{align}{padding[0]}}"
                h_str = f'{int(data["Measurements"][str(keys)]["h_index"]):{padding[1]}}'
                k_str = f'{int(data["Measurements"][str(keys)]["k_index"]):{padding[1]}}'
                l_str = f'{int(data["Measurements"][str(keys)]["l_index"]):{padding[1]}}'
                if data["Measurements"][str(keys)]["fit"]["export_fit"] is True:
                    area = float(data["Measurements"][str(keys)]["fit"]["g_amp"][0]) + float(
                        data["Measurements"][str(keys)]["fit"]["l_amp"][0]
                    )
                else:
                    area = float(data["Measurements"][str(keys)]["fit"]["int_area"]) - float(
                        data["Measurements"][str(keys)]["fit"]["int_background"][0]
                    )

                if data["meta"]["zebra_mode"] == "bi":
                    int_str = f'{"{:8.2f}".format(correction(area, lorentz, data["meta"]["zebra_mode"], data["Measurements"][str(keys)]["twotheta_angle"])):{align}{padding[2]}}'                    
                    angle_str1 = f'{data["Measurements"][str(keys)]["twotheta_angle"]:{padding[3]}}'
                    angle_str2 = f'{data["Measurements"][str(keys)]["omega_angle"]:{padding[3]}}'
                    angle_str3 = f'{data["Measurements"][str(keys)]["chi_angle"]:{padding[3]}}'
                    angle_str4 = f'{data["Measurements"][str(keys)]["phi_angle"]:{padding[3]}}'
                elif data["meta"]["zebra_mode"] == "nb":
                    int_str = f'{"{:8.2f}".format(correction(area, lorentz, data["meta"]["zebra_mode"], data["Measurements"][str(keys)]["gamma_angle"],data["Measurements"][str(keys)]["nu_angle"])):{align}{padding[2]}}'
                    angle_str1 = f'{data["Measurements"][str(keys)]["gamma_angle"]:{padding[3]}}'
                    angle_str2 = f'{data["Measurements"][str(keys)]["omega_angle"]:{padding[3]}}'
                    angle_str3 = f'{data["Measurements"][str(keys)]["nu_angle"]:{padding[3]}}'
                    angle_str4 = f'{data["Measurements"][str(keys)]["unkwn_angle"]:{padding[3]}}'

                sigma_str = f'{"{:8.2f}".format(float(data["Measurements"][str(keys)]["fit"]["g_width"][0])):{align}{padding[2]}}'
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

            except KeyError:
                print("Measurement skipped - no fit value for:", keys)
