path = 'C:\\Users\\Jakub\\Desktop\\'
name = 'test'
extension = '.comm'


def fill_string(string, total_lenght):
    """fills string to tatal_length with whitespaces
    :arg string - string to fill
    :arg total_lenght - length of the final string
    :return string of the lenght of total_lenght
    """
    white_spaces = ' '*total_lenght
    return_string = white_spaces[:-len(str(string))] + str(string)
    return return_string


def export_comm(data, name):
    """exports data in the *.comm format
    :arg data - data to export, is dict after peak fitting
    arg: name - name of the exported file
    """
    out_file = open(str(path+name+extension), "w")
    for keys in data["Measurements"]:
        try:
            meas_number_str = fill_string(keys[1:], 6)
            h_str = fill_string(int(data["Measurements"][str(keys)]['h_index']), 4)
            k_str = fill_string(int(data["Measurements"][str(keys)]['k_index']), 4)
            l_str = fill_string(int(data["Measurements"][str(keys)]['l_index']), 4)
            int_str = fill_string("{:10.2f}".format(data["Measurements"][str(keys)]['fit']['g_amp'][0]), 10)
            sigma_str = fill_string("{:10.2f}".format(data["Measurements"][str(keys)]['fit']['g_width'][0]), 10)
            twotheta_str = fill_string("{:8.2f}".format(data["Measurements"][str(keys)]["twotheta_angle"]), 8)
            omega_str = fill_string("{:8.2f}".format(data["Measurements"][str(keys)]["omega_angle"]), 8)
            chi_str = fill_string("{:8.2f}".format(data["Measurements"][str(keys)]["chi_angle"]), 8)
            phi_str = fill_string("{:8.2f}".format(data["Measurements"][str(keys)]["phi_angle"]), 8)
            line = meas_number_str + h_str + l_str + k_str + int_str + sigma_str + twotheta_str + omega_str + chi_str + phi_str + "\n"
            out_file.write(line)
            print('Measurement written:', keys)

        except KeyError:
            print('Measurement skipped - no fit value for:', keys)

    out_file.close()

