import re
import numpy as np

META_VARS_STR = (
    "instrument",
    "title",
    "sample",
    "user",
    "ProposalID",
    "original_filename",
    "date",
    "zebra_mode",
    "proposal",
    "proposal_user",
    "proposal_title",
    "proposal_email",
)
META_VARS_FLOAT = (
    "mf",
    "2-theta",
    "chi",
    "phi",
    "nu",
    "temp",
    "wavelenght",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "cex1",
    "cex2",
    "mexz",
    "moml",
    "mcvl",
    "momu",
    "mcvu",
    "detectorDistance",
    "snv",
    "snh",
    "snvm",
    "snhm",
    "s1vt",
    "s1vb",
    "s1hr",
    "s1hl",
    "s2vt",
    "s2vb",
    "s2hr",
    "s2hl",
)
META_UB_MATRIX = ("ub1j", "ub2j", "ub3j")


def load_1D(filepath):
    """
    Loads *.ccl or *.dat file (Distinguishes them based on last 3 chars in string of filepath
    to add more variables to read, extend the elif list
    the file must include '#data' and number of points in right place to work properly

    :arg filepath
    :returns det_variables
    - dictionary of all detector/scan variables and dictinionary for every measurement.
    Names of these dictionaries are M + measurement number. They include HKL indeces, angles,
    monitors, stepsize and array of counts
    """
    det_variables = {"file_type": str(filepath)[-3:], "meta": {}}
    with open(filepath, "r") as infile:
        for line in infile:
            det_variables["Measurements"] = {}
            if "=" in line:
                variable, value = line.split("=")
                variable = variable.strip()
                if variable in META_VARS_FLOAT:
                    det_variables["meta"][variable] = float(value)
                elif variable in META_VARS_STR:
                    det_variables["meta"][variable] = str(value)[:-1].strip()
                elif variable in META_UB_MATRIX:
                    det_variables["meta"][variable] = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))

            elif "#data" in line:
                if det_variables["file_type"] == "ccl":
                    data = infile.readlines()
                    position = -1
                    for lines in data:
                        position = position + 1
                        if (
                            bool(re.match("(\s\s\s\d)", lines[0:4])) == True
                            or bool(re.match("(\s\s\d\d)", lines[0:4])) == True
                            or bool(re.match("(\s\d\d\d)", lines[0:4])) == True
                            or bool(re.match("(\d\d\d\d)", lines[0:4])) == True
                        ):
                            counts = []
                            measurement_number = int(lines.split()[0])
                            d = {}
                            d["h_index"] = float(lines.split()[1])
                            d["k_index"] = float(lines.split()[2])
                            d["l_index"] = float(lines.split()[3])
                            if det_variables["meta"]["zebra_mode"] == "bi":
                                d["twotheta_angle"] = float(lines.split()[4])  # gamma
                                d["omega_angle"] = float(lines.split()[5])  # omega
                                d["chi_angle"] = float(lines.split()[6])  # nu
                                d["phi_angle"] = float(lines.split()[7])  # doesnt matter
                            elif det_variables["meta"]["zebra_mode"] == "nb":
                                d["gamma_angle"] = float(lines.split()[4])  # gamma
                                d["omega_angle"] = float(lines.split()[5])  # omega
                                d["nu_angle"] = float(lines.split()[6])  # nu

                            next_line = data[position + 1]
                            d["number_of_measurements"] = int(next_line.split()[0])
                            d["angle_step"] = float(next_line.split()[1])
                            d["monitor"] = float(next_line.split()[2])
                            d["unkwn1"] = float(next_line.split()[3])
                            d["unkwn2"] = float(next_line.split()[4])
                            d["date"] = str(next_line.split()[5])
                            d["time"] = str(next_line.split()[6])
                            d["scan_type"] = str(next_line.split()[7])
                            for i in range(
                                int(int(next_line.split()[0]) / 10)
                                + (int(next_line.split()[0]) % 10 > 0)
                            ):
                                fileline = data[position + 2 + i].split()
                                numbers = [int(w) for w in fileline]
                                counts = counts + numbers
                            d["omega"] = np.linspace(
                                float(lines.split()[5])
                                - (int(next_line.split()[0]) / 2) * float(next_line.split()[1]),
                                float(lines.split()[5])
                                + (int(next_line.split()[0]) / 2) * float(next_line.split()[1]),
                                int(next_line.split()[0]),
                            )
                            d["counts"] = counts
                            det_variables["Measurements"][str("M" + str(measurement_number))] = d

                elif det_variables["file_type"] == "dat":
                    data = infile.readlines()
                    num_of_points = int(data[1].split()[0])
                    omega = []
                    counts = []
                    monitor1 = []
                    monitor2 = []
                    monitor3 = []
                    time = []
                    for position in range(num_of_points):
                        omega.append(float(data[position + 3].split()[1]))
                        counts.append(float(data[position + 3].split()[2]))
                        monitor1.append(float(data[position + 3].split()[3]))
                        monitor2.append(float(data[position + 3].split()[4]))
                        monitor3.append(float(data[position + 3].split()[5]))
                        time.append(float(data[position + 3].split()[6]))
                    det_variables["Measurements"]["omega"] = omega
                    det_variables["Measurements"]["counts"] = counts
                    det_variables["Measurements"]["Monitor1"] = monitor1
                    det_variables["Measurements"]["Monitor2"] = monitor2
                    det_variables["Measurements"]["Monitor3"] = monitor3
                    det_variables["Measurements"]["time"] = time
                else:
                    print("Unknown file extention")

    return det_variables
