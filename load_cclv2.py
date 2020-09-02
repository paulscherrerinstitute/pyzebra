import re
import numpy as np

def load_cclv2(filepath):

    """
    Loads *.ccl file
    to add more variables to read, extend the elif list
    the file must include '#data' and number of points in right place to work properly

    :arg filepath
    :returns det_variables
    - dictionary of all detector/scan variables and dictinionary for every measurement within dictionary ["Measurements"].
    Names of these dictionaries are M + measurement number. They include HKL indeces, angles, monitors,
    stepsize and array of counts
    Structure of the dictionary:
    det_variables:
        instrument
        user
        ...
        sdh1
        Measurements
            M1
            M2
            M3 {hkl, angles, number of points,..., [counts], [omega] }

    to get to counts: anyvariable["Measurements"]["M***"][counts]
    to ge to omega: anyvariable["Measurements"]["M***"][omega]
    """
    try:
        variables = 'value'
        det_variables = {"variables": variables}

        with open(filepath, 'r') as infile:
            for line in infile:

                if '=' in line:
                    variable, value = line.split('=')
                    variable = variable.strip()
                    if variable == 'instrument':
                        det_variables["instrument"] = str(value)[:-1].strip()
                    elif variable == 'title':
                        det_variables["title"] = str(value)[:-1].strip()
                    elif variable == 'sample':
                        det_variables["sample"] = str(value)[:-1]
                    elif variable == 'user':
                        det_variables["user"] = str(value)[:-1].strip()
                    elif variable == 'ProposalID':
                        det_variables["ProposalID"] = str(value)[:-1].strip()
                    elif variable == 'original_filename':
                        det_variables["original_filename"] = str(value)[:-1].strip()
                    elif variable == 'date':
                        det_variables["date"] = str(value)[:-1].strip()
                    elif variable == 'mf':
                        det_variables["mf"] = float(value)
                    elif variable == 'zebra_mode':
                        det_variables["zebra_mode"] = str(value)[:-1].strip()
                    elif variable == 'proposal':
                        det_variables["proposal"] = str(value)[:-1].strip()
                    elif variable == 'proposal_user':
                        det_variables["proposal_user"] = str(value)[:-1].strip()
                    elif variable == 'proposal_title':
                        det_variables["proposal_title"] = str(value)[:-1].strip()
                    elif variable == 'proposal_email':
                        det_variables["proposal_email"] = str(value)[:-1].strip()

                    # elif variable == 'omega':
                    #      det_variables["omega"] = float(value)
                    elif variable == '2-theta':
                        det_variables["2-theta"] = float(value)
                    elif variable == 'chi':
                        det_variables["chi"] = float(value)
                    elif variable == 'phi':
                        det_variables["phi"] = float(value)
                    elif variable == 'nu':
                        det_variables["nu"] = float(value)
                    elif variable == 'temp':
                        det_variables["temp"] = float(value)

                    elif variable == 'wavelenght':
                        det_variables["wavelength"] = float(value)
                    elif variable == 'ub1j':
                        det_variables["ub1"] = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
                    elif variable == 'ub2j':
                        det_variables["ub2"] = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
                    elif variable == 'ub3j':
                        det_variables["ub3"] = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
                    elif variable == 'a':
                        det_variables["lattice_par_a"] = float(value)
                    elif variable == 'b':
                        det_variables["lattice_par_b"] = float(value)
                    elif variable == 'c':
                        det_variables["lattice_par_c"] = float(value)
                    elif variable == 'alpha':
                        det_variables["lattice_ang_alpha"] = float(value)
                    elif variable == 'beta':
                        det_variables["lattice_ang_beta"] = float(value)
                    elif variable == 'gamma':
                        det_variables["lattice_ang_gamma"] = float(value)
                    elif variable == 'cex1':
                        det_variables["cex1"] = float(value)
                    elif variable == 'cex2':
                        det_variables["cex2"] = float(value)
                    elif variable == 'mexz':
                        det_variables["mexz"] = float(value)
                    elif variable == 'moml':
                        det_variables["moml"] = float(value)
                    elif variable == 'mcvl':
                        det_variables["mcvl"] = float(value)
                    elif variable == 'momu':
                        det_variables["momu"] = float(value)
                    elif variable == 'mcvu':
                        det_variables["mcvu"] = float(value)
                    elif variable == 'detectorDistance':
                        det_variables["detectorDistance"] = float(value)
                    elif variable == 'snv':
                        det_variables["snv"] = float(value)
                    elif variable == 'snh':
                        det_variables["snh"] = float(value)
                    elif variable == 'snvm':
                        det_variables["snvm"] = float(value)
                    elif variable == 'snhm':
                        det_variables["snhm"] = float(value)
                    elif variable == 's1vt':
                        det_variables["s1vt"] = float(value)
                    elif variable == 's1vb':
                        det_variables["s1vb"] = float(value)
                    elif variable == 's1hr':
                        det_variables["s1hr"] = float(value)
                    elif variable == 's1hl':
                        det_variables["s1hl"] = float(value)
                    elif variable == 's2vt':
                        det_variables["s2vt"] = float(value)
                    elif variable == 's2vb':
                        det_variables["s1vb"] = float(value)
                    elif variable == 's2hr':
                        det_variables["s1hr"] = float(value)
                    elif variable == 's2hl':
                        det_variables["s2hl"] = float(value)
                elif '#data' in line:
                    det_variables["Measurements"] = {}
                    data = infile.readlines()
                    num_of_measurements = data.count('om')
                    #print('Num of meas: ', num_of_measurements)
                    num_of_om = 0


                    position = - 1
                    for line in data:
                        position = position + 1
                        if bool(re.match('(\s\s\s\d)', line[0:4])) == True or bool(re.match('(\s\s\d\d)', line[0:4])) == True or bool(re.match('(\s\d\d\d)', line[0:4])) == True or bool(re.match('(\d\d\d\d)', line[0:4])) == True:
                            #print('matched value: ', line[0:4])
                            counts = []
                            num_of_om = num_of_om + 1
                            measurement_number = int(line.split()[0])
                            d = {}
                            d["h_index"] = float(line.split()[1])
                            d["k_index"] = float(line.split()[2])
                            d["l_index"] = float(line.split()[3])
                            d["twotheta_angle"] = float(line.split()[4])
                            d["omega_angle"] = float(line.split()[5])
                            d["chi_angle"] = float(line.split()[6])
                            d["phi_angle"] = float(line.split()[7])
                            next_line = data[position + 1]
                            d["number_of_measurements"] = int(next_line.split()[0])
                            d["angle_step"] = float(next_line.split()[1])
                            d["monitor"] = float(next_line.split()[2])
                            d["unkwn1"] = float(next_line.split()[3])
                            d["unkwn2"] = float(next_line.split()[4])
                            d["date"] = str(next_line.split()[5])
                            d["time"] = str(next_line.split()[6])
                            d["scan_type"] = str(next_line.split()[7])
                            for i in range(int(int(next_line.split()[0]) / 10) + (int(next_line.split()[0]) % 10 > 0)):
                                fileline = data[position + 2 + i].split()
                                numbers = [int(w) for w in fileline]
                                counts = counts + numbers
                            d["omega"] = np.linspace(float(line.split()[5])-(int(next_line.split()[0])/2)* float(next_line.split()[1]),float(line.split()[5])+(int(next_line.split()[0])/2)* float(next_line.split()[1]), int(next_line.split()[0]))
                            d["counts"] = (counts)
                            det_variables["Measurements"][str('M'+ str(measurement_number))] = d



            return det_variables
    except IOError:
        print("File not found or path is incorrect")
