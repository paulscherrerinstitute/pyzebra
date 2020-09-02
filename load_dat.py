import re


def load_dat(filepath):

    """
    Loads *.dat file
    to add more variables to read, extend the elif list
    the file must include '#data' and number of points in right place to work properly

    :arg filepath
    :returns det_variables
    - dictionary of all detector/scan variables and 6 columns of data - omega, counts, monitor 1/2/3, time
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
                    elif variable == 'proposal':
                        det_variables["proposal"] = str(value)[:-1].strip()
                    elif variable == 'proposal_user':
                        det_variables["proposal_user"] = str(value)[:-1].strip()
                    elif variable == 'proposal_title':
                        det_variables["proposal_title"] = str(value)[:-1].strip()
                    elif variable == 'proposal_email':
                        det_variables["proposal_email"] = str(value)[:-1].strip()
                    elif variable == 'zebra_mode':
                        det_variables["zebra_mode"] = str(value)[:-1].strip()
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
                    elif variable == 'mf':
                        det_variables["mf"] = float(value)
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
                        det_variables["detectorDistance"] = str(value)
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

                    det_variables["omega"] = omega
                    det_variables["counts"] = counts
                    det_variables["Monitor1"] = monitor1
                    det_variables["Monitor2"] = monitor2
                    det_variables["Monitor3"] = monitor3
                    det_variables["time"] = time

            return det_variables
    except IOError:
        print("File not found or path is incorrect")
