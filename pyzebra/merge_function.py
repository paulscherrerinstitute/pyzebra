import numpy as np
import uncertainties as u


def create_tuples(x, y, y_err):
    """creates tuples for sorting and merginng of the data
    Counts need to be normalized to monitor before"""
    t = list()
    for i in range(len(x)):
        tup = (x[i], y[i], y_err[i])
        t.append(tup)
    return t


def normalize_all(dictionary, monitor=100000):
    for scan in dictionary["scan"]:
        counts = np.array(scan["Counts"])
        sigma = np.sqrt(counts) if "sigma" not in scan else scan["sigma"]
        monitor_ratio = monitor / scan["monitor"]
        scan["Counts"] = counts * monitor_ratio
        scan["sigma"] = np.array(sigma) * monitor_ratio
        scan["monitor"] = monitor
    print("Normalized %d scans to monitor %d" % (len(dictionary["scan"]), monitor))


def merge(scan1, scan2):
    """merges the two tuples and sorts them, if om value is same, Counts value is average
    averaging is propagated into sigma if dict1 == dict2, key[1] is deleted after merging
    :arg dict1 : dictionary to which measurement will be merged
    :arg dict2 : dictionary from which measurement will be merged
    :arg scand_dict_result : result of scan_dict after auto function
    :arg keep : if true, when monitors are same, does not change it, if flase, takes monitor
    always
    :arg monitor : final monitor after merging
    note: dict1 and dict2 can be same dict
    :return dict1 with merged scan"""

    # load om and Counts
    x1, x2 = scan1["om"], scan2["om"]
    # print(scan1["om"])
    # print(scan2["om"])
    cor_y1, y_err1 = scan1["Counts"], scan1["sigma"]
    cor_y2, y_err2 = scan2["Counts"], scan2["sigma"]
    # creates touples (om, Counts, sigma) for sorting and further processing
    tuple_list = create_tuples(x1, cor_y1, y_err1) + create_tuples(x2, cor_y2, y_err2)
    # Sort the list on om and add 0 0 0 tuple to the last position
    sorted_t = sorted(tuple_list, key=lambda tup: tup[0])
    sorted_t.append((0, 0, 0))
    om, Counts, sigma = [], [], []
    seen = list()
    for i in range(len(sorted_t) - 1):
        if sorted_t[i][0] not in seen:
            if sorted_t[i][0] != sorted_t[i + 1][0]:
                om = np.append(om, sorted_t[i][0])
                Counts = np.append(Counts, sorted_t[i][1])
                sigma = np.append(sigma, sorted_t[i][2])
            else:
                om = np.append(om, sorted_t[i][0])
                counts1, counts2 = sorted_t[i][1], sorted_t[i + 1][1]
                sigma1, sigma2 = sorted_t[i][2], sorted_t[i + 1][2]
                count_err1 = u.ufloat(counts1, sigma1)
                count_err2 = u.ufloat(counts2, sigma2)
                avg = (count_err1 + count_err2) / 2
                Counts = np.append(Counts, avg.n)
                sigma = np.append(sigma, avg.s)
                seen.append(sorted_t[i][0])
        else:
            continue
    scan1["om"] = om
    scan1["Counts"] = Counts
    scan1["sigma"] = sigma
    if "history" not in scan1:
        scan1["history"] = str("Merged with scan %d" % scan2["idx"])
    else:
        scan1["history"] = scan1["history"] + str(", merged with scan %d" % scan2["idx"])
    print("merging done")


def check_UB(dict1, dict2, precision=0.01):
    return np.max(np.abs(dict1["meta"]["ub"] - dict2["meta"]["ub"])) < precision


def check_zebramode(dict1, dict2):
    if dict1["meta"]["zebra_mode"] == dict2["meta"]["zebra_mode"]:
        return True
    else:
        return False


def check_angles(scan1, scan2, angles, precision):
    truth_list = list()
    for item in angles:
        if abs(abs(scan1[item]) - abs(scan2[item])) <= precision[item]:
            truth_list.append(True)
        else:
            truth_list.append(False)
    if all(truth_list):
        return True
    else:
        return False


def check_temp_mag(scan1, scan2):
    temp_diff = 1
    mag_diff = 0.001
    truth_list = list()
    try:
        if abs(abs(scan1["mf"]) - abs(scan2["mf"])) <= mag_diff:
            truth_list.append(True)
        else:
            truth_list.append(False)
    except KeyError:
        print("Magnetic field is missing")

    try:
        if abs(abs(scan1["temp"]) - abs(scan2["temp"])) <= temp_diff:
            truth_list.append(True)
        else:
            truth_list.append(False)
    except KeyError:
        print("temperature missing")

    if all(truth_list):
        return True
    else:
        return False


def merge_dups(dictionary):

    if dictionary["meta"]["data_type"] == "dat":
        return

    if dictionary["meta"]["zebra_mode"] == "bi":
        angles = ["twotheta", "omega", "chi", "phi"]
    elif dictionary["meta"]["zebra_mode"] == "nb":
        angles = ["gamma", "omega", "nu"]

    precision = {
        "twotheta": 0.1,
        "chi": 0.1,
        "nu": 0.1,
        "phi": 0.05,
        "omega": 5,
        "gamma": 0.05,
    }

    for i in range(len(dictionary["scan"])):
        for j in range(len(dictionary["scan"])):
            if i == j:
                continue
            else:
                # print(i, j)
                if check_angles(
                    dictionary["scan"][i], dictionary["scan"][j], angles, precision
                ) and check_temp_mag(dictionary["scan"][i], dictionary["scan"][j]):
                    merge(dictionary["scan"][i], dictionary["scan"][j])
                    print("merged %d with %d within the dictionary" % (i, j))

                    del dictionary["scan"][j]
                    merge_dups(dictionary)
                    break
        else:
            continue
        break


def add_scan(dict1, dict2, scan_to_add):
    max_scan = len(dict1["scan"])
    dict1["scan"].append(dict2["scan"][scan_to_add])
    if dict1.get("extra_meta") is None:
        dict1["extra_meta"] = {}
    dict1["extra_meta"][max_scan + 1] = dict2["meta"]
    del dict2["scan"][scan_to_add]


def process(dict1, dict2, angles, precision):
    # stop when the second dict is empty
    # print(dict2["scan"])
    if dict2["scan"]:
        # check UB matrixes
        if check_UB(dict1, dict2):
            # iterate over second dict and check for matches
            for i in range(len(dict2["scan"])):
                for j in range(len(dict1["scan"])):
                    if check_angles(dict1["scan"][j], dict2["scan"][i], angles, precision):
                        # angles good, see the mag and temp
                        if check_temp_mag(dict1["scan"][j], dict2["scan"][i]):
                            merge(dict1["scan"][j], dict2["scan"][i])
                            print("merged %d with %d from different dictionaries" % (i, j))
                            del dict2["scan"][i]
                            process(dict1, dict2, angles, precision)
                            break
                        else:
                            add_scan(dict1, dict2, i)
                            print("Diffrent T or M, scan added")
                            process(dict1, dict2, angles, precision)
                            break
                    else:
                        add_scan(dict1, dict2, i)
                        print("Mismatch in angles, scan added")
                        process(dict1, dict2, angles, precision)
                        break
                else:
                    continue
                break

        else:
            # ask user if he really wants to add
            print("UBs are different, do you really wish to add  datasets? Y/N")
            dict1 = add_dict(dict1, dict2)
    return


"""
    1. check for bisecting or normal beam geometry in data files; select stt, om, chi, phi for bisecting; select stt, om, nu for normal beam
    2. in the ccl files, check for identical stt, chi and nu within 0.1 degree, and, at the same time, for identical om and phi within 0.05 degree;
    3. in the dat files, check for identical stt, chi and nu within 0.1 degree, and, at the same time,
    for identical phi within 0.05 degree, and, at the same time, for identical om within 5 degree."""


def unified_merge(dict1, dict2):
    if not check_zebramode(dict1, dict2):
        print("You are trying to add two files with different zebra mdoe")
        return

    # decide angles
    if dict1["meta"]["zebra_mode"] == "bi":
        angles = ["twotheta", "omega", "chi", "phi"]
    elif dict1["meta"]["zebra_mode"] == "nb":
        angles = ["gamma", "omega", "nu"]

    # precision of angles to check
    precision = {
        "twotheta": 0.1,
        "chi": 0.1,
        "nu": 0.1,
        "phi": 0.05,
        "omega": 5,
        "gamma": 0.1,
    }
    if (dict1["meta"]["data_type"] == "ccl") and (dict2["meta"]["data_type"] == "ccl"):
        precision["omega"] = 0.05

    process(dict1, dict2, angles, precision)


def add_dict(dict1, dict2):
    """adds two dictionaries, meta of the new is saved as meata+original_filename and
    measurements are shifted to continue with numbering of first dict
    :arg dict1 : dictionarry to add to
    :arg dict2 : dictionarry from which to take the measurements
    :return dict1 : combined dictionary
    Note: dict1 must be made from ccl, otherwise we would have to change the structure of loaded
    dat file"""
    try:
        if dict1["meta"]["zebra_mode"] != dict2["meta"]["zebra_mode"]:
            print("You are trying to add scans measured with different zebra modes")
            return
    # this is for the qscan case
    except KeyError:
        print("Zebra mode not specified")
    max_measurement_dict1 = len(dict1["scan"])
    new_filenames = np.arange(
        max_measurement_dict1 + 1, max_measurement_dict1 + 1 + len(dict2["scan"])
    )

    if dict1.get("extra_meta") is None:
        dict1["extra_meta"] = {}

    new_meta_name = "meta" + str(dict2["meta"]["original_filename"])
    if new_meta_name not in dict1:
        for keys, name in zip(range(len(dict2["scan"])), new_filenames):
            dict2["scan"][keys]["file_of_origin"] = str(dict2["meta"]["original_filename"])
            dict1["scan"].append(dict2["scan"][keys])
            dict1["extra_meta"][name] = dict2["meta"]

        dict1[new_meta_name] = dict2["meta"]
    else:
        raise KeyError(
            str(
                "The file %s has alredy been added to %s"
                % (dict2["meta"]["original_filename"], dict1["meta"]["original_filename"])
            )
        )
    return dict1
