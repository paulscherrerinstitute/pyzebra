import numpy as np
import uncertainties as u

from .fit2 import create_uncertanities


def add_dict(dict1, dict2):
    """adds two dictionaries, meta of the new is saved as meata+original_filename and
    measurements are shifted to continue with numbering of first dict
    :arg dict1 : dictionarry to add to
    :arg dict2 : dictionarry from which to take the measurements
    :return dict1 : combined dictionary
    Note: dict1 must be made from ccl, otherwise we would have to change the structure of loaded
    dat file"""
    max_measurement_dict1 = max([int(str(keys)[1:]) for keys in dict1["Measurements"]])
    if dict2["meta"]["data_type"] == ".ccl":
        new_filenames = [
            "M" + str(x + max_measurement_dict1)
            for x in [int(str(keys)[1:]) for keys in dict2["Measurements"]]
        ]
        new_meta_name = "meta" + str(dict2["meta"]["original_filename"])
        if new_meta_name not in dict1:
            for keys, name in zip(dict2["Measurements"], new_filenames):
                dict2["Measurements"][keys]["file_of_origin"] = str(
                    dict2["meta"]["original_filename"]
                )
                dict1["Measurements"][name] = dict2["Measurements"][keys]

            dict1[new_meta_name] = dict2["meta"]

        else:
            raise KeyError(
                str(
                    "The file %s has alredy been added to %s"
                    % (dict2["meta"]["original_filename"], dict1["meta"]["original_filename"])
                )
            )
    elif dict2["meta"]["data_type"] == ".dat":
        d = {}
        new_name = "M" + str(max_measurement_dict1 + 1)
        hkl = dict2["meta"]["title"]
        d["h_index"] = float(hkl.split()[-3])
        d["k_index"] = float(hkl.split()[-2])
        d["l_index"] = float(hkl.split()[-1])
        d["number_of_measurements"] = len(dict2["Measurements"]["NP"])
        d["om"] = dict2["Measurements"]["om"]
        d["Counts"] = dict2["Measurements"]["Counts"]
        d["monitor"] = dict2["Measurements"]["Monitor1"][0]
        d["temperature"] = dict2["meta"]["temp"]
        d["mag_field"] = dict2["meta"]["mf"]
        d["omega_angle"] = dict2["meta"]["omega"]
        dict1["Measurements"][new_name] = d
        print(hkl.split())
        for keys in d:
            print(keys)

        print("s")

    return dict1


def auto(dict):
    """takes just unique tuples from all tuples in dictionary returend by scan_dict
    intendet for automatic merge if you doesent want to specify what scans to merge together
    args: dict - dictionary from scan_dict function
    :return dict - dict without repetitions"""
    for keys in dict:
        tuple_list = dict[keys]
        new = list()
        for i in range(len(tuple_list)):
            if tuple_list[0][0] == tuple_list[i][0]:
                new.append(tuple_list[i])
        dict[keys] = new
    return dict


def scan_dict(dict):
    """scans dictionary for duplicate hkl indexes
    :arg dict : dictionary to scan
    :return  dictionary with matching scans, if there are none, the dict is empty
    note: can be checked by "not d", true if empty
    """

    d = {}
    for i in dict["Measurements"]:
        for j in dict["Measurements"]:
            if dict["Measurements"][str(i)] != dict["Measurements"][str(j)]:
                itup = (
                    dict["Measurements"][str(i)]["h_index"],
                    dict["Measurements"][str(i)]["k_index"],
                    dict["Measurements"][str(i)]["l_index"],
                )
                jtup = (
                    dict["Measurements"][str(j)]["h_index"],
                    dict["Measurements"][str(j)]["k_index"],
                    dict["Measurements"][str(j)]["l_index"],
                )
                if itup != jtup:
                    pass
                else:

                    if str(itup) not in d:
                        d[str(itup)] = list()
                        d[str(itup)].append((i, j))
                    else:
                        d[str(itup)].append((i, j))
            else:
                continue
    return d


def compare_hkl(dict1, dict2):
    """Compares two dictionaries based on hkl indexes and return dictionary with str(h k l) as
    key and tuple with keys to same measurement in dict1 and dict2
    :arg dict1 : first dictionary
    :arg dict2 : second dictionary
    :return d : dict with matches
    example of one key: '0.0 0.0 -1.0 : ('M1', 'M9')' meaning that 001 hkl measurement is M1 in
    first dict and M9 in second"""
    d = {}
    dupl = 0
    for keys in dict1["Measurements"]:
        for key in dict2["Measurements"]:
            if (
                dict1["Measurements"][str(keys)]["h_index"]
                == dict2["Measurements"][str(key)]["h_index"]
                and dict1["Measurements"][str(keys)]["k_index"]
                == dict2["Measurements"][str(key)]["k_index"]
                and dict1["Measurements"][str(keys)]["l_index"]
                == dict2["Measurements"][str(key)]["l_index"]
            ):

                if (
                    str(
                        (
                            str(dict1["Measurements"][str(keys)]["h_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["k_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["l_index"])
                        )
                    )
                    not in d
                ):
                    d[
                        str(
                            str(dict1["Measurements"][str(keys)]["h_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["k_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["l_index"])
                        )
                    ] = (str(keys), str(key))
                else:
                    dupl = dupl + 1
                    d[
                        str(
                            str(dict1["Measurements"][str(keys)]["h_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["k_index"])
                            + " "
                            + str(dict1["Measurements"][str(keys)]["l_index"])
                            + "_dupl"
                            + str(dupl)
                        )
                    ] = (str(keys), str(key))
            else:
                continue

    return d


def create_tuples(x, y, y_err):
    """creates tuples for sorting and merginng of the data
    Counts need to be normalized to monitor before"""
    t = list()
    for i in range(len(x)):
        tup = (x[i], y[i], y_err[i])
        t.append(tup)
    return t


def normalize(dict, key, monitor):
    """Normalizes the measurement to monitor, checks if sigma exists, otherwise creates it
    :arg dict : dictionary to from which to tkae the scan
    :arg key : which scan to normalize from dict1
    :arg monitor : final monitor
    :return counts - normalized counts
    :return sigma - normalized sigma"""

    counts = np.array(dict["Measurements"][key]["Counts"])
    sigma = (
        np.sqrt(counts)
        if "sigma" not in dict["Measurements"][key]
        else dict["Measurements"][key]["sigma"]
    )
    monitor_ratio = monitor / dict["Measurements"][key]["monitor"]
    scaled_counts = counts * monitor_ratio
    scaled_sigma = np.array(sigma) * monitor_ratio

    return scaled_counts, scaled_sigma


def merge(dict1, dict2, keys, auto=True, monitor=100000):
    """merges the two tuples and sorts them, if om value is same, Counts value is average
    averaging is propagated into sigma if dict1 == dict2, key[1] is deleted after merging
    :arg dict1 : dictionary to which measurement will be merged
    :arg dict2 : dictionary from which measurement will be merged
    :arg keys : tuple with key to dict1 and dict2
    :arg auto : if true, when monitors are same, does not change it, if flase, takes monitor always
    :arg monitor : final monitor after merging
    note: dict1 and dict2 can be same dict
    :return dict1 with merged scan"""
    if auto:
        if dict1["Measurements"][keys[0]]["monitor"] == dict2["Measurements"][keys[1]]["monitor"]:
            monitor = dict1["Measurements"][keys[0]]["monitor"]

    # load om and Counts
    x1, x2 = dict1["Measurements"][keys[0]]["om"], dict2["Measurements"][keys[1]]["om"]
    cor_y1, y_err1 = normalize(dict1, keys[0], monitor=monitor)
    cor_y2, y_err2 = normalize(dict2, keys[1], monitor=monitor)
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

    if dict1 == dict2:
        del dict1["Measurements"][keys[1]]

    note = (
        f"This measurement was merged with measurement {keys[1]} from "
        f'file {dict2["meta"]["original_filename"]} \n'
    )
    if "notes" not in dict1["Measurements"][str(keys[0])]:
        dict1["Measurements"][str(keys[0])]["notes"] = note
    else:
        dict1["Measurements"][str(keys[0])]["notes"] += note

    dict1["Measurements"][keys[0]]["om"] = om
    dict1["Measurements"][keys[0]]["Counts"] = Counts
    dict1["Measurements"][keys[0]]["sigma"] = sigma
    dict1["Measurements"][keys[0]]["monitor"] = monitor
    print("merging done")
    return dict1


def substract_measurement(dict1, dict2, keys, auto=True, monitor=100000):
    """Substracts two measurement (measurement key2 from dict2 from measurent key1 in dict1), expects om to be same
    :arg dict1 : dictionary to which measurement will be merged
    :arg dict2 : dictionary from which measurement will be merged
    :arg keys : tuple with key to dict1 and dict2
    :arg auto : if true, when monitors are same, does not change it, if flase, takes monitor always
    :arg monitor : final monitor after merging
    :returns d : dict1 with substracted Counts from  dict2 and sigma that comes from the substraction"""

    if len(dict1["Measurements"][keys[0]]["om"]) != len(dict2["Measurements"][keys[1]]["om"]):
        raise ValueError("Omegas have different lengths, cannot be substracted")

    if auto:
        if dict1["Measurements"][keys[0]]["monitor"] == dict2["Measurements"][keys[1]]["monitor"]:
            monitor = dict1["Measurements"][keys[0]]["monitor"]

    cor_y1, y_err1 = normalize(dict1, keys[0], monitor=monitor)
    cor_y2, y_err2 = normalize(dict2, keys[1], monitor=monitor)

    dict1_count_err = create_uncertanities(cor_y1, y_err1)
    dict2_count_err = create_uncertanities(cor_y2, y_err2)

    res = np.subtract(dict1_count_err, dict2_count_err)

    res_nom = []
    res_err = []
    for k in range(len(res)):
        res_nom = np.append(res_nom, res[k].n)
        res_err = np.append(res_err, res[k].s)

    if len([num for num in res_nom if num < 0]) >= 0.3 * len(res_nom):
        print(
            f"Warning! percentage of negative numbers in measurement subsracted {keys[0]} is "
            f"{len([num for num in res_nom if num < 0]) / len(res_nom)}"
        )

    dict1["Measurements"][str(keys[0])]["Counts"] = res_nom
    dict1["Measurements"][str(keys[0])]["sigma"] = res_err
    dict1["Measurements"][str(keys[0])]["monitor"] = monitor
    note = (
        f'Measurement {keys[1]} from file {dict2["meta"]["original_filename"]} '
        f"was substracted from this measurement \n"
    )
    if "notes" not in dict1["Measurements"][str(keys[0])]:
        dict1["Measurements"][str(keys[0])]["notes"] = note
    else:
        dict1["Measurements"][str(keys[0])]["notes"] += note
    return dict1


def compare_dict(dict1, dict2):
    """takes two ccl dictionaries and compare different values for each key
    :arg dict1 : dictionary 1 (ccl)
    :arg dict2 : dictionary 2 (ccl)
    :returns warning : dictionary with keys from primary files (if they differ) with
    information of how many measurement differ and which ones differ
    :returns report_string string comparing all different values respecively of measurements"""

    if dict1["meta"]["data_type"] != dict2["meta"]["data_type"]:
        print("select two dicts")
        return
    S = []
    conflicts = {}
    warnings = {}

    comp = compare_hkl(dict1, dict2)
    d1 = scan_dict(dict1)
    d2 = scan_dict(dict2)
    if not d1:
        S.append("There are no duplicates in %s (dict1) \n" % dict1["meta"]["original_filename"])
    else:
        S.append(
            "There are %d duplicates in %s (dict1) \n"
            % (len(d1), dict1["meta"]["original_filename"])
        )
        warnings["Duplicates in dict1"] = list()
        for keys in d1:
            S.append("Measurements %s with hkl %s \n" % (d1[keys], keys))
            warnings["Duplicates in dict1"].append(d1[keys])
    if not d2:
        S.append("There are no duplicates in %s (dict2) \n" % dict2["meta"]["original_filename"])
    else:
        S.append(
            "There are %d duplicates in %s (dict2) \n"
            % (len(d2), dict2["meta"]["original_filename"])
        )
        warnings["Duplicates in dict2"] = list()
        for keys in d2:
            S.append("Measurements %s with hkl %s \n" % (d2[keys], keys))
            warnings["Duplicates in dict2"].append(d2[keys])

    # compare meta
    S.append("Different values in meta: \n")
    different_meta = {
        k: dict1["meta"][k]
        for k in dict1["meta"]
        if k in dict2["meta"] and dict1["meta"][k] != dict2["meta"][k]
    }
    exlude_meta_set = ["original_filename", "date", "title"]
    for keys in different_meta:
        if keys in exlude_meta_set:
            continue
        else:
            if keys not in conflicts:
                conflicts[keys] = 1
            else:
                conflicts[keys] = conflicts[keys] + 1

            S.append("   Different values in %s \n" % str(keys))
            S.append("           dict1: %s \n" % str(dict1["meta"][str(keys)]))
            S.append("           dict2: %s \n" % str(dict2["meta"][str(keys)]))

    # compare Measurements
    S.append(
        "Number of measurements in %s = %s \n"
        % (dict1["meta"]["original_filename"], len(dict1["Measurements"]))
    )
    S.append(
        "Number of measurements in %s = %s \n"
        % (dict2["meta"]["original_filename"], len(dict2["Measurements"]))
    )
    S.append("Different values in Measurements:\n")
    select_set = ["om", "Counts", "sigma"]
    exlude_set = ["time", "Counts", "date", "notes"]
    for keys1 in comp:
        for key2 in dict1["Measurements"][str(comp[str(keys1)][0])]:
            if key2 in exlude_set:
                continue
            if key2 not in select_set:
                try:
                    if (
                        dict1["Measurements"][comp[str(keys1)][0]][str(key2)]
                        != dict2["Measurements"][str(comp[str(keys1)][1])][str(key2)]
                    ):
                        S.append(
                            "Measurement value "
                            "%s"
                            ", with hkl %s differs in meausrements %s and %s \n"
                            % (key2, keys1, comp[str(keys1)][0], comp[str(keys1)][1])
                        )
                        S.append(
                            "     dict1:   %s \n"
                            % str(dict1["Measurements"][comp[str(keys1)][0]][str(key2)])
                        )
                        S.append(
                            "     dict2:   %s \n"
                            % str(dict2["Measurements"][comp[str(keys1)][1]][str(key2)])
                        )
                        if key2 not in conflicts:
                            conflicts[key2] = {}
                            conflicts[key2]["amount"] = 1
                            conflicts[key2]["measurements"] = str(comp[str(keys1)])
                        else:

                            conflicts[key2]["amount"] = conflicts[key2]["amount"] + 1
                            conflicts[key2]["measurements"] = (
                                conflicts[key2]["measurements"] + " " + (str(comp[str(keys1)]))
                            )
                except KeyError as e:
                    print("Missing keys, some files were probably merged or substracted")
                    print(e.args)

            else:
                try:
                    comparison = list(
                        dict1["Measurements"][comp[str(keys1)][0]][str(key2)]
                    ) == list(dict2["Measurements"][comp[str(keys1)][1]][str(key2)])
                    if len(list(dict1["Measurements"][comp[str(keys1)][0]][str(key2)])) != len(
                        list(dict2["Measurements"][comp[str(keys1)][1]][str(key2)])
                    ):
                        if str("different length of %s" % key2) not in warnings:
                            warnings[str("different length of %s" % key2)] = list()
                            warnings[str("different length of %s" % key2)].append(
                                (str(comp[keys1][0]), str(comp[keys1][1]))
                            )
                        else:
                            warnings[str("different length of %s" % key2)].append(
                                (str(comp[keys1][0]), str(comp[keys1][1]))
                            )
                    if not comparison:
                        S.append(
                            "Measurement value "
                            "%s"
                            " differs in measurement %s and %s \n"
                            % (key2, comp[str(keys1)][0], comp[str(keys1)][1])
                        )
                        S.append(
                            "       dict1:   %s \n"
                            % str(list(dict1["Measurements"][comp[str(keys1)][0]][str(key2)]))
                        )
                        S.append(
                            "       dict2:   %s \n"
                            % str(list(dict2["Measurements"][comp[str(keys1)][1]][str(key2)]))
                        )
                        if key2 not in conflicts:
                            conflicts[key2] = {}
                            conflicts[key2]["amount"] = 1
                            conflicts[key2]["measurements"] = str(comp[str(keys1)])
                        else:
                            conflicts[key2]["amount"] = conflicts[key2]["amount"] + 1
                            conflicts[key2]["measurements"] = (
                                conflicts[key2]["measurements"] + " " + (str(comp[str(keys1)]))
                            )
                except KeyError as e:
                    print("Missing keys, some files were probably merged or substracted")
                    print(e.args)

    for keys in conflicts:
        try:
            conflicts[str(keys)]["measurements"] = conflicts[str(keys)]["measurements"].split(" ")
        except:
            continue
    report_string = "".join(S)
    return warnings, conflicts, report_string


def guess_next(dict1, dict2, comp):
    """iterates thorough the scans and tries to decide if the scans should be
    substracted or merged"""
    threshold = 0.05
    for keys in comp:
        if (
            abs(
                (
                    dict1["Measurements"][str(comp[keys][0])]["temperature"]
                    - dict2["Measurements"][str(comp[keys][1])]["temperature"]
                )
                / dict2["Measurements"][str(comp[keys][1])]["temperature"]
            )
            < threshold
            and abs(
                (
                    dict1["Measurements"][str(comp[keys][0])]["mag_field"]
                    - dict2["Measurements"][str(comp[keys][1])]["mag_field"]
                )
                / dict2["Measurements"][str(comp[keys][1])]["mag_field"]
            )
            < threshold
        ):
            comp[keys] = comp[keys] + tuple("m")
        else:
            comp[keys] = comp[keys] + tuple("s")

    return comp


def process_dict(dict1, dict2, comp):
    """substracts or merges scans, guess_next function must run first """
    for keys in comp:
        if comp[keys][2] == "s":
            substract_measurement(dict1, dict2, comp[keys])
        elif comp[keys][2] == "m":
            merge(dict1, dict2, comp[keys])

    return dict1
