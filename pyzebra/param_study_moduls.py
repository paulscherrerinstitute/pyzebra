import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import uncertainties as u
from mpl_toolkits.mplot3d import Axes3D  # dont delete, otherwise waterfall wont work
import collections

from .ccl_io import load_1D


def create_tuples(x, y, y_err):
    """creates tuples for sorting and merginng of the data
    Counts need to be normalized to monitor before"""
    t = list()
    for i in range(len(x)):
        tup = (x[i], y[i], y_err[i])
        t.append(tup)
    return t


def load_dats(filepath):
    """reads the txt file, get headers and data
    :arg filepath to txt file or list of filepaths to the files
    :return ccl like dictionary"""
    if isinstance(filepath, str):
        data_type = "txt"
        file_list = list()
        with open(filepath, "r") as infile:
            col_names = next(infile).split(",")
            col_names = [col_names[i].rstrip() for i in range(len(col_names))]
            for line in infile:
                if "END" in line:
                    break
                file_list.append(tuple(line.split(",")))
    elif isinstance(filepath, list):
        data_type = "list"
        file_list = filepath
    dict1 = {}
    for i in range(len(file_list)):
        if not dict1:
            if data_type == "txt":
                dict1 = load_1D(file_list[0][0])
            else:
                dict1 = load_1D(file_list[0])
        else:
            if data_type == "txt":
                dict1 = add_dict(dict1, load_1D(file_list[i][0]))
            else:

                dict1 = add_dict(dict1, load_1D(file_list[i]))
        dict1["scan"].append({})
        if data_type == "txt":
            for x in range(len(col_names) - 1):
                dict1["scan"][i + 1]["params"][col_names[x + 1]] = float(file_list[i][x + 1])
    return dict1


def create_dataframe(dict1, variables):
    """Creates pandas dataframe from the dictionary
    :arg ccl like dictionary
    :return pandas dataframe"""
    # create dictionary to which we pull only wanted items before transforming it to pd.dataframe
    pull_dict = {}
    pull_dict["filenames"] = list()
    for keys in variables:
        for item in variables[keys]:
            pull_dict[item] = list()
    pull_dict["fit_area"] = list()
    pull_dict["int_area"] = list()
    pull_dict["Counts"] = list()

    for keys in pull_dict:
        print(keys)

    # populate the dict
    for keys in range(len(dict1["scan"])):
        if "file_of_origin" in dict1["scan"][keys]:
            pull_dict["filenames"].append(dict1["scan"][keys]["file_of_origin"].split("/")[-1])
        else:
            pull_dict["filenames"].append(dict1["meta"]["original_filename"].split("/")[-1])

        pull_dict["fit_area"].append(dict1["scan"][keys]["fit"]["fit_area"])
        pull_dict["int_area"].append(dict1["scan"][keys]["fit"]["int_area"])
        pull_dict["Counts"].append(dict1["scan"][keys]["Counts"])
        for key in variables:
            for i in variables[key]:
                pull_dict[i].append(_finditem(dict1["scan"][keys], i))

    return pd.DataFrame(data=pull_dict)


def sort_dataframe(dataframe, sorting_parameter):
    """sorts the data frame and resets index"""
    data = dataframe.sort_values(by=sorting_parameter)
    data = data.reset_index(drop=True)
    return data


def make_graph(data, sorting_parameter, style):
    """Makes the graph from the data based on style and sorting parameter
    :arg data : pandas dataframe with data after sorting
    :arg sorting_parameter to pull the correct variable and name
    :arg style of the graph - waterfall, scatter, heatmap
    :return matplotlib figure"""
    if style == "waterfall":
        mpl.rcParams["legend.fontsize"] = 10
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        for i in range(len(data)):
            x = data["om"][i]
            z = data["Counts"][i]
            yy = [data[sorting_parameter][i]] * len(x)
            ax.plot(x, yy, z, label=str("%s = %f" % (sorting_parameter, yy[i])))

        ax.legend()
        ax.set_xlabel("Omega")
        ax.set_ylabel(sorting_parameter)
        ax.set_zlabel("counts")

    elif style == "scatter":
        fig = plt.figure()
        plt.errorbar(
            data[sorting_parameter],
            [data["fit_area"][i].n for i in range(len(data["fit_area"]))],
            [data["fit_area"][i].s for i in range(len(data["fit_area"]))],
            capsize=5,
            ecolor="green",
        )
        plt.xlabel(str(sorting_parameter))
        plt.ylabel("Intesity")

    elif style == "heat":
        new_om = list()
        for i in range(len(data)):
            new_om = np.append(new_om, np.around(data["om"][i], 2), axis=0)
        unique_om = np.unique(new_om)
        color_matrix = np.zeros(shape=(len(data), len(unique_om)))
        for i in range(len(data)):
            for j in range(len(data["om"][i])):
                if np.around(data["om"][i][j], 2) in np.unique(new_om):
                    color_matrix[i, j] = data["Counts"][i][j]
                else:
                    continue

        fig = plt.figure()
        plt.pcolormesh(unique_om, data[sorting_parameter], color_matrix, shading="gouraud")
        plt.xlabel("omega")
        plt.ylabel(sorting_parameter)
        plt.colorbar()
        plt.clim(color_matrix.mean(), color_matrix.max())

    return fig


def save_dict(obj, name):
    """saves dictionary as pickle file in binary format
    :arg obj - object to save
    :arg name - name of the file
    NOTE: path should be added later"""
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    """load dictionary from picle file
    :arg name - name of the file to load
    NOTE: expect the file in the same folder, path should be added later
    :return dictionary"""
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


# pickle, mat, h5, txt, csv, json
def save_table(data, filetype, name, path=None):
    print("Saving: ", filetype)
    path = "" if path is None else path
    if filetype == "pickle":
        # to work with uncertanities, see uncertanity module
        with open(path + name + ".pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    if filetype == "mat":
        # matlab doesent allow some special character to be in var names, also cant start with
        # numbers, in need, add some to the romove_character list
        data["fit_area_nom"] = [data["fit_area"][i].n for i in range(len(data["fit_area"]))]
        data["fit_area_err"] = [data["fit_area"][i].s for i in range(len(data["fit_area"]))]
        data["int_area_nom"] = [data["int_area"][i].n for i in range(len(data["int_area"]))]
        data["int_area_err"] = [data["int_area"][i].s for i in range(len(data["int_area"]))]
        data = data.drop(columns=["fit_area", "int_area"])
        remove_characters = [" ", "[", "]", "{", "}", "(", ")"]
        for character in remove_characters:
            data.columns = [
                data.columns[i].replace(character, "") for i in range(len(data.columns))
            ]
        sio.savemat((path + name + ".mat"), {name: col.values for name, col in data.items()})
    if filetype == "csv" or "txt":
        data["fit_area_nom"] = [data["fit_area"][i].n for i in range(len(data["fit_area"]))]
        data["fit_area_err"] = [data["fit_area"][i].s for i in range(len(data["fit_area"]))]
        data["int_area_nom"] = [data["int_area"][i].n for i in range(len(data["int_area"]))]
        data["int_area_err"] = [data["int_area"][i].s for i in range(len(data["int_area"]))]
        data = data.drop(columns=["fit_area", "int_area", "om", "Counts"])
        if filetype == "csv":
            data.to_csv(path + name + ".csv")
        if filetype == "txt":
            with open((path + name + ".txt"), "w") as outfile:
                data.to_string(outfile)
    if filetype == "h5":
        hdf = pd.HDFStore((path + name + ".h5"))
        hdf.put("data", data)
        hdf.close()
    if filetype == "json":
        data.to_json((path + name + ".json"))


def normalize(scan, monitor):
    """Normalizes the measurement to monitor, checks if sigma exists, otherwise creates it
    :arg dict : dictionary to from which to tkae the scan
    :arg key : which scan to normalize from dict1
    :arg monitor : final monitor
    :return counts - normalized counts
    :return sigma - normalized sigma"""

    counts = np.array(scan["Counts"])
    sigma = np.sqrt(counts) if "sigma" not in scan else scan["sigma"]
    monitor_ratio = monitor / scan["monitor"]
    scaled_counts = counts * monitor_ratio
    scaled_sigma = np.array(sigma) * monitor_ratio

    return scaled_counts, scaled_sigma


def merge(scan1, scan2, keep=True, monitor=100000):
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

    if keep:
        if scan1["monitor"] == scan2["monitor"]:
            monitor = scan1["monitor"]

    # load om and Counts
    x1, x2 = scan1["om"], scan2["om"]
    cor_y1, y_err1 = normalize(scan1, monitor=monitor)
    cor_y2, y_err2 = normalize(scan2, monitor=monitor)
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
    scan1["monitor"] = monitor
    print("merging done")


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
    new_meta_name = "meta" + str(dict2["meta"]["original_filename"])
    if new_meta_name not in dict1:
        for keys, name in zip(dict2["scan"], new_filenames):
            dict2["scan"][keys]["file_of_origin"] = str(dict2["meta"]["original_filename"])
            dict1["scan"][name] = dict2["scan"][keys]

        dict1[new_meta_name] = dict2["meta"]
    else:
        raise KeyError(
            str(
                "The file %s has alredy been added to %s"
                % (dict2["meta"]["original_filename"], dict1["meta"]["original_filename"])
            )
        )
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


def scan_dict(dict, precision=0.5):
    """scans dictionary for duplicate angles indexes
    :arg dict : dictionary to scan
    :arg precision : in deg, sometimes angles are zero so its easier this way, instead of
    checking zero division
    :return  dictionary with matching scans, if there are none, the dict is empty
    note: can be checked by "not d", true if empty
    """

    if dict["meta"]["zebra_mode"] == "bi":
        angles = ["twotheta", "omega", "chi", "phi"]
    elif dict["meta"]["zebra_mode"] == "nb":
        angles = ["gamma", "omega", "nu"]
    else:
        print("Unknown zebra mode")
        return

    d = {}
    for i in range(len(dict["scan"])):
        for j in range(len(dict["scan"])):
            if dict["scan"][i] != dict["scan"][j]:
                itup = list()
                for k in angles:
                    itup.append(abs(abs(dict["scan"][i][k]) - abs(dict["scan"][j][k])))

                if all(i <= precision for i in itup):
                    print(itup)
                    print([dict["scan"][i][k] for k in angles])
                    print([dict["scan"][j][k] for k in angles])
                    if str([np.around(dict["scan"][i][k], 0) for k in angles]) not in d:
                        d[str([np.around(dict["scan"][i][k], 0) for k in angles])] = list()
                        d[str([np.around(dict["scan"][i][k], 0) for k in angles])].append((i, j))
                    else:
                        d[str([np.around(dict["scan"][i][k], 0) for k in angles])].append((i, j))

                else:
                    pass

            else:
                continue

    return d


def _finditem(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = _finditem(v, key)
            if item is not None:
                return item


def most_common(lst):
    return max(set(lst), key=lst.count)


def variables(dictionary):
    """Funcrion to guess what variables will be used in the param study
    i call pripary variable the one the array like variable, usually omega
    and secondary the slicing variable, different for each scan,for example temperature"""
    # find all variables that are in all scans
    stdev_precision = 0.05
    all_vars = list()
    for keys in range(len(dictionary["scan"])):
        all_vars.append([key for key in dictionary["scan"][keys] if key != "params"])
        if dictionary["scan"][keys]["params"]:
            all_vars.append(key for key in dictionary["scan"][keys]["params"])

    all_vars = [i for sublist in all_vars for i in sublist]
    # get the ones that are in all scans
    b = collections.Counter(all_vars)
    inall = [key for key in b if b[key] == len(dictionary["scan"])]
    # delete those that are obviously wrong
    wrong = [
        "NP",
        "Counts",
        "Monitor1",
        "Monitor2",
        "Monitor3",
        "h",
        "k",
        "l",
        "n_points",
        "monitor",
        "Time",
        "omega",
        "twotheta",
        "chi",
        "phi",
        "nu",
    ]
    inall_red = [i for i in inall if i not in wrong]

    # check for primary variable, needs to be list, we dont suspect the
    # primary variable be as a parameter (be in scan[params])
    primary_candidates = list()
    for key in range(len(dictionary["scan"])):
        for i in inall_red:
            if isinstance(_finditem(dictionary["scan"][key], i), list):
                if np.std(_finditem(dictionary["scan"][key], i)) > stdev_precision:
                    primary_candidates.append(i)
    # check which of the primary are in every scan
    primary_candidates = collections.Counter(primary_candidates)
    second_round_primary_candidates = [
        key for key in primary_candidates if primary_candidates[key] == len(dictionary["scan"])
    ]

    if len(second_round_primary_candidates) == 1:
        print("We've got a primary winner!", second_round_primary_candidates)
    else:
        print("Still not sure with primary:(", second_round_primary_candidates)

    # check for secondary variable, we suspect a float\int or not changing array
    # we dont need to check for primary ones
    secondary_candidates = [i for i in inall_red if i not in second_round_primary_candidates]
    # print("secondary candidates", secondary_candidates)
    # select arrays and floats and ints
    second_round_secondary_candidates = list()
    for key in range(len(dictionary["scan"])):
        for i in secondary_candidates:
            if isinstance(_finditem(dictionary["scan"][key], i), float):
                second_round_secondary_candidates.append(i)
            elif isinstance(_finditem(dictionary["scan"][key], i), int):
                second_round_secondary_candidates.append(i)
            elif isinstance(_finditem(dictionary["scan"][key], i), list):
                if np.std(_finditem(dictionary["scan"][key], i)) < stdev_precision:
                    second_round_secondary_candidates.append(i)

    second_round_secondary_candidates = collections.Counter(second_round_secondary_candidates)
    second_round_secondary_candidates = [
        key
        for key in second_round_secondary_candidates
        if second_round_secondary_candidates[key] == len(dictionary["scan"])
    ]
    # print("secondary candidates after second round", second_round_secondary_candidates)
    # now we check if they vary between the scans
    third_round_sec_candidates = list()
    for i in second_round_secondary_candidates:
        check_array = list()
        for keys in range(len(dictionary["scan"])):
            check_array.append(np.average(_finditem(dictionary["scan"][keys], i)))
        # print(i, check_array, np.std(check_array))
        if np.std(check_array) > stdev_precision:
            third_round_sec_candidates.append(i)
    if len(third_round_sec_candidates) == 1:
        print("We've got a secondary winner!", third_round_sec_candidates)
    else:
        print("Still not sure with secondary :(", third_round_sec_candidates)

    return {"primary": second_round_primary_candidates, "secondary": third_round_sec_candidates}
