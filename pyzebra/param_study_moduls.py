from load_1D import load_1D
from ccl_dict_operation import add_dict
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # dont delete, otherwise waterfall wont work
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import scipy.io as sio


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
        dict1["scan"][i + 1]["params"] = {}
        if data_type == "txt":
            for x in range(len(col_names) - 1):
                dict1["scan"][i + 1]["params"][col_names[x + 1]] = file_list[i][x + 1]

    return dict1


def create_dataframe(dict1):
    """Creates pandas dataframe from the dictionary
    :arg ccl like dictionary
    :return pandas dataframe"""
    # create dictionary to which we pull only wanted items before transforming it to pd.dataframe
    pull_dict = {}
    pull_dict["filenames"] = list()
    for key in dict1["scan"][1]["params"]:
        pull_dict[key] = list()
    pull_dict["temperature"] = list()
    pull_dict["mag_field"] = list()
    pull_dict["fit_area"] = list()
    pull_dict["int_area"] = list()
    pull_dict["om"] = list()
    pull_dict["Counts"] = list()

    # populate the dict
    for keys in dict1["scan"]:
        if "file_of_origin" in dict1["scan"][keys]:
            pull_dict["filenames"].append(dict1["scan"][keys]["file_of_origin"].split("/")[-1])
        else:
            pull_dict["filenames"].append(dict1["meta"]["original_filename"].split("/")[-1])
        for key in dict1["scan"][keys]["params"]:
            pull_dict[str(key)].append(float(dict1["scan"][keys]["params"][key]))
        pull_dict["temperature"].append(dict1["scan"][keys]["temperature"])
        pull_dict["mag_field"].append(dict1["scan"][keys]["mag_field"])
        pull_dict["fit_area"].append(dict1["scan"][keys]["fit"]["fit_area"])
        pull_dict["int_area"].append(dict1["scan"][keys]["fit"]["int_area"])
        pull_dict["om"].append(dict1["scan"][keys]["om"])
        pull_dict["Counts"].append(dict1["scan"][keys]["Counts"])

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
    """ saves dictionary as pickle file in binary format
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
