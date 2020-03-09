import h5py


def read_cami(filepath):
    """Read and parse content of a cami file.

    Args:
        filepath (str): File path of a cami file.

    Returns:
        dict: A dictionary with section names and their content.
    """
    cami_content = dict()
    with open(filepath, "r") as cami_file:
        line = cami_file.readline()
        while line:
            if line.startswith("#begin"):
                # read section
                section = line[7:-1]  # len("#begin ") = 7
                cami_content[section] = []
                line = cami_file.readline()
                while not line.startswith("#end"):
                    cami_content[section].append(line[:-1])
                    line = cami_file.readline()

            # read next line after section's end
            line = cami_file.readline()

    return cami_content


def read_detector_data(filepath):
    """Read detector data from an h5 file.

    Args:
        filepath (str): File path of an h5 file.

    Returns:
        ndarray: A 3D array of data.
    """
    with h5py.File(filepath, "r") as h5f:
        detector_data = h5f["/entry1/area_detector2/data"][:]

        # reshape data to a correct shape (2006 issue)
        n, cols, rows = detector_data.shape
        detector_data = detector_data.reshape(n, rows, cols)

    return detector_data


def open_cami(filepath):
    """Open cami scan (?)

    Args:
        filepath (str): File path of a cami file.

    Returns:
        dict: A dictionary with h5 names and their detector data.
    """
    data = dict()
    cami_content = read_cami(filepath)
    for file in cami_content["filelist"]:
        data[file] = read_detector_data(file)

    return data
