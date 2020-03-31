import h5py

def read_h5meta(filepath):
    """Read and parse content of a h5meta file.

    Args:
        filepath (str): File path of a h5meta file.

    Returns:
        dict: A dictionary with section names and their content.
    """
    h5meta_content = dict()
    with open(filepath, "r") as h5meta_file:
        line = h5meta_file.readline()
        while line:
            if line.startswith("#begin"):
                # read section
                section = line[7:-1]  # len("#begin ") = 7
                h5meta_content[section] = []
                line = h5meta_file.readline()
                while not line.startswith("#end"):
                    h5meta_content[section].append(line[:-1])
                    line = h5meta_file.readline()

            # read next line after section's end
            line = h5meta_file.readline()

    return h5meta_content


def read_detector_data(filepath):
    """Read detector data and angles from an h5 file.

    Args:
        filepath (str): File path of an h5 file.

    Returns:
        ndarray: A 3D array of data, rot_angle, pol_angle, tilt_angle.
    """
    with h5py.File(filepath, "r") as h5f:
        detector_data = h5f["/entry1/area_detector2/data"][:]

        # reshape data to a correct shape (2006 issue)
        n, cols, rows = detector_data.shape
        detector_data = detector_data.reshape(n, rows, cols)

        rot_angle = h5f["/entry1/area_detector2/rotation_angle"][:] # om, sometimes ph
        pol_angle = h5f["/entry1/ZEBRA/area_detector2/polar_angle"][:] # gammad
        tlt_angle = h5f["/entry1/ZEBRA/area_detector2/tilt_angle"][:]  # nud
        ddist     = h5f["/entry1/ZEBRA/area_detector2/distance"][:]    
        wave      = h5f["/entry1/ZEBRA/monochromator/wavelength"][:] 
        chi_angle = h5f["/entry1/sample/chi"][:] # ch
        phi_angle = h5f["/entry1/sample/phi"][:] # ph
        UB        = h5f["/entry1/sample/UB"][:] 

    return detector_data,pol_angle,rot_angle,tlt_angle,chi_angle,phi_angle,ddist,wave,UB

def open_h5meta(filepath):
    """Open h5meta file like *.cami

    Args:
        filepath (str): File path of a h5meta file.

    Returns:
        dict: A dictionary with h5 names and their detector data and angles.
    """
    data = dict()
    h5meta_content = read_h5meta(filepath)
    for file in h5meta_content["filelist"]:
        data[file] = read_detector_data(file)

    return data
