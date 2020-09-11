import pickle


def save_dict(obj, name):
    """ saves dictionary as pickle file in binary format
    :arg obj - object to save
    :arg name - name of the file
    NOTE: path should be added later"""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    """load dictionary from picle file
    :arg name - name of the file to load
    NOTE: expect the file in the same folder, path should be added later
    :return dictionary"""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
