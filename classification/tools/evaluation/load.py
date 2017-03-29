import pickle
import os

def load(directory_path):
    """Load all pickled files in folder

    Args:
        directory_path: the path to the pickled object files
    Returns:
        items: dictionary containing the objects with file name - extension as key
    """

    items = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".sav"):
            # name the object after the file without extension
            item = os.path.splitext(filename)[0]

            items[item] = pickle.load(open(directory_path + '/' + filename, 'rb'))

    return items
