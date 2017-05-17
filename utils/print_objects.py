import pickle
import os
import getopt
import sys

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

            items[item] = pickle.load(
                open(directory_path + '/' + filename, 'rb'))

    return items

def print_objects(directory_path):
    items = load(directory_path)

    for key, item in items.iteritems():
        print(key)
        print(item)

if __name__ == "__main__":
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'p:')
    except getopt.GetoptError as err:
         print(err)
         exit()

    for opt, arg in options:
        if opt in ('-p'):
            print_objects(arg)
