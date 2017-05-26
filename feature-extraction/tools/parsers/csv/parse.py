import csv

def parse(targets_path):
    """Parse a csv containing an ordered list of targets on the first row.

    Args:
        targets_path: The path to the csv file

    Returns:
        row: A list of targets from the first row of the csv.
    """
    with open(targets_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '\"')

        for row in reader:
            return row
