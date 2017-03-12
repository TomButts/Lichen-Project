import csv

def read_features_csv(csv_path):
    targets = []
    data = []

    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '\"')

        for row in reader:
            # create target list
            targets.append(row[0])

            # create data list
            data.append(row[1:])

    return targets, data
