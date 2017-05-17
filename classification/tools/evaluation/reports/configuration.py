"""
This file formats and writes the configuration csv
during model evaluation.

"""
import pandas as pd
import csv

def write_config_info(model_config, grid_options, config, directory_path):
    output_path = directory_path + '/configuration.csv'

    evaluation_csv = open(output_path, 'wb')
    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)

    headers = None
    selectors = None

    if 'selectors' in config:
        selectors = config['selectors']

        if 'variance_threshold' in config:
            headers = ['Variance Threshold']
            # , 'Percentile Mode', 'Percentage'
            selectors = [selectors['variance_threshold']]

        if 'feature_percentile' in config and header != None:
            headers.append('Percentile Mode')
            headers.append('Percentage')
            selectors.append(selectors['feature_percentile']['mode'])
            selectors.append(selectors['feature_percentile']['percentage'])
        elif 'feature_percentile' in config:
            header = ['Percentile Mode', 'Percentage']
            selectors = [
                selectors['feature_percentile']['mode'],
                selectors['feature_percentile']['percentage']]

    if headers != None:
        writer.writerow(headers)
        writer.writerow(selectors)

    # line break
    writer.writerow('')

    headers = ['Scaling Type', 'Transform Factor']
    writer.writerow(headers)

    meta = [config['scaling'], config['transform_factor']]
    writer.writerow(meta)

    # model = pd.DataFrame(model_config)
    # model.to_csv(directory_path + '/model_config.csv')

    for name, grid in grid_options.iteritems():
        grid_data = pd.DataFrame(grid)
        grid_data.to_csv(directory_path + '/' + name + '_grid_search.csv')
