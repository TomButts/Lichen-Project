import pandas as pd
import csv

def write_config_info(model_config, grid_options, config, directory_path):
    output_path = directory_path + '/configuration.csv'

    evaluation_csv = open(output_path, 'wb')
    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)

    if 'selectors' in config:
        selectors = config['selectors']

        headers = ['Variance Threshold', 'Percentile Mode', 'Percentage']
        writer.writerow(headers)

        selectors = [
            selectors['variance_threshold'],
            selectors['feature_percentile']['mode'],
            selectors['feature_percentile']['percentage']]

        writer.writerow(selectors)

    # line break
    writer.writerow('')

    headers = ['Scaling Type', 'Transform Factor']
    writer.writerow(headers)

    meta = [config['scaling'], config['transform_factor']]
    writer.writerow(meta)

    model = pd.DataFrame(model_config)
    model.to_csv(directory_path + '/model_parameters.csv')

    for name, grid in grid_options.iteritems():
        grid_data = pd.DataFrame(grid)
        grid_data.to_csv(directory_path + '/' + name + '_grid_search.csv')
