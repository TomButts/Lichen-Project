import csv

def write_dataset_info(info, directory_path):
    output_path = directory_path + '/info.csv'

    evaluation_csv = open(output_path, 'wb')

    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)

    # class sample number break down
    headers = info['class_names'][:]

    # add first column header
    headers.insert(0, 'Dataset')

    writer.writerow(headers)

    train_row = ['Train']
    test_row = ['Test']
    validation_row = ['Validation']

    for name in info['class_names']:
        train_row.append(info['training']['train'][name])
        test_row.append(info['training']['test'][name])
        validation_row.append(info['validation']['class_count'][name])

    writer.writerow(train_row)
    writer.writerow(test_row)
    writer.writerow(validation_row)

    # line break
    writer.writerow('')

    meta_row_headers = ['Training Total', 'Validation Total', 'Features', 'Features After Selection']
    meta_row = [
        info['training']['total'],
        info['validation']['total'],
        info['training']['features'],
        info['training']['features_after_selection']
        ]

    writer.writerow(meta_row_headers)
    writer.writerow(meta_row)
