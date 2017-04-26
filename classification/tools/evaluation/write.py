import csv
import time
import numpy as np

def write_csv(scores):
    # initialise csv output path
    now = time.strftime("%d-%b-%H%M%S")
    output_path = '/Users/tom/Masters-Project/Lichen-Project/classification/output/evaluations/evaluation-' + now + '.csv'

    evaluation_csv = open(output_path, 'wb')
    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)

    # header = ['model_index', 'f1_mean', 'f1_std', 'f1_cv',
    #           'precision_mean', 'precision_std', 'precision_cv',
    #           'recall_mean', 'recall_std', 'recall_cv',
    #           'accuracy_mean', 'accuracy_std', 'accuracy_cv',
    #           'neg_ll_mean', 'neg_ll_std', 'neg_ll_cv']
    #
    # writer.writerow(header)
    #
    # index = 1
    #
    # for dictionary in scores:
    #     row = [index,
    #            dictionary['cross_val_f1'].mean(),
    #            dictionary['cross_val_f1'].std(),
    #            len(dictionary['cross_val_f1']),
    #            dictionary['cross_val_precision'].mean(),
    #            dictionary['cross_val_precision'].std(),
    #            len(dictionary['cross_val_precision']),
    #            dictionary['cross_val_recall'].mean(),
    #            dictionary['cross_val_recall'].std(),
    #            len(dictionary['cross_val_recall']),
    #            dictionary['cross_val_accuracy'].mean(),
    #            dictionary['cross_val_accuracy'].std(),
    #            len(dictionary['cross_val_accuracy'])]
    # 
    #     if 'neg_log_loss' in dictionary:
    #         row.append(dictionary['neg_log_loss'].mean())
    #         row.append(dictionary['neg_log_loss'].std())
    #         row.append(len(dictionary['neg_log_loss']))
    #
    #     writer.writerow(row)
    #
    #     index += 1
