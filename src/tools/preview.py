from prettytable import PrettyTable

dataset_name = 'DatasetA'

with open("../../data/{}_train/label_list.txt".format(dataset_name)) as f:
    train_labels = [line.strip('\n') for line in f.readlines()]

all_train_labels = {}
all_train_label_counts = {}

for labels in train_labels:
    label, meaning = labels.split('\t')
    all_train_labels[label] = meaning
    all_train_label_counts[label] = 0

with open("../../data/{}_train/train.txt".format(dataset_name)) as f:
    train_data = [line.strip('\n') for line in f.readlines()]

for each_data in train_data:
    img_file, label = each_data.split('\t')
    all_train_label_counts[label] += 1



with open("../../data/{}_test/label_list.txt".format(dataset_name)) as f:
    test_labels = [line.strip('\n') for line in f.readlines()]


all_test_labels = {}
table = PrettyTable(['index', 'label', 'meaing', 'counts'])

for labels in test_labels:
    label, meaning = labels.split('\t')
    all_test_labels[label] = meaning

labels = list(set(all_train_labels.keys()).union(set(all_test_labels.keys())))
labels = [int(label.strip('ZJL')) for label in labels]
labels.sort()

for index, label in enumerate(labels, 1):
    label = 'ZJL'+str(label)
    if label in all_train_label_counts.keys():
        table.add_row([index, label, all_test_labels[label], all_train_label_counts[label]])
    else:
        table.add_row([index, label, all_test_labels[label], 0])

print(table)





