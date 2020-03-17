import csv
import torch
from torch.utils.data import dataset
import os
import pandas as pd

from mscn.util import *
from mscn.datasets import LoadForest

def load_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    with open(file_name + ".csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps", len(samples))
    #  for i in range(len(samples)):
    #      assert len(tables[i]) == len(samples[i])
    #      print(i, len(tables[i]), len(samples[i]))
    #  exit(1)

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label

# NOTICE: hard code for forest
def fetch_and_store_sample(table, num_samples, seed):
    np.random.seed(seed)
    sample_df = table.data.sample(n=num_samples)
    sample_df.to_csv(os.path.join('sample', 'forest_{}_{}.csv'.format(num_samples, seed)), index=False)
    return sample_df.values

def load_sample(num_samples, seed):
    sample_df = pd.read_csv(os.path.join('sample', 'forest_{}_{}.csv'.format(num_samples, seed)))
    return sample_df.values

def load_dicts(table):
    # Get column name dict
    column_names = set(range(table.data.shape[1]))
    column2vec, _ = get_set_encoding(column_names)

    # Get operator name dict
    # NOTICE: [] has already converted to two operators: >= and <= later before
    operators = set(['=', '>=', '<='])
    op2vec, _ = get_set_encoding(operators)

    min_max_dict = table.MinMaxDict()

    return column2vec, op2vec, min_max_dict

def load_and_encode_train_data(queries, labels, num_samples, seed):
    # NOTICE: only for single table, no join
    num_queries = len(queries)

    table = LoadForest()
    sample = fetch_and_store_sample(table, num_samples, seed)

    column2vec, op2vec, min_max_dict = load_dicts(table)

    # Get feature encoding and proper normalization
    samples_enc = get_sample_bitmap(sample, queries)
    predicates_enc = encode_data(queries, min_max_dict, column2vec, op2vec)
    label_norm, min_val, max_val = normalize_labels(labels)

    #  for i in range(4):
    #      print(i, len(samples_enc[i]))
    #      print('predicates', len(predicates_enc[i]))
    #      print(label_norm[i])
    #      print('================')

    # Split in training and validation samples
    # random split, align with chi
    all_indices = list(range(num_queries))
    np.random.shuffle(all_indices)
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train
    train_idx = all_indices[:num_train]
    test_idx = all_indices[num_train:]

    samples_train = [samples_enc[i] for i in train_idx]
    predicates_train = [predicates_enc[i] for i in train_idx]
    labels_train = [label_norm[i] for i in train_idx]

    samples_test = [samples_enc[i] for i in test_idx]
    predicates_test = [predicates_enc[i] for i in test_idx]
    labels_test = [label_norm[i] for i in test_idx]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [column2vec, op2vec]
    train_data = [samples_train, predicates_train]
    test_data = [samples_test, predicates_test]
    return dicts, min_max_dict, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_data


def make_dataset(samples, predicates, labels, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        # wangxy: # joins + 1 >= # tables, for single table # join = 0
        num_pad = 0 + 1 - sample_tensor.shape[0]
        assert num_pad == 0, num_pad
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)
    #  print(sample_tensors.shape, sample_masks.shape)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)
    #  print(predicate_tensors.shape, predicate_masks.shape)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, target_tensor,
                                 sample_masks, predicate_masks)


def get_train_datasets(queries, labels, num_samples, seed):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        queries, labels, num_samples, seed)
    train_dataset = make_dataset(*train_data, labels=labels_train,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_dataset, test_dataset
