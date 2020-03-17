import os
import csv
import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_sample, load_dicts, make_dataset
from mscn.model import SetConv
from mscn.datasets import LoadForest
from mscn.queries import LoadForestQueries

def error_metric(est_card, card):
    # HACK: the error should not less than 1.1 when est_card != card and one of them is 0
    # TODO: may be add 1 on both est_card and card?
    if card == 0 and est_card != 0:
        return max(1.1, est_card)
    if card != 0 and est_card == 0:
        return max(1.1, card)
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals) - 1

def qerror_loss(preds, targets, min_val, max_val, collapse=True):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        qerror.append(error_metric(preds[i], targets[i]))
    return torch.mean(torch.cat(qerror)) if collapse else qerror

def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        qerror.append(error_metric(float(preds_unnorm[i]), float(labels_unnorm[i])))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return np.array(qerror)

def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, targets, sample_masks, predicate_masks = data_batch

        if cuda:
            samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
            sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
        samples, predicates, targets = Variable(samples), Variable(predicates), Variable(
            targets)
        sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)

        t = time.time()
        outputs = model(samples, predicates, sample_masks, predicate_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total

def train(query, num_samples, num_epochs, batch_size, hid_units, cuda, seed):

    # load queires
    queries, labels = LoadForestQueries(query, split_close_range=True)

    # Load training and validation data
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_data = get_train_datasets(
        queries, labels, num_samples, seed)
    column2vec, op2vec = dicts

    # Train model
    predicate_feats = len(column2vec) + len(op2vec) + 1
    state = {
        'min_val': min_val,
        'max_val': max_val
    }

    model = SetConv(num_samples, predicate_feats, hid_units)
    model_size = model.size()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_valid_loss = float('inf')

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, targets, sample_masks, predicate_masks = data_batch

            if cuda:
                samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
                sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
            samples, predicates, targets = Variable(samples), Variable(predicates), Variable(
                targets)
            sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, sample_masks, predicate_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

        # Get final training and validation set predictions
        #  preds_train, t_total = predict(model, train_data_loader, cuda)
        #  print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

        preds_test, t_total = predict(model, test_data_loader, cuda)
        print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

        # Unnormalize
        #  preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
        #  labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)
        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
        labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

        # Print metrics
        #  print("\nQ-Error training set:")
        #  train_qerror = print_qerror(preds_train_unnorm, labels_train_unnorm)

        print("\nQ-Error validation set:")
        test_qerror = print_qerror(preds_test_unnorm, labels_test_unnorm)
        print("")

        valid_loss = test_qerror.mean()
        if valid_loss < best_valid_loss:
            print('best valid loss for now!', valid_loss)
            best_valid_loss = valid_loss
            state['model_state_dict'] = model.state_dict()
            torch.save(state, os.path.join('model', '{}_{}_{}_{:.2f}.pt'
                                                        .format(num_samples, hid_units, seed, model_size)))

def test(query, num_samples, model_name, batch_size, hid_units, cuda, seed):
    # load queires
    queries, labels = LoadForestQueries(query, split_close_range=True)

    # load sample
    sample = load_sample(num_samples, seed)

    # load dicts from data
    table = LoadForest()
    column2vec, op2vec, column_min_max_vals = load_dicts(table)

    # load model
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(num_samples, predicate_feats, hid_units)
    model_size = model.size()
    state = torch.load(os.path.join('model', '{}_{}_{}_{:.2f}.pt'
                                    .format(num_samples, hid_units, seed, model_size)))
    model.load_state_dict(state['model_state_dict'])

    # load min max label from model dict
    min_val = state['min_val']
    max_val = state['max_val']
    print('min val: {}, max_val: {}'.format(min_val, max_val))

    # Get feature encoding and proper normalization
    samples_enc = get_sample_bitmap(sample, queries)
    predicates_enc = encode_data(queries, column_min_max_vals, column2vec, op2vec)
    labels_test, _, _ = normalize_labels(labels, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_enc])

    # Get test set predictions
    test_data = make_dataset(samples_enc, predicates_enc, labels_test, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error:")
    test_error = print_qerror(preds_test_unnorm, labels)

    # TODO: align output with naru
    # Write predictions
    file_name = os.path.join('results', '{}_{}.csv'.format(query, model_name))
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        for i in range(len(preds_test_unnorm)):
            writer.writerow((test_error[i], preds_test_unnorm[i], labels[i]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="train / test", type=str, default='train')
    parser.add_argument("--query", help="query name", type=str, default='q20k')
    parser.add_argument("--model", help="model name", type=str, default='')
    parser.add_argument("--seed", help="random seed", type=int, default=123)
    parser.add_argument("--samples", help="number of materialized samples", type=int, default=1000)
    parser.add_argument("--epochs", help="number of epochs (default: 500)", type=int, default=500)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 28)", type=int, default=28)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cmd == 'train':
        train(args.query, args.samples, args.epochs, args.batch, args.hid, args.cuda, args.seed)
    elif args.cmd == 'test':
        test(args.query, args.samples, args.model, args.batch, args.hid, args.cuda, args.seed)


if __name__ == "__main__":
    main()
