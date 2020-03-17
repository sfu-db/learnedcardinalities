import numpy as np

def in_between(data, val):
    assert len(val) == 2
    lrange, rrange = val
    return (data >= lrange) & (data <= rrange)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '[]': in_between
}


# Helper functions for data processing

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def normalize_labels(labels, min_val=None, max_val=None):
    # +1 to deal with 0 scenario
    labels = np.array([np.log(float(l+1)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    # -1 to deal with 0 scenario, need to restrict to >= 0
    return np.array(np.maximum(np.round(np.exp(labels) - 1), 0), dtype=np.int64)


def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc

def get_sample_bitmap(sample, queries):
    bitmaps = []
    for i, query in enumerate(queries):
        bitmaps.append(list())
        columns, operators, values = query
        bitmap = np.array([1] * sample.shape[0])
        for c, o, v in zip(columns, operators, values):
            bitmap *= OPS[o](sample[:, c], v)
        bitmaps[i].append(bitmap)
    return bitmaps

def encode_data(queries, min_max_dict, column2vec, op2vec):
    predicates_enc = []
    for i, query in enumerate(queries):
        predicates_enc.append(list())
        columns, operators, values = query
        for c, o, v in zip(columns, operators, values):
            norm_val = normalize_data(v, c, min_max_dict)
            pred_vec = []
            pred_vec.append(column2vec[c])
            pred_vec.append(op2vec[o])
            pred_vec.append(norm_val)
            pred_vec = np.hstack(pred_vec)
            predicates_enc[i].append(pred_vec)

        # for no predicate scenario
        if len(predicates_enc[i]) == 0:
            predicates_enc[i].append(np.zeros((len(column2vec) + len(op2vec) + 1)))

    return predicates_enc
