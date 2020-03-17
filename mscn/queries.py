"""Query Formalization"""
import os
import csv

DATA_PATH = '/var/tmp/cardDB/data'

def ParseInteger(lrange, rrange):
    # TODO
    if lrange == 'MAX' or rrange == 'MIN':
        return None
    assert lrange != 'MAX' and rrange != 'MIN', (lrange, rrange)
    if lrange == 'MIN' and rrange == 'MAX':
        return None
    elif lrange == 'MIN':
        return '<=', int(float(rrange))
    elif rrange == 'MAX':
        return '>=', int(float(lrange))
    elif lrange == rrange:
        return '=', int(float(lrange))
    lrange = int(float(lrange))
    rrange = int(float(rrange))
    assert lrange < rrange, (lrange, rrange)
    return '[]', (lrange, rrange)

def LoadForestQueries(filename='query', split_close_range=False):
    csv_file = os.path.join(DATA_PATH, 'forest', '{}.csv'.format(filename))
    print('load from query file: {}'.format(csv_file))
    num_attr = 10
    queries = []
    labels = []

    with open(csv_file, 'r') as infile:
        reader = csv.reader(infile, delimiter=',', quotechar='|')
        for query in reader:
            #  print(query)
            col_idxs = []
            ops = []
            vals = []
            for i in range(num_attr):
                # all values in forests are integers
                parsed = ParseInteger(query[i*2], query[i*2+1])
                if parsed is None:
                    continue
                else:
                    t_op, t_val = parsed
                    if t_op == '[]' and split_close_range:
                        assert type(t_val) is tuple, t_val
                        col_idxs.append(i)
                        ops.append('>=')
                        vals.append(t_val[0])
                        col_idxs.append(i)
                        ops.append('<=')
                        vals.append(t_val[1])
                    else:
                        col_idxs.append(i)
                        ops.append(t_op)
                        vals.append(t_val)
            queries.append((col_idxs, ops, vals))
            labels.append(int(query[-1]))

    print('{} queries in total'.format(len(queries)))

    return queries, labels




