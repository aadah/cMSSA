import os
import numpy as np
import pandas as pd
import rarfile

from collections import defaultdict


# Load MHEALTH dataset, located for download at:
#   https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset
#
# Once unzipped, the path to the root directory can be
# passed in to load the dataset. The `partition` argument
# allows for the user to specify an interger > 1; the loader
# will split each time series into that number of pieces.
# This is usually done to artificially increase the number
# of time series.
#
# Returns a dict mapping an activity to all time series of that activity.
# Each key is simply a human-readable string.
def load_mhealth_data(root_path, partition=None):
    tasks = [
        'null',
        'standing',
        'sitting',
        'lying',
        'walking',
        'climbing',  # climbing stairs
        'bending',  # bending waist forward
        'raising',  # raising arms in front
        'crouching',  # bending knees
        'cycling',
        'jogging',
        'running',
        'jumping'  # jumping front/back- wards
    ]

    data = defaultdict(list)

    for i in range(10):
        subject = i + 1
        path = os.path.join(root_path, 'mHealth_subject{}.log'.format(subject))
        df = pd.read_csv(path, sep='\t')

        # cardio signals + label
        ecg_data = df.values[:, [3, 4, -1]]
        labels = ecg_data[:, -1]
        breaks = np.argwhere(labels[1:] != labels[:-1]).reshape([-1])+1
        arrs = np.split(ecg_data, breaks, axis=0)

        for arr in arrs:
            assert len(set(arr[:, -1])) == 1
            task = tasks[int(arr[0, -1])]
            X = arr[:, :-1]

            if partition is not None:
                Xs = np.array_split(X, partition, axis=0)
                data[task].extend(Xs)
            else:
                data[task].append(X)

    return data


# Load Epilepsy dataset, available for download here:
#   http://timeseriesclassification.com/description.php?Dataset=Epilepsy
#
# Once unzipped, the path to the root directory can be
# passed in to load the dataset. The `dataset` argument
# can be either "train" or "test", to load that segment
# of the data.
#
# Returns a dict mapping an identifier to a list of time series.
# The identifiers are one of 4 activities.
def load_epilepsy_data(root_path, dataset='train'):
    path = os.path.join(root_path, 'Epilepsy_{}.arff'.format(dataset.upper()))
    f = open(path)
    data = defaultdict(list)
    for line in f:
        if line.startswith("'"):
            cat = line.split(',')[-1]
            data_str = line[1:-(len(cat)+2)]
            cat = cat.strip()
            parts = data_str.split('\\n')
            assert len(parts) == 3
            for part in parts:
                d = np.array([float(num) for num in part.split(',')])
                data[cat].append(d.reshape((-1, 1)))
    return data


# Load Earthquakes dataset, available for download here:
#   http://timeseriesclassification.com/description.php?Dataset=Earthquakes
#
# Once unzipped, the path to the root directory can be
# passed in to load the dataset. The `dataset` argument
# can be either "train" or "test", to load that segment
# of the data.
#
# Returns a dict mapping an identifier to a list of time series.
# The identifiers are either "EARTHQUAKE" or "NON-EARTHQUAKE".
def load_earthquake_data(root_path, dataset='train'):
    path = os.path.join(root_path, 'Earthquakes_{}.txt'.format(dataset.upper()))
    data = pd.read_csv(path).values
    eq = defaultdict(list)
    for i in range(data.shape[0]):
        label = data[i, 0]
        series = data[i, 1:].reshape((-1, 1))
        if label == 1:
            eq['EARTHQUAKE'].append(series)
        else:
            eq['NON-EARTHQUAKE'].append(series)
    return eq


# Load EMG Physical Activity dataset, available for download here:
#   https://archive.ics.uci.edu/ml/datasets/EMG+Physical+Action+Data+Set
#
# The `path` is the path to the downloaded RAR file. The `partition` may be
# an integer > 1, specifying how many partitions to create per time series.
#
# Returns a dict mapping an activity to all time series of that activity.
# Each key is of the form '[normal|aggressive]/ACTIVITY'.
def load_emg_data(path, partition=None):
    data = defaultdict(list)

    def split(p):
        parts = []
        head, tail = p, ''
        while head != '':
            head, tail = os.path.split(head)
            parts.append(tail)
        parts.reverse()
        return parts

    def read_timeseries(f):
        X = []
        for line in f:
            try:
                X.append(list(map(float, line.decode('utf-8').strip().split('\t'))))
            except Exception:
                # all of sub3's files end in a line of all tabs, so just skip
                pass
        return np.array(X)

    rf = rarfile.RarFile(path)
    for f in rf.infolist():
        if not f.filename.endswith('.txt'):
            continue
        parts = split(f.filename)

        # catches readme.txt
        if len(parts) < 4:
            continue

        activity = os.path.splitext(parts[-1])[0].lower()
        kind = parts[-3].lower()
        with rf.open(f.filename) as ff:
            X = read_timeseries(ff)
        key = '{}/{}'.format(kind, activity)
        if partition is not None:
            Xs = np.array_split(X, partition, axis=0)
            data[key].extend(Xs)
        else:
            data[key].append(X)

    return data


# Helper function for generating synthetic sum-of-sinusoids time series data.
# Used to generate demonstrate illustrative example in NeurIPS 2018 paper.
# Has arguments for mixing in a hidden sub-signal, as done in the paper.
def get_dummy_data(measurement_noise=0.1,
                   num_signals=10,
                   f=np.sin,
                   T=100,
                   dt=0.01,
                   add_sub=False,
                   return_sub=False,
                   sub_yscale=1.0,
                   sub_xscale=10.0,
                   sub_xshift=0,  # integer. slides the signal's mask
                   sub_width=300,
                   sub_spacing=1000,
                   seed=42):
    ts = np.arange(0, T, dt)

    if measurement_noise:
        signal = np.random.normal(0, measurement_noise, ts.shape)
    else:
        signal = 0

    np.random.seed(seed)
    xscales = np.random.randn(num_signals) * 1.0  # mostly [-1,1]
    yscales = np.random.randn(num_signals) * 1.0  # mostly [-1,1]
    xshifts = np.random.randn(num_signals) * T  # mostly [-T,T]
    yshifts = np.random.randn(num_signals) * 1.0  # mostly [-1,1]

    for i in range(len(xscales)):
        signal += yscales[i] * f(xscales[i]*ts + xshifts[i]) + yshifts[i]

    if add_sub:
        idxs = np.arange(len(ts)) - sub_xshift
        mask = idxs % sub_spacing < sub_width
        sub = f(sub_xscale * ts) * sub_yscale
        sub *= mask
        signal += sub
        if return_sub:
            return signal.reshape([-1, 1]), sub.reshape([-1, 1])

    return signal.reshape([-1, 1])
