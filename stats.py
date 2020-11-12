import glob
import sys
import ntpath
import collections
import re
import numpy as np


np.set_printoptions(threshold=sys.maxsize)

basedir = sys.argv[1]


outs = collections.defaultdict(lambda: collections.defaultdict(str))
files = glob.glob(basedir + "/**/lottery_branch_save_activations*/*.act.npy")
for filepath in files:
    key = ntpath.basename(filepath)
    res = re.match(r'.*level_(?P<level>[0-9]+).*', filepath)
    level = int(res.group('level'))
    outs[key][level] = filepath

def dist(a, b):
    h = np.sum(np.abs(a-b))
    return h

def dist2(a, b):
    h = np.sum(np.abs(a*b - a))
    return h




for name, out_seq in outs.items():
    s_a = []
    
    totals = []
    changed = []

    for i in range(20):
        fi = out_seq[i]
        fi_data = np.load(fi)
        perc = np.percentile(fi_data, 75)
        perc = 0.01
        print("perc: " ,perc)

        #print(np.histogram(fi_data, bins=20))
        fi_data[fi_data<=perc] = 0.0
        fi_data[fi_data>perc] = 1.0
        print("count nonzero", np.count_nonzero(fi_data))
        s_a.append(fi_data)

        """
        fi_mask = fi.replace('/model.', '/').replace('.act.npy', '.weight.mask.npy')
        weight_mask = np.load(fi_mask).transpose()
        act_mask = np.sum(weight_mask, axis=0)
        #print(act_mask)
        print("am",np.count_nonzero(act_mask == 0))
        """

        #s = np.array(s_a)
        #for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    if True:
        st = np.copy(s_a)
        res = np.zeros((20, 20))
        for i in range(20):
            for o in range(20):
                #print("cnt i,o", np.count_nonzero(st[i]), np.count_nonzero(st[o]))
                res[i,o] = dist(st[i], st[o])

        print(name)
        print(res[19])