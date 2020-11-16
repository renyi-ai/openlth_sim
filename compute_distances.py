import glob
import sys
import ntpath
import collections
import re
import numpy as np
import os


np.set_printoptions(threshold=sys.maxsize)

# Compute euclidean distances between N examples.
# Input: N x ..., Output: N x N.
def distances(x):
    x_vector = np.reshape(x, (x.shape[0],-1))
    
    D = x_vector.dot(x_vector.T)
    print("Centering...")
    return center_gram(D)
  

#Center symmetric matrix.
def center_gram(gram):
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  means = np.mean(gram, 0, dtype=np.float64)
  means -= np.mean(means) / 2
  gram -= means[:, None]
  gram -= means[None, :]

  return gram

#Load paths in a 2 element list of dictionaries.
#Each dictionary maps i to the path of activations of the i'th layer (of level=level).
def load_paths(basedir1, basedir2, level):
    basedirs = [basedir1, basedir2]
    layer_to_path = []
    for i in range(2):
        layer_to_path.append(collections.defaultdict(lambda: collections.defaultdict(str)))
        files = glob.glob(basedirs[i] + "/**/lottery_branch_save_activations*/*.act.npy")
        for filepath in files:
            res = re.match(r'.*level_(?P<level>[0-9]+).*net.(?P<net>[0-9]+).*', filepath)
            if res and int(res.group('level')) == level:
                layer = int(res.group('net'))
                layer_to_path[i][layer] = filepath
    return layer_to_path


#MAIN

#Parameters: Number of layers, level of pruning.
level = 0
paths = load_paths(sys.argv[1], sys.argv[2], level = level)
n_layers = 2#max(paths[0].keys())


#SETUP environment - directory, experiment ids.
distances_basedir = '../storage/distances/'
assert os.path.isdir(distances_basedir), '{} is not a valid base directory'.format(distances_basedir)
#Create experiment directory.
exp_id = 0
while os.path.isdir(distances_basedir + "experiment_{}/".format(exp_id)):
    exp_id += 1
distances_dir = distances_basedir + "experiment_{}/".format(exp_id)
try:
    os.mkdir(distances_dir)
except OSError:
    print("Failed to create {}.".format(distances_dir))
else:
    print("Created {}".format(distances_dir))
#Save IDs.
with open(distances_dir + 'experiment_{}.txt'.format(exp_id), "w+") as exp_placeholder:
    experiment_info = '{}\n'.format(n_layers)+\
                      "Computing distances on first {} layers, on level {}, for experiments: \n".format(n_layers, level)+\
                      sys.argv[1] + '\n' + \
                      sys.argv[2] + '\n'
    exp_placeholder.write(experiment_info)


#Compute Centered distances.
for layer in range(n_layers):
    for i in range(2):
        layer_activations = np.load(paths[i][layer])
        print("Computing distances of dataset {}, layer {} of shape {}...".format(i, layer, layer_activations.shape))
        activation_distances = distances(layer_activations)
        np.save(distances_dir + 'distances_{}_layer_{}.npy'.format(i, layer), activation_distances)

with open(distances_dir + 'experiment_{}.txt'.format(exp_id), "a") as exp_placeholder:
    exp_placeholder.write("Distances computed.")
