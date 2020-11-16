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
  

def cka(centered_gram_x, centered_gram_y):
  scaled_hsic = centered_gram_x.ravel().dot(centered_gram_y.ravel())

  normalization_x = np.linalg.norm(centered_gram_x)
  normalization_y = np.linalg.norm(centered_gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


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
n_layers = 1#max(paths[0].keys())


#SETUP environment - directory, experiment ids.
distances_basedir = '../storage/distances/'
assert os.path.isdir(distances_basedir), '{} is not a valid base directory'.format(distances_basedir)
#Create experiment directory.
i = 0
while os.path.isdir(distances_basedir + "experiment_{}/".format(i)):
    i += 1
distances_dir = distances_basedir + "experiment_{}/".format(i)
try:
    os.mkdir(distances_dir)
except OSError:
    print("Failed to create {}.".format(distances_dir))
else:
    print("Created {}".format(distances_dir))
#Save IDs.
with open(distances_dir + 'experiment_{}.txt'.format(i), "w+") as experiment_id:
    experiment_id.write("Computing on first {} layers, for experiments \n".format(n_layers))
    experiment_id.write(sys.argv[1] + "\n")
    experiment_id.write(sys.argv[2])


#Compute Centered distances.
for layer in range(n_layers):
    for i in range(2):
        layer_activations = np.load(paths[i][layer])
        print("Computing distances of dataset {}, layer {}...".format(i, layer))
        activation_distances = distances(layer_activations)
        np.save(distances_dir + 'distances_{}_layer_{}.npy'.format(i, layer), activation_distances)

#Compute CKA.
cka_matrix = np.zeros((n_layers, n_layers))
for layer_0 in range(n_layers):
    distances_0 = np.load(distances_dir + 'distances_0_layer_{}.npy'.format(layer_0))
    for layer_1 in range(layer_0, n_layers):
        print("Loading data for layer {}...".format(layer_1))
        distances_1 = np.load(distances_dir + 'distances_1_layer_{}.npy'.format(layer_1))

        print("Computing CKA of layers {} vs {}".format(layer_0, layer_1))
        cka_matrix[layer_0, layer_1] = cka(distances_0, distances_1)

        print('CKA on layers {}, {} = {}'.format(layer_0, layer_1, cka_matrix[layer_0, layer_1]))
print(cka_matrix)
np.save(distances_dir + 'cka_matrix.npy', cka_matrix)

with open(distances_dir + 'experiment_{}.txt'.format(i), "w+") as experiment_id:
    experiment_id.write("Experiment finished.")
