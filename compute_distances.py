import glob
import sys
import ntpath
import collections
import re
import numpy as np
import os

#USAGE: python compute_distances.py DIR1 DIR2, both ending in replicate_#/
#Computes for same level.

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

  #n = gram.shape[0]
  #np.fill_diagonal(gram, 0)
  #means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
  #means -= np.sum(means) / (2 * (n - 1))
  #gram -= means[:, None]
  #gram -= means[None, :]
  #np.fill_diagonal(gram, 0)
  means = np.mean(gram, 0, dtype=np.float64)
  means -= np.mean(means) / 2
  gram -= means[:, None]
  gram -= means[None, :]

  return gram

#Load paths in a 2 element list of dictionaries.
#Each dictionary maps j to the path of activations of the j'th layer (of pruning level=level).
def load_paths(basedir1, basedir2):#, level):

    def name_format(path):
        if path.endswith('/'):
            return path
        else:
            return path + '/'
    
    basedirs = [name_format(basedir1), name_format(basedir2)]
    layer_to_path = []
    for i in range(2):
        layer_to_path.append(collections.defaultdict(lambda: collections.defaultdict(str)))
        direct = basedirs[i] + "lottery_branch_save_activations*/*.act.npy"
        files = glob.glob(direct)
        #print(files)
        for filepath in files:
            #print(filepath)
            #NON CONV:#
            #res = re.match(r'.*level_(?P<level>[0-9]+).*net.(?P<layer>[0-9]+).*', filepath)
            #CONV:#
            res = re.match(r'.*level_(?P<level>[0-9]+).*blocks.(?P<layer>[0-9]+).*conv(?P<conv>[0-9]+).*', filepath)
            if res: #and int(res.group('level')) == level:
                layer = int(res.group('layer'))
                conv = int(res.group('conv'))
                layer_to_path[i][2*layer+conv-1] = filepath
                #layer_to_path[i][layer] = filepath
    return layer_to_path


#MAIN

#Parameters: Number of layers, level of pruning.
#level = 7
print(sys.argv[1], sys.argv[2])
paths = load_paths(sys.argv[1], sys.argv[2])#, level = level)

assert len(paths[0].keys()) == len(paths[1].keys()), 'Loading activations failed, number of layers mismatched.'

n_layers = len(paths[0].keys())
#ERROR CHECK:
print(n_layers, paths[0].keys())
for layer in paths[0].keys():
    for i in range(2):
        print(paths[i][layer])
        

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
                      "Computing distances on {} layers, for experiments: \n".format(n_layers)+\
                      sys.argv[1] + '\n' + \
                      sys.argv[2] + '\n'
    exp_placeholder.write(experiment_info)


#Compute Centered distances.
for layer in paths[0].keys():
    for i in range(2):
        print("Loading activations of dataset {}, layer {}:".format(i, layer))
        print(paths[i][layer])
        layer_activations = np.load(paths[i][layer])
        print("Computing distances of layer {}, shape {}".format(layer, layer_activations.shape))
        activation_distances = distances(layer_activations)
        np.save(distances_dir + 'distances_{}_layer_{}.npy'.format(i, layer), activation_distances)

with open(distances_dir + 'experiment_{}.txt'.format(exp_id), "a") as exp_placeholder:
    exp_placeholder.write("Distances computed.")
