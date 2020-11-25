import glob
import sys
import ntpath
import collections
import re
import numpy as np
import os

#USAGE: python compute_cka.py # where # is the experiment number (directory # of computed distances)
#SET distances_dir


experiment_id = sys.argv[1]
mod_n_layers = 1
distances_dir = '../storage/distances/experiment_{}/'.format(experiment_id)
with open(distances_dir + 'experiment_{}.txt'.format(experiment_id)) as f:
  #saved number of layers as first line...
  n_layers = int(f.readline())
print("{} layers.".format(n_layers))

def cka(centered_gram_x, centered_gram_y):
  scaled_hsic = centered_gram_x.ravel().dot(centered_gram_y.ravel())

  normalization_x = np.linalg.norm(centered_gram_x)
  normalization_y = np.linalg.norm(centered_gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

#Compute CKA.
cka_matrix = np.zeros((n_layers, n_layers))
for layer_0 in range(n_layers):
    distances_0 = np.load(distances_dir + 'distances_0_layer_{}.npy'.format(layer_0 * mod_n_layers))
    for layer_1 in range(layer_0, n_layers):
        print("Loading data for layer {}...".format(layer_1))
        distances_1 = np.load(distances_dir + 'distances_1_layer_{}.npy'.format(layer_1 * mod_n_layers))

        print("Computing CKA of layers {} vs {}".format(layer_0, layer_1))
        cka_matrix[layer_0, layer_1] = cka(distances_0, distances_1)

        print('CKA on layers {}, {} = {}'.format(layer_0, layer_1, cka_matrix[layer_0, layer_1]))
print(cka_matrix)
np.save(distances_dir + 'cka_matrix.npy', cka_matrix)

with open(distances_dir + 'experiment_{}.txt'.format(experiment_id), "a") as exp_placeholder:
    exp_placeholder.write("\n" + "CKA computed.")
