# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
import torch
import torch.nn as nn
import datasets
from platforms.platform import get_platform
from functools import partial
import numpy as np

import collections


class Branch(base.Branch):
    def branch_function(self, start_at_step_zero: bool = False):
        state_step = self.lottery_desc.train_end_step
        model = PrunedModel(models.registry.load(self.level_root, state_step, self.lottery_desc.model_hparams), Mask.load(self.level_root))
        mask = Mask.load(self.level_root)
        mask.save(self.branch_root)

        model.eval()
        model.to(get_platform().torch_device)
        print("Saving outputs.")

        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        test_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=False)

        activations = collections.defaultdict(list)
        def save_activation(name, mod, inp, out):
            activations[name].append(out.cpu())

        for name, m in model.named_modules():
            print(name)
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))
        with torch.no_grad():
            for examples, labels in train_loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)
        
        activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
        for name, act in activations.items():
            np.save(os.path.join(self.branch_root, name + ".act.npy"), act)

        for name, ma in mask.numpy().items():
            np.save(os.path.join(self.branch_root, name + ".mask.npy"), ma)

    @staticmethod
    def description():
        return "Save model activations."

    @staticmethod
    def name():
        return 'save_activations'
