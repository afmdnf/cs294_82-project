import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest


class FilterPruner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations, self.gradients = [], []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_plan(self, num_filters_to_prune):
        filters_to_prune_per_layer = {}
        for (l, f, _) in self.lowest_ranking_filters(num_filters_to_prune):
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        return filters_to_prune


class PruningFineTuner:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct, total = 0, 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Test Accuracy: ", float(correct) / total)
        self.model.train()

    def train(self, optimizer=None, epochs=5):
        if optimizer is None:
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epochs):
            print("Training Epoch: [%d/%d]" % (i+1, epochs))
            self.train_epoch(optimizer)
            self.test()
            print()

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        batch_input = Variable(batch)

        if rank_filters:
            output = self.pruner.forward(batch_input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(batch_input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filters=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        self.test()
        for param in self.model.features.parameters():
            param.requires_grad = True

        num_filters = self.total_num_filters()
        num_filters_to_prune_per_iter = 512
        iterations = int(float(num_filters) / num_filters_to_prune_per_iter)
        iterations = int(iterations * 2.0 / 3)
        print("Number of pruning iterations to remove 67% filters:", iterations)

        for _ in range(iterations):
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iter)
            layers_pruned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_pruned:
                    layers_pruned[layer_index] = 0
                layers_pruned[layer_index] = layers_pruned[layer_index] + 1

            print("Layers to be pruned:", layers_pruned)
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index)

            self.model = model
            msg = str(100*float(self.total_num_filters()) / num_filters) + "%"
            print("Filters pruned:", str(msg))
            self.test()
            print("Fine tuning...")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epochs=2)
            print("======================================")

        print("Final fine tuning...")
        self.train(optimizer, epochs=5)
        torch.save(model.state_dict(), "pruned_model")


class MyVGG16Model(torch.nn.Module):
    def __init__(self):
        super(MyVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.train:
        model = MyVGG16Model()
    elif args.prune:
        model = torch.load("trained_model", map_location=lambda storage, loc: storage)

    fine_tuner = PruningFineTuner(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epochs=5)
        torch.save(model, "trained_model")
        fine_tuner.prune()
    elif args.prune:
        fine_tuner.prune()
