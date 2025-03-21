import torch.utils.data as data
from os.path import join
import numpy as np
import torch
import ast

device = torch.device("cpu")


class READ_DATA(data.Dataset):
    def __init__(self, root):
        self.root = root

        infile = open(self.root, 'r').readlines()
        indices = []
        for line in infile:
            if line.startswith('/scratch/project_2003370/'):
                index = infile.index(line)
                indices.append(index)
        self.items = [(join(str(infile[indices[ind]:indices[ind + 1]])),
                       (join(str(infile[indices[ind]:indices[ind + 1]]).split('_')[-1]))) for ind in
                      range(len(indices) - 1)]

        self.data = [x[0].split('_[')[0].replace('[', '').replace('\'', '') for x in self.items]
        self.targets = [
            ast.literal_eval(x[1].replace(' ', ',').replace('\\n', '').replace('\']', '').replace('\',,\'', '', )) for x
            in self.items]
        self.items = [x for x in self.items]
        self.targets = np.array(self.targets)

        self.data2 = []
        self.targets2 = []
        self.scenes2 = []
        self.items2 = []

        for a in range(len(self.data)):
            self.data2.append((self.data[a]))
            self.targets2.append((self.targets[a]))
            self.items2.append((self.items[a]))

        self.data = self.data2
        self.targets = self.targets2
        self.items = self.items2

        print("Total files =", np.array(self.targets).shape, np.array(self.data).shape)

