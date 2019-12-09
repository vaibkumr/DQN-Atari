import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import math
import random

def torch_state(state):
    return torch.Tensor(state).permute(0, 3, 1, 2).cuda()

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]
    #the numpy notation a[a:b:c] gives every cth element between a and b. [::2]
    #means every 2nd element between start and end

def preprocess(img):
    i = to_grayscale(downsample(img))
    return i[..., None]
    #numpy notation a[..., None] adds another dimension at the end

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] #an array of namedtuples
        self.position = 0
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward')) #Tahnks to official pytorch tutorial for this wonderful idea


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
