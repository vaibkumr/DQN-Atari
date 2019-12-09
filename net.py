import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size = 5, stride = 2):
    """
    Really nice function provided by official pytorch docs
    I verififed it from this amazing amazing AMAZING (can't stress enough)
    paper: https://arxiv.org/abs/1603.07285
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1


class Net(nn.Module):
    """
    Structure taken from here:
    https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/master/dqn/convnet_atari3.lua
    Note to self: Learn lua
    """
    def __init__(self, N_ACTIONS):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1)

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        # lc_inp = convw * convh * bs

        self.fc1 = nn.Linear(2560, 512) #Getting an error is faster xD
        self.fc2 = nn.Linear(512, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 2560)))
        return self.fc2(x)
