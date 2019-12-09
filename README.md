# DQN for Atari
![](https://i.imgur.com/RkMH4sO.jpg)
One of my main motivations towards studying Reinfrocement Learning was being able to create an agent that could play atari games (since I was terrible at them). I have implemented DQN before but it was a comparatively very simple task (job scheduling). Playing Atari games is a completely different task (and much diffucult in terms of computational complexity). Even cartpool was so much easier.

I wrote this simple, intuitive and easy to understand DQN implementation in PyTorch a few months ago. Sharing it for others now, if you find any bugs please raise an issue (there must be many).

I have trained this model on the `BreakoutDeterministic-v4` gym environment

Check out `config/config.ini` for the hyperparameters used (mostly taken from other mentioned sources).

# Requirements
- Pytorch
- openAI gym
- numpy
- matplotlib

# How to?
### Train:
`git clone https://github.com/TimeTraveller-San/DQN-Atari`
`python train_atari.py`
> How long does it take to train?

On my GTX 1060 6GB, 16 GB RAM and i7 6700k
- it took the agent ~1 hour to start hitting the ball
- it took the agent ~13 hours to reach the score of ~50
### Have a glance at the harbinger of humanity's demise (test agent):
`python test_atari.py --m <MODEL PATH>`

# TODO (ideas)
- Try pretrained CNNs.. small resnets and EfficientNets
- Parallel programming for environment render (CPU < 30% used and RAM < 4GB used for now)
- Transfer learning? (have to test how well does an agent generalize over different games, I expect it to not generalize with current architecture at all)
- Parameter tuning
- Double-DQN (RIP my GPU x_x)
- Write in Lua (like the repo below)
- Train for other than breakout

# Thanks to
- [Paper1](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) - I failed terribly by trying to implement this by just reading the paper. Nevertheless, it was very helpful
- [Paper2](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) - Demis Hassabis is a god
- https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner - I shamelessly stole my hyperparameters from here (I have no idea how to tune them with my limited resources x_x). Also, Note to self: Learn Lua
- [Official PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - PyTorch tutorials are the best!



# 8 bit is life
![](https://steamuserimages-a.akamaihd.net/ugc/351646214584191107/95FC195376E31B4F0DC30C3FF137BD2DF73B151D/)
