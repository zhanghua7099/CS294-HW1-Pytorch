import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
import numpy as np
import tf_util


class Neural_Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Neural_Network, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def load_policy(observation):
    observation = torch.from_numpy(observation)
    observation = Variable(observation).cuda().float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Neural_Network(376, 17).to(device)
    policy.load_state_dict(torch.load('./model/test_param.pth'))
    predict = policy(observation)
    predict = predict.data.cpu().numpy()
    return predict


envname = 'Humanoid-v2'
with tf.Session():
    tf_util.initialize()
    import gym
    env = gym.make(envname)
    max_steps = None or env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    for i in range(20):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = load_policy(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # if render:
            env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
