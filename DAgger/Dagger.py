from torch.utils.data import DataLoader, Dataset
import load_policy
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
import numpy as np
import tf_util


class GetDaggerData(Dataset):
    def __init__(self, X_train, y_train):
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


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


def load(path):
    all = np.load(path)
    X = all['observations']
    y = all['actions']
    y1 = y.reshape(y.shape[0], y.shape[2])
    return X, y1


def np_to_Variable(observation):
    observation = torch.from_numpy(observation)
    observation = Variable(observation).cuda().float()
    return observation


def Tensor_to_np(action):
    action = action.data.cpu().numpy()
    return action


def train(Dataset):
    train_loader = DataLoader(dataset=Dataset, batch_size=32, shuffle=True)
    total_step = len(train_loader)
    # 训练
    for epoch in range(5):
        for i, (o, a) in enumerate(train_loader):
            inputs = o.to(device).float()
            a = a.to(device)
            # forward
            out = dagger_policy(inputs)
            loss = criterion(out, a)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 5, i + 1, total_step, loss.item()))
    # torch.save(dagger_policy.state_dict(), './model/' + str(k) + 'Dagger.pth')


def Data_Aggregate(init_data, agg_data):
    return np.concatenate([init_data, agg_data])


if __name__ == '__main__':
    reward = []
    envname = 'Humanoid-v2'
    # load the expert policy
    expert_policy_file = 'experts/Humanoid-v2.pkl'
    policy_fn = load_policy.load_policy(expert_policy_file)
    # generate neural network using Pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dagger_policy = Neural_Network(376, 17).to(device)
    # Loading model parameters pre-trained by behavioral cloning
    dagger_policy.load_state_dict(torch.load('./model/BehaverCloning.pth'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dagger_policy.parameters(), lr=0.0005)
    # Loading the expert data as the initial data set
    X_train_init, y_train_init = load('expert_data/Humanoid-v2.pkl')
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = None or env.spec.timestep_limit
        for k in range(20):
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
                    observa = np_to_Variable(obs)
                    predict = dagger_policy(observa)
                    action = Tensor_to_np(predict)
                    observations.append(obs)
                    # use the expert policy to get the correct action
                    expert_action = policy_fn(obs[None, :])
                    actions.append(expert_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    # env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            reward.append(np.mean(returns))
            # Data aggregation
            observations = np.array(observations)
            actions = np.array(actions)
            actions = actions.reshape(actions.shape[0], actions.shape[2])
            X_train_dagger = Data_Aggregate(observations, X_train_init)
            y_train_dagger = Data_Aggregate(actions, y_train_init)
            DaggerDataset = GetDaggerData(X_train_dagger, y_train_dagger)
            # train the model
            train(DaggerDataset)

    print(reward)
