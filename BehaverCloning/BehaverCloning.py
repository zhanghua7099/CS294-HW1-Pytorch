import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load(path):
    all = np.load(path)
    X = all['observations']
    y = all['actions']
    y1 = y.reshape(y.shape[0], y.shape[2])
    return X, y1


class GetPolicyData(Dataset):
    def __init__(self, data_path):
        X_train, y_train = load(data_path)
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


if __name__ == '__main__':
    train_path = './expert_data/Humanoid-v2.pkl'
    # pkl to dataset
    dealDataset = GetPolicyData(train_path)
    # load train data
    train_loader = DataLoader(dataset=dealDataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Neural_Network(376, 17).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    total_step = len(train_loader)
    for epoch in range(20):
        for i, (observations, actions) in enumerate(train_loader):
            inputs = observations.to(device).float()
            actions = actions.to(device)
            # forward
            out = model(inputs)
            loss = criterion(out, actions)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 20, i + 1, total_step, loss.item()))
    torch.save(model.state_dict(), './model/test_param.pth')
