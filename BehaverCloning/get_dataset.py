import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable


def load(path):
    all = np.load(path)
    X = all['observations']
    y = all['actions']
    y1 = y.reshape(y.shape[0], y.shape[2])
    return X, y1


class DealDataset(Dataset):
    def __init__(self, data_path):
        X_train, y_train = load(data_path)
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dealDataset = DealDataset('./expert_data/Humanoid-v2.pkl')
train_loader = DataLoader(dataset=dealDataset, batch_size=32, shuffle=True)
print(train_loader)
for epoch in range(2):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
