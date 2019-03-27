import pickle
import numpy as np

f = open('./expert_data/Humanoid-v2.pkl','rb')

info = pickle.load(f)

print(info.keys())
actions = info["actions"]

print(np.ndim(actions))
print(np.ndim(actions[0]))
print(actions[0])
