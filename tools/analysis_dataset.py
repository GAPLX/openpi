import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


repo_id = "danceinyourface/pickplace_19"
dataset = LeRobotDataset(repo_id)

print(f"number of episodes: {dataset.num_episodes} \n")

sample = dataset[300]  # frame
for k in sample.keys():
    v = sample[k]
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)}")

print("\ntimestamp: ", sample["timestamp"])
print("frame_index: ", sample["frame_index"])
print("episode_index: ", sample["episode_index"])
print("index: ", sample["index"])
print("task_index: ", sample["task_index"])
