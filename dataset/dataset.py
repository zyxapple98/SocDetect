from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from .preprocess import id_2_event, npvideo_2_clip


class ClipDataset(Dataset):

    def __init__(self, data_path, train=True):
        super(ClipDataset, self).__init__()
        self.source = data_path
        self.clip_index = np.load(os.path.join(data_path, 'X.npy'))
        self.labels = np.load(os.path.join(data_path, 'y.npy'))

        len = self.clip_index.shape[0]
        test_id = np.random.choice(np.arange(len),
                                   size=int(len / 5),
                                   replace=False)
        train_id = np.delete(np.arange(len), test_id)
        if train:
            subset_idx = train_id
        else:
            subset_idx = test_id
        self.clip_index = self.clip_index[subset_idx]
        self.labels = self.labels[subset_idx]

    def __getitem__(self, index):
        clip_index = self.clip_index[index]
        label = self.labels[index]

        clip_path = os.path.join(self.source, id_2_event[int(label[0])],
                                 'frames', str(clip_index[-1]))
        clip = []
        for index in clip_index[:-1]:
            path = os.path.join(clip_path, str(index) + '.png')
            image = cv2.imread(path)
            clip.append(image)
        clip = npvideo_2_clip(clip)
        return (clip, torch.tensor(label))

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = ClipDataset('/home/trunk/zyx/SocDetect/data')
    # print(dataset[0])
    # print(len(dataset))
