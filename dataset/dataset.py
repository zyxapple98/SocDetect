from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from .preprocess import id_2_event, clip_transform, sample_data


class ClipDataset(Dataset):

    def __init__(self, data_path):
        super(ClipDataset, self).__init__()
        self.source = data_path
        self.clip_index, self.labels = sample_data(data_path)

    def __getitem__(self, index):
        clip_index = self.clip_index[index]
        label = self.labels[index]

        clip_path = os.path.join(self.source, id_2_event[int(label[0])],
                                 'frames', str(clip_index[-1]))
        clip = []
        for index in clip_index[:-1]:
            path = os.path.join(clip_path, str(index) + '.png')
            image = Image.open(path)
            image = clip_transform(image)
            clip.append(image)
        return (torch.stack(clip), torch.tensor(label))

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = ClipDataset('/home/trunk/zyx/SocDetect/data')
    # print(dataset[0])
    # print(len(dataset))
