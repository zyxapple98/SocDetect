import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import torch
from torchvision import transforms
from tqdm import trange

event_2_id = {"play": 0, "throwin": 1, "challenge": 2, "background": 3}
id_2_event = list(event_2_id.keys())


def event_2_npvideo(row, before=1, after=1, sample_every=2, verbose=False):
    """
    transform an event to a numpy video
    default setting: collect frames from 5 sec before event to 5 sec after event,
    and sample a frame from every 10 frames,
    resulting in 26 low-resolution frames for each event
    """
    if verbose:
        print(row["event_attributes"])
    filename = "temp.mp4"
    ffmpeg_extract_subclip(
        f"../../trunk/zyx/SocDetect/train/{row['video_id']}.mp4",
        int(row['time']) - before,
        int(row['time']) + after,
        targetname=filename)
    vidcap = cv2.VideoCapture("temp.mp4")
    np_video = []

    success, image = vidcap.read()
    count = 0
    while success:
        if not count % sample_every:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, [398, 224])
            np_video.append(image)
        success, image = vidcap.read()

        count += 1
    return np_video  # RGB np arrays


clip_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def npvideo_2_clip(np_video):
    clip = []
    for image in np_video:
        image = Image.fromarray(image)
        image = clip_transform(image)
        clip.append(image)
    return torch.stack(clip)


def event_2_clip(row,
                 out_path='data/frames',
                 before=1,
                 after=1,
                 sample_every=2,
                 save=False):
    clip = npvideo_2_clip(event_2_npvideo(row, before, after, sample_every))
    if save:
        path = os.path.join(out_path, str(row['index']))
        if not os.path.exists(path):
            os.makedirs(path)
        for i, img in enumerate(clip):
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(np.uint8((img / 2 + 0.5) * 255),
                               cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(path, f"{i}.png"), img)
    return clip


def slide(row):
    label = event_2_id[row['event']]
    samples = []
    labels = []
    offset = 0
    while offset < 51:
        idx = list(range(offset, offset + 51, 5))
        idx.append(row['index'])
        samples.append(idx)
        labels.append(np.array([label, (50 - offset) / 50]))
        offset += 2
    return np.array(samples), np.array(labels)


def generate_data(root_dir, class_name, out_path='data'):
    print(f"generating data of {class_name}...")
    data_path = os.path.join(root_dir, 'train.csv')
    df = pd.read_csv(data_path)
    save_path = os.path.join(root_dir, out_path, class_name)
    if os.path.exists(save_path):
        return
    X = []
    y = []
    if class_name in ["play", "throwin", "challenge"]:
        df_event = df[df["event"] == class_name].reset_index()
        for i in trange(df_event.shape[0]):
            # generate frames
            event_2_clip(df_event.iloc[i],
                         out_path=os.path.join(save_path, 'frames'),
                         before=2,
                         after=2,
                         sample_every=1,
                         save=True)

            # generate labels
            samples, labels = slide(df_event.iloc[i])
            X.append(samples)
            y.append(labels)

    elif class_name == 'background':
        df_mark = df[df["event"].isin(["start", "end"])].reset_index()
        tp = 0
        background_num = 0
        last_video_id = df_mark['video_id'][0]
        for i in trange(df_mark.shape[0]):
            row = df_mark.iloc[i]
            if row['video_id'] != last_video_id:
                tp = 0
            last_video_id = row['video_id']
            if row['event'] == 'start':
                interval = row['time'] - tp
                clip_num = interval // 2
                while clip_num > 0:
                    row_background = {
                        'time': tp + clip_num * 2 - 1,
                        'index': background_num,
                        'video_id': row['video_id'],
                        'event_attributes': None
                    }
                    event_2_clip(row_background,
                                 out_path=os.path.join(save_path, 'frames'),
                                 before=1,
                                 after=1,
                                 sample_every=1,
                                 save=True)
                    idx = list(range(0, 51, 5))
                    idx.append(row_background['index'])
                    X.append(np.array(idx))
                    y.append(np.array([event_2_id['background'], np.nan]))
                    clip_num -= 1
                    background_num += 1
            else:
                tp = row['time']

    else:
        return
    X = np.vstack(X)
    y = np.vstack(y)
    print(X.shape)
    print(y.shape)
    np.save(os.path.join(save_path, 'X.npy'), X)
    np.save(os.path.join(save_path, 'y.npy'), y)


def sample_data(data_path, sample_size=(4000, 4000, 4000, 4000)):
    """
    data_path: directory of generated data
    sample_size: sample number of each class (play, throwin, challenge, background)
    """
    class_names = os.listdir(data_path)
    assert id_2_event <= class_names  # check if all classes have generated data
    X = []
    y = []
    for i, sample_num in enumerate(sample_size):
        class_path = os.path.join(data_path, id_2_event[i])
        X_class = np.load(os.path.join(class_path, 'X.npy'))
        y_class = np.load(os.path.join(class_path, 'y.npy'))
        size = X_class.shape[0]
        id = np.random.choice(np.arange(size), size=sample_num, replace=False)
        X_class = X_class[id]
        y_class = y_class[id]
        if i == 3:
            X_b = X_class
            y_b = y_class
        else:
            X.append(X_class)
            y.append(y_class)
    X = np.vstack(X)
    y = np.vstack(y)

    print(X.shape)
    print(y.shape)
    print(X_b.shape)
    print(y_b.shape)
    np.save(os.path.join(data_path, 'X.npy'), X)
    np.save(os.path.join(data_path, 'y.npy'), y)
    np.save(os.path.join(data_path, 'X_b.npy'), X_b)
    np.save(os.path.join(data_path, 'y_b.npy'), y_b)


def main():
    root_dir = '/home/trunk/zyx/SocDetect'
    for class_name in id_2_event:
        generate_data(root_dir, class_name)
    sample_data(os.path.join(root_dir, 'data'))


if __name__ == '__main__':
    main()
