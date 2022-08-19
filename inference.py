import torch
import torch.nn as nn
import os
from train import load_trained_model
from evaluate import event_detection_ap, tolerances
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from dataset.preprocess import id_2_event, clip_transform
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import trange

softmax = nn.Softmax(dim=1)

def inference(model, file_path, gpu=False, batch_size = 8, sample_every=5, clip_length=11):
    times = []
    events = []
    scores = []

    batch_num = 0
    vidcap = cv2.VideoCapture(file_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip = []
    batch = []
    for count in trange(frame_num):
        ret, image = vidcap.read()
        assert ret == True
        if not count % sample_every:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, [398, 224])
            image = Image.fromarray(image)
            image = clip_transform(image)
            clip.append(image)
            if len(clip) == clip_length:
                batch.append(torch.stack(clip))
                clip = [clip[0]]
                if len(batch) == batch_size:
                    if gpu:
                        out_cls, out_reg = model(torch.stack(batch).cuda())
                        event = torch.argmax(out_cls, dim=1).cpu().numpy().tolist()
                    else:
                        out_cls, out_reg = model(torch.stack(batch))
                        event = torch.argmax(out_cls, dim=1).numpy().tolist()
                    start_time = batch_num * batch_size * 2
                    batch_num += 1
                    for i, ev in enumerate(event):
                        if ev != 3:  # not background
                            score = softmax(out_cls)[i, ev].item()
                            time = start_time + i*2 + out_reg[i, 0].item()  * 2
                            times.append(time)
                            events.append(id_2_event[ev])
                            scores.append(score)
                    batch.clear()

    # handle last unfinished batch
    if gpu:
        out_cls, out_reg = model(torch.stack(batch).cuda())
        event = torch.argmax(out_cls, dim=1).cpu().numpy().tolist()
    else:
        out_cls, out_reg = model(torch.stack(batch))
        event = torch.argmax(out_cls, dim=1).numpy().tolist()
    start_time = batch_num * 2
    for i, ev in enumerate(event):
        if ev != 3:  # not background
            score = softmax(out_cls)[i, ev].item()
            time = start_time + i*2 + out_reg[i, 0].item()  * 2
            times.append(time)
            events.append(id_2_event[ev])
            scores.append(score)
    
    return times, events, scores

def output(out_file='submission.csv', gpu=False):
    model = load_trained_model()
    model.eval()
    if gpu:
        model.cuda()
    test_path = '/home/trunk/zyx/SocDetect/test'
    ID = []
    TIME = []
    EVENT = []
    SCORE = []
    for file in os.listdir(test_path):
        file_path = os.path.join(test_path, file)
        times, events, scores = inference(model, file_path, gpu=gpu)
        ids = [file.split('.')[0]] * len(times)
        ID.extend(ids)
        TIME.extend(times)
        EVENT.extend(events)
        SCORE.extend(scores)
    df = pd.DataFrame({
        'video_id': ID,
        'time': TIME,
        'event': EVENT,
        'score': SCORE
    }).round(decimals=2)
    df.to_csv(out_file, index=False)


def output_train(out_file='submission.csv', gpu=False):
    model = load_trained_model()
    model.eval()
    if gpu:
        model.cuda()
    file_path = '/home/trunk/zyx/SocDetect/train/1606b0e6_0.mp4'
    ID = []
    TIME = []
    EVENT = []
    SCORE = []
    times, events, scores = inference(model, file_path, gpu=gpu)
    ids = ['1606b0e6_0'] * len(times)
    ID.extend(ids)
    TIME.extend(times)
    EVENT.extend(events)
    SCORE.extend(scores)
    df = pd.DataFrame({
        'video_id': ID,
        'time': TIME,
        'event': EVENT,
        'score': SCORE
    }).round(decimals=2)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    output_file = 'submission.csv'
    output_train(output_file, gpu=False)
    solution = pd.read_csv('/home/trunk/zyx/SocDetect/train.csv',
                           usecols=['video_id', 'time', 'event'])

    dummy_submission = []
    times = np.arange(0, 60 * 90, 0.5)
    for event in ["play"]:
        df = pd.DataFrame({
            "video_id": ['1606b0e6_0'] * len(times),
            "time": times,
            "event": [event] * len(times),
            "score": [1.0] * len(times)
        })
        dummy_submission.append(df)
    dummy_submission = pd.concat(dummy_submission)

    train_submission = pd.read_csv(output_file)

    score_base = event_detection_ap(
        solution[solution['video_id'] == '1606b0e6_0'], dummy_submission,
        tolerances)
    score = event_detection_ap(solution[solution['video_id'] == '1606b0e6_0'],
                               train_submission, tolerances)
    print("baseline score:%.4f" % score_base)
    print("   model score:%.4f" % score)