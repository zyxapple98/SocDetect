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

softmax = nn.Softmax(dim=1)


def video_duration(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration
    return -1


def spot_2_clip(file_path, spot, before=1, after=1, sample_every=5):
    filename = "temp.mp4"
    ffmpeg_extract_subclip(file_path,
                           spot - before,
                           spot + after,
                           targetname=filename)
    vidcap = cv2.VideoCapture("temp.mp4")
    clip = []

    success, image = vidcap.read()
    count = 0
    while success:
        if not count % sample_every:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, [398, 224])
            image = Image.fromarray(image)
            image = clip_transform(image)
            clip.append(image)
        success, image = vidcap.read()

        count += 1
    return torch.stack(clip)


def inference(model, file_path, threshold=0.6):
    duration = video_duration(file_path)
    assert duration >= 0
    times = []
    events = []
    scores = []
    clip_spots = list(range(1, int(duration) - 1))
    for spot in clip_spots:
        clip = spot_2_clip(file_path, spot).unsqueeze(0).cuda()
        assert clip.shape[1] == 11
        out_cls, out_reg = model(clip)
        event = torch.argmax(out_cls, dim=1).item()
        score = softmax(out_cls)[0, event].item()
        time = spot - 1 + out_reg.squeeze().item() * 2
        if score > threshold:
            times.append(time)
            events.append(id_2_event[event])
            scores.append(score)
    return times, events, scores


def output(out_file='submission.csv'):
    model = load_trained_model()
    model.eval()
    model.cuda()
    test_path = '/home/trunk/zyx/SocDetect/train'
    ID = []
    TIME = []
    EVENT = []
    SCORE = []
    for file in os.listdir(test_path):
        file_path = os.path.join(test_path, file)
        times, events, scores = inference(model, file_path)
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


def output_train(out_file='submission.csv'):
    model = load_trained_model()
    model.eval()
    model.cuda()
    file_path = '/home/trunk/zyx/SocDetect/train/1606b0e6_0.mp4'
    ID = []
    TIME = []
    EVENT = []
    SCORE = []
    times, events, scores = inference(model, file_path)
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
    # output_train(output_file)
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