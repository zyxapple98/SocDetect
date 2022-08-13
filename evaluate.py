import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from typing import Dict, Tuple
from pathlib import Path
import glob
import os

tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def filter_detections(detections: pd.DataFrame,
                      intervals: pd.DataFrame) -> pd.DataFrame:
    """Drop detections not inside a scoring interval."""
    detection_time = detections.loc[:, 'time'].sort_values().to_numpy()
    intervals = intervals.to_numpy()
    is_scored = np.full_like(detection_time, False, dtype=bool)

    i, j = 0, 0
    while i < len(detection_time) and j < len(intervals):
        time = detection_time[i]
        int_ = intervals[j]

        # If the detection is prior in time to the interval, go to the next detection.
        if time < int_.left:
            i += 1
        # If the detection is inside the interval, keep it and go to the next detection.
        elif time in int_:
            is_scored[i] = True
            i += 1
        # If the detection is later in time, go to the next interval.
        else:
            j += 1

    return detections.loc[is_scored].reset_index(drop=True)


def match_detections(tolerance: float, ground_truths: pd.DataFrame,
                     detections: pd.DataFrame) -> pd.DataFrame:
    """Match detections to ground truth events. Arguments are taken from a common event x tolerance x video evaluation group."""
    detections_sorted = detections.sort_values('score',
                                               ascending=False).dropna()

    is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)
    gts_matched = set()
    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        best_error = tolerance
        best_gt = None

        for gt in ground_truths.itertuples(index=False):
            error = abs(det.time - gt.time)
            if error < best_error and gt not in gts_matched:
                best_gt = gt
                best_error = error

        if best_gt is not None:
            is_matched[i] = True
            gts_matched.add(best_gt)

    detections_sorted['matched'] = is_matched

    return detections_sorted


def precision_recall_curve(
        matches: np.ndarray, scores: np.ndarray,
        p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind='stable')[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / p  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray,
                            p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, float],
) -> float:

    assert_index_equal(solution.columns,
                       pd.Index(['video_id', 'time', 'event']))
    assert_index_equal(submission.columns,
                       pd.Index(['video_id', 'time', 'event', 'score']))

    # Extract scoring intervals.
    intervals = (solution.query("event in ['start', 'end']").assign(
        interval=lambda x: x.groupby(['video_id', 'event']).cumcount()).pivot(
            index='interval', columns=['video_id', 'event'],
            values='time').stack('video_id').swaplevel().sort_index().
                 loc[:, ['start', 'end']].apply(
                     lambda x: pd.Interval(*x, closed='both'), axis=1))

    # Extract ground-truth events.
    ground_truths = (
        solution.query("event not in ['start', 'end']").reset_index(drop=True))

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts('event').to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    detections_filtered = []
    for (det_group, dets), (int_group,
                            ints) in zip(detections.groupby('video_id'),
                                         intervals.groupby('video_id')):
        assert det_group == int_group
        detections_filtered.append(filter_detections(dets, ints))
    detections_filtered = pd.concat(detections_filtered, ignore_index=True)

    # Create table of event-class x tolerance x video_id values
    aggregation_keys = pd.DataFrame(
        [(ev, tol, vid) for ev in tolerances.keys() for tol in tolerances[ev]
         for vid in ground_truths['video_id'].unique()],
        columns=['event', 'tolerance', 'video_id'],
    )

    # Create match evaluation groups: event-class x tolerance x video_id
    detections_grouped = (aggregation_keys.merge(
        detections_filtered, on=['event', 'video_id'],
        how='left').groupby(['event', 'tolerance', 'video_id']))
    ground_truths_grouped = (aggregation_keys.merge(
        ground_truths, on=['event', 'video_id'],
        how='left').groupby(['event', 'tolerance', 'video_id']))

    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(dets['tolerance'].iloc[0], gts, dets))
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths['event'].unique()
    ap_table = (detections_matched.query("event in @event_classes").groupby(
        ['event', 'tolerance']).apply(lambda group: average_precision_score(
            group['matched'].to_numpy(),
            group['score'].to_numpy(),
            class_counts[group['event'].iat[0]],
        )))

    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby('event').mean().mean()

    return mean_ap


def dummy_test():
    """
    make dummy inference: predict every frame as 'play'
    baseline AP score: 0.1356
    """
    data_dir = Path('/home/trunk/zyx/SocDetect')
    solution = pd.read_csv(data_dir / 'train.csv',
                           usecols=['video_id', 'time', 'event'])
    solution.head()

    files = glob.glob("/home/trunk/zyx/SocDetect/train/*")
    train_submission = []
    for f in files:
        times = np.arange(0, 60 * 90, 0.5)
        for event in ["play"]:
            df = pd.DataFrame({
                "video_id":
                [os.path.basename(f).replace(".mp4", "")] * len(times),
                "time":
                times,
                "event": [event] * len(times),
                "score": [1.0] * len(times)
            })
            train_submission.append(df)
    train_submission = pd.concat(train_submission)
    print(train_submission.head())
    score = event_detection_ap(solution, train_submission, tolerances)
    print(score)


if __name__ == '__main__':
    dummy_test()