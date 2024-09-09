import os
import math
import numpy as np
import matplotlib.pyplot as plt
from knapsack import knapsack_ortools
from scipy.stats import rankdata, kendalltau, spearmanr

def get_rc_func(metric):
    if metric == 'kendalltau':
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == 'spearmanr':
        f = lambda x, y: spearmanr(rankdata(-x), rankdata(-y))
    else:
        raise RuntimeError
    return f

def get_correlation(metric,user_anno,pred_x):
    f = get_rc_func(metric)
    R = [f(x,pred_x)[0] for x in user_anno]

    return float(np.mean(R))

def f1_score(pred: np.ndarray, test: np.ndarray) -> float:
    assert pred.shape == test.shape
    pred = np.asarray(pred, dtype=np.bool)
    test = np.asarray(test, dtype=np.bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2*precision * recall / (precision+recall)
    return float(f1)


def downsample_summ(summ: np.ndarray) -> np.ndarray:
    return summ[::15]

def get_all_frame_score(pred: np.ndarray,
                        n_frames: int,
                        picks: np.ndarray) -> np.ndarray:
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    for i in range(len(picks) - 1):
        pos_left, pos_right = picks[i], picks[i+1]
        if i == len(pred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = pred[i]
    return frame_scores

def generate_summary(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15) -> np.ndarray:
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if picks.dtype != int:
        picks = picks.astype(np.int32)
    if picks[-1] != n_frames:
        picks = np.concatenate([picks, [n_frames]])
    for i in range(len(picks) - 1):
        pos_left, pos_right = picks[i], picks[i+1]
        if i == len(pred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = pred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    picks = knapsack_ortools(seg_score, nfps, n_segs, limits)

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary

def get_summ_f1score(video_name,pred_summ: np.ndarray, test_summ: np.ndarray, eval_metric: str = 'max') -> float:
    pred_summ = np.asarray(pred_summ, dtype=np.bool)
    test_summ = np.asarray(test_summ, dtype=np.bool)
    _, n_frames = test_summ.shape

    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))

    f1s = [f1_score(user_summ, pred_summ) for user_summ in test_summ]
    
    if eval_metric == 'avg':
        final_f1 = np.mean(f1s)
    elif eval_metric == 'max':
        final_f1 = np.max(f1s)
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')
    
    return float(final_f1)