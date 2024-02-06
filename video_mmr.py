from numpy import prod
from scipy.spatial.distance import cosine
from Objects import Frame, Shot, Video
from scipy.stats import gmean
from tools.wrappers_funcs import timeit
import numpy as np
# from numba import njit
from multiprocessing import Pool
from functools import partial

def calculate_similarities(frame_feature, frames_features, similarity_type='AM'):
    similarities = np.zeros(len(frames_features))
    for i, g_frame in enumerate(frames_features):
        similarities[i] = cosine(frame_feature, g_frame)
    
    if similarity_type == 'AM':
        return np.average(similarities)
    elif similarity_type == 'GM':
        return gmean(similarities) 

def sim2(frame_feature, frames_features):
    out_sim2 = np.zeros(len(frames_features))
    for i, g_frame in enumerate(frames_features):
        out_sim2[i] = cosine(frame_feature, g_frame)
    return out_sim2


def calculate_shot_mmr(video_index, shot_index, videos, all_frames, lmda):
    current_shot_frames = np.array([
        frame.features for frame in all_frames
        if frame.num_video == videos[video_index].num_video and frame.num_shot == videos[video_index].shots[shot_index].num_shot
    ])
    other_shot_frames = np.array([
        frame.features for frame in all_frames
        if frame.num_video != videos[video_index].num_video or frame.num_shot != videos[video_index].shots[shot_index].num_shot
    ])

    shot_out = np.zeros(len(videos[video_index].shots[shot_index].frames))

    for i, frame in enumerate(videos[video_index].shots[shot_index].frames):
        am_similarity = calculate_similarities(frame.features, other_shot_frames, 'AM')
        max_sim2 = np.max(sim2(frame.features, current_shot_frames))
        shot_out[i] = lmda * am_similarity - (1 - lmda) * max_sim2

    return video_index, shot_index, np.sum(shot_out)

@timeit
def video_mmr(videos: list[Video], lmda):
    all_frames = np.array([frame for video in videos for shot in video.shots for frame in shot.frames])

    # Prepare a list of tasks for multiprocessing
    tasks = [(video_index, shot_index, videos, all_frames, lmda)
             for video_index, video in enumerate(videos)
             for shot_index, _ in enumerate(video.shots)]

    # Use multiprocessing Pool to parallelize the tasks
    with Pool(processes=6) as pool:  # Adjust the number of processes as needed
        results = pool.starmap(calculate_shot_mmr, tasks)

    # Update the mmr_sum for each shot with the results from multiprocessing
    for video_index, shot_index, mmr_sum in results:
        videos[video_index].shots[shot_index].mmr_sum = mmr_sum
