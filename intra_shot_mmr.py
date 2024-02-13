from numpy import prod
from scipy.spatial.distance import cosine
from Objects import Frame,Video
from scipy.stats import gmean
from tools.wrappers_funcs import timeit
import numpy as np
# from numba import njit
from multiprocessing import Pool

def calculate_similarities(frame_feature, frames_features, similarity_type='AM') -> float:
    '''
        Calculate the similarity between a frame and a list of frames using the cosine similarity

        Args:
            frame_feature: np.array - the feature vector of the frame
            frames_features: np.array - the feature vectors of the frames to compare with
            similarity_type: str - the type of similarity to calculate (AM or GM)

        Returns:
            float - the similarity between the frame and the list of frames

    '''
    similarities = np.zeros(len(frames_features))
    for i, g_frame in enumerate(frames_features):
        similarities[i] = cosine(frame_feature, g_frame)
    
    if similarity_type == 'AM':
        return np.average(similarities)
    elif similarity_type == 'GM':
        return gmean(similarities) 

def sim2(frame_feature, frames_features) -> np.array:
    '''
        Calculate the cosine similarity between a frame and a list of frames using the cosine similarity

        Args:
            frame_feature: np.array - the feature vector of the frame
            frames_features: np.array - the feature vectors of the frames to compare with
        
        Returns:
            out_sim2: np.array - the similarity between the frame and each frame in the list
    '''
    out_sim2 = np.zeros(len(frames_features))
    for i, g_frame in enumerate(frames_features):
        out_sim2[i] = cosine(frame_feature, g_frame)
    return out_sim2


def calculate_shot_mmr(video_index, shot_index, videos:list[Video], all_frames:list[Frame], lmda) -> tuple[int, int, float]:
    '''
        Calculate the MMR for a single shot

        Args:
            video_index: int - the index of the video in the videos list
            shot_index: int - the index of the shot in the video.shots list
            videos: list[Video] - list of Video objects
            all_frames: np.array[Frame] - array of all Frame objects
            lmda: float - the lambda parameter for MMR
        
        Returns:
            video_index: int - the index of the video in the videos list
            shot_index: int - the index of the shot in the video.shots list
            np.sum(shot_out): float - the MMR for the shot

        Change the similarity type to 'GM' if you want to use the geometric mean instead of the arithmetic mean
        
    '''
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
        similarity = calculate_similarities(frame.features, other_shot_frames, 'AM')
        max_sim2 = np.max(sim2(frame.features, current_shot_frames))
        # shot_out[i] = lmda * similarity - (1 - lmda) * max_sim2
        shot_out[i] = similarity

    return video_index, shot_index, np.sum(shot_out)

@timeit
def calculate_video_mmr(videos: list[Video], lmda) -> None:
    '''
        Calculate the MMR for each shot in each video using multiprocessing
        
        Args:
            videos: list[Video] - list of Video objects
            lmda: float - the lambda parameter for MMR

        Returns:    
            None (updates the mmr_sum attribute of each shot in each video object in the videos list)


        You can change the number of processes in the Pool to match the number of cores in your machine
        Pool(processes=6) - for a 6-core machine
        Pool(processes=12) - for a 12-core machine 
        and so on
    '''
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





