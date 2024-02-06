from Objects import Frame, Shot, Video, Tour20Vidoes, FeaturesType, AverageSimilarityMeasure, ShotValueRepresentation
from tools.files_tools import download_object_as_pickle, upload_pickle_object
from metrics.calc_metrics import calc_precision_recall_f1_accuracy, calc_golden_summary
from video_mmr import video_mmr
from cv_tools import extract_histogram_features, frames_to_video
from kaggle import extract_color_videos_frames_features_final

import numpy as np

from blessed import Terminal



def fill_classes_with_area_videos(area_videos):
    video_objects:list[Video] = []

    for video_name, video_shots_frames_features in area_videos:
        shots_list:list[Shot] = []

        for shot_index, frames_features_vector_by_cnn_for_shot in video_shots_frames_features:
            frames_list:list[Frame] = []

            for frame_index, (frame, features) in enumerate(frames_features_vector_by_cnn_for_shot):
                # you can change feature[0] with your cnn features
                frames_list.append(Frame(video_name[2], shot_index, frame_index, np.array(extract_histogram_features(frame)), frame))

            shots_list.append(Shot(video_name[2], shot_index, np.array(frames_list)))

        video_objects.append(Video(video_name[2], video_name, np.array(shots_list)))

    return np.array(video_objects)




def mmr_apprach(video_objects:list[Video], area_name ,feature_type:FeaturesType , average_sim_measure:AverageSimilarityMeasure, shot_value_representation:ShotValueRepresentation) -> None:
    # video_shot_mmr = []
    # for video in video_objects:
    #     for shot in video.shots:
    #         if average_sim_measure == AverageSimilarityMeasure.AM.value:
    #             if shot_value_representation == ShotValueRepresentation.AVERAGE.value :
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_average_value()))
    #             elif shot_value_representation == ShotValueRepresentation.MEDIAN.value:
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_median_value()))
    #             elif shot_value_representation == ShotValueRepresentation.STDEV.value:
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_stdev()))
    #             elif shot_value_representation == ShotValueRepresentation.SUM.value:
    #                 # video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_sum_value()))
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_sum))
            
    #         elif average_sim_measure == AverageSimilarityMeasure.GM.value:
    #             if shot_value_representation == ShotValueRepresentation.AVERAGE.value :
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_average_value_GM()))
    #             elif shot_value_representation == ShotValueRepresentation.MEDIAN.value:
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_median_value_GM()))
    #             elif shot_value_representation == ShotValueRepresentation.STDEV.value:
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_stdev_GM()))
    #             elif shot_value_representation == ShotValueRepresentation.SUM.value:
    #                 video_shot_mmr.append((video.num_video,shot.num_shot,shot.mmr_sum_value_GM()))

    all_shots:list[Shot] = [shot for video in video_objects for shot in video.shots]
    sorted_video_shot_mmr = sorted(all_shots, key=lambda x: x.mmr_sum, reverse=True)

    y_true,_,_,_ = calc_golden_summary(all_shots,area_name)
    
    retrival_shots_threshold = len(y_true)

    # choosen_shots = []
    # for (video_num, shot_num, _ ) in sorted_video_shot_mmr:
    #     if len(choosen_shots) < retrival_shots_threshold:
    #         choosen_shots.append((int(video_num), shot_num))
    choosen_shots:list[Shot] = sorted_video_shot_mmr[:retrival_shots_threshold]

    print("|| ===================================================================================================")
    print(f'{"|| Method":<24}\t{"Num shots":<16}{"Precision":<16}{"Recall":<16}{"F1 Score":<16}{"Accuracy":<15}')
    
    calc_precision_recall_f1_accuracy(all_shots,area_name, choosen_shots, f"MMR-{feature_type} {shot_value_representation}")



############################################################################## apply video_mmr

if __name__ == '__main__':
    train_flag = True

    all_areas_names = Tour20Vidoes.list_name()
    # all_areas_names =[Tour20Vidoes.GM.name]

    for area_name in all_areas_names:
        try:
            video_objects = None

            feature_types:list[FeaturesType] = [FeaturesType.HSV]

            if train_flag:
                for feature_type in feature_types:
                    area_videos = extract_color_videos_frames_features_final(area_name)
                    video_objects:list[Video] = fill_classes_with_area_videos(area_videos)
                    
                    lambda_value = 0.6
                    
                    print(f"\nApplying MMR(lambda = {lambda_value}) for all frames in the videos :) ......\n")

                    video_mmr(video_objects,lambda_value)
            
            # download_object_as_pickle(f'{area_name}_video_objects',video_objects)
            
            # video_objects = upload_pickle_object(f'{area_name}_video_objects')

            mmr_apprach(video_objects, area_name, FeaturesType.HSV.value, AverageSimilarityMeasure.AM.value, 'sum')
        except Exception as e:
            print(f"An error occurred in area {area_name}: {e}")
