from Objects import Frame, Shot, Video, Tour20Vidoes, FeaturesType, AverageSimilarityMeasure, ShotValueRepresentation
from tools.files_tools import download_object_as_pickle, upload_pickle_object
from metrics.calc_metrics import print_evaluation_result, retreive_users_summaires_from_tour20_dataset
from enhanced_mmr import calculate_video_mmr
from cv_tools import extract_histogram_features, frames_to_video
from kaggle import extract_color_videos_frames_features_final
import numpy as np

# rename the function 
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

############################################################################## apply video_mmr

if __name__ == '__main__':
    train_flag = True

    all_areas_names = Tour20Vidoes.list_name()
    # all_areas_names =all_areas_names[2:]
    # all_areas_names =[Tour20Vidoes.AT.name]

    for area_name in all_areas_names:
        try:
            video_objects = None

            feature_types:list[FeaturesType] = [FeaturesType.HSV]

            if train_flag:
                for feature_type in feature_types:
                    area_videos = extract_color_videos_frames_features_final(area_name)
                    video_objects:list[Video] = fill_classes_with_area_videos(area_videos)
                    
                    lambda_value = 0.4
                    
                    print(f"\nApplying MMR(lambda = {lambda_value}) for all frames in the videos :) ......\n")

                    calculate_video_mmr(video_objects,lambda_value)
            
            # download_object_as_pickle(f'{area_name}_video_objects',video_objects)
            # video_objects = upload_pickle_object(f'{area_name}_video_objects')
            

            feature_type = FeaturesType.HSV.value
            average_sim_measure = AverageSimilarityMeasure.AM.value
            shot_value_representation = 'sum'

            all_shots:list[Shot] = [shot for video in video_objects for shot in video.shots]
            sorted_video_shot_mmr = sorted(all_shots, key=lambda x: x.mmr_sum, reverse=True)

            y_true,_,_,_ = retreive_users_summaires_from_tour20_dataset(all_shots,area_name)
            
            retrival_threshold = len(y_true)

            choosen_shots:list[Shot] = sorted_video_shot_mmr[:retrival_threshold]

            print("|| ===================================================================================================")
            print(f'{"|| Method":<24}\t{"Num shots":<16}{"Precision":<16}{"Recall":<16}{"F1 Score":<16}{"Accuracy":<15}')
            
            print_evaluation_result(all_shots,area_name, choosen_shots, f"MMR-{feature_type} {shot_value_representation}")

        except Exception as e:
            print(f"An error occurred in area {area_name}: {e}")
