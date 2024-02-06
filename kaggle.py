import os
import pandas as pd
import numpy as np

from cv_tools import video_to_frames

########################______ Tour20 Dataset ______########################
def read_files_from_tour20_dataset(tour20_videos_directory, tour20_videos_output):
    """
        take the dataset directory, then loop in each folder within it, then extrat each video data(video_name and video_directory_path) 
        then append it to the videos_in_file_dir variable.
        at the assign this folder videos data into the tour20_videos_output file
    """
    for file in os.listdir(tour20_videos_directory):
        file_dir=f'{tour20_videos_directory}/{file}' 
        videos_in_file_dir=[]
        for video in os.listdir(file_dir):
            # remove the last four chars of video_name {.mp4} to get the name of the video
            video_name=video[:-4]
            video_dir=f'{tour20_videos_directory}/{file}/{video}'
            video_info={ 
                         'video_name': video
                        ,'video_path':video_dir
                        }
            videos_in_file_dir.append(video_info)
        tour20_videos_output[f'{file}']=videos_in_file_dir
        
def read_segmentation_tour20_dataset(tour20_segmentation_directory, tour20_segmentation_output):
    for file in os.listdir(tour20_segmentation_directory):
        file_dir=f'{tour20_segmentation_directory}/{file}' 
        
        # all files pairs AT1 - AT6 
        file_pairs = {}
        
        for segmentation_file in os.listdir(file_dir):
            
            seg_dir=f'{file_dir}/{segmentation_file}'
            
            # Open the file for reading
            with open(f'{seg_dir}', 'r') as file:
                # Read all lines from the file
                lines = file.readlines()

            # Initialize an empty list to store the pairs
            pairs = []

            # Iterate over the lines in steps of 2
            for i in range(0, len(lines), 2):
                # Convert the lines to integers and create a pair
                start_frame = int(lines[i].strip())
                end_frame = int(lines[i + 1].strip())
                pair = [start_frame, end_frame]

                # Append the pair to the list
                pairs.append(pair)
            
            file_pairs[f'{segmentation_file[-3:]}']=pairs
        tour20_segmentation_output[f'{segmentation_file[-3:-1]}']=file_pairs


def read_userSummaries_tour20_dataset(tour20_segmentation_directory, tour20_segmentation_output):
    for user_dir in os.listdir(tour20_segmentation_directory):
        user_dir_path = os.path.join(tour20_segmentation_directory, user_dir)

        file_pairs = {}

        for excel_file in os.listdir(user_dir_path):
            if excel_file.endswith(".xlsx"):
                excel_file_path = os.path.join(user_dir_path, excel_file)

                # Read the Excel file
                df = pd.read_excel(excel_file_path, sheet_name="Sheet1")
                area_columns = {}
                
                # Iterate over columns in the DataFrame
                previous=''
                for column in df.columns:
                    # Skip the 'Unnamed' columns if any
                    if previous=='GT':
                        # Save the header of the column
                        column_header = 'GT-video-index'
                        # Convert non-NaN and non-empty values in the column to integers
                        column_values = [int(val) for val in df[column].dropna() if pd.notna(val) and val != '  ' and val !=' ']
                        area_columns[column_header] = column_values
                    if 'Unnamed' not in column:
                        # Save the header of the column
                        column_header = column
                        # Convert non-NaN and non-empty values in the column to integers
                        column_values = [int(val) for val in df[column].dropna() if pd.notna(val) and val != '  ' and val !=' ']
                        
                        area_columns[column_header] = column_values
                    
                    previous=column
            file_pairs[excel_file[0:2]]=area_columns
        tour20_segmentation_output[user_dir] = file_pairs


def upload_all_tour20_files():
    tour20_videos_directory = 'Tour20/Tour20-Videos/Tour20-Videos'
    tour20_videos_output={}
    read_files_from_tour20_dataset(tour20_videos_directory, tour20_videos_output)
    print("..... Done! (read videos)")

    tour20_segmentation_directory = 'Tour20/Tour20-Segmentation/Tour20-Segmentation'
    tour20_segments_output={}
    read_segmentation_tour20_dataset(tour20_segmentation_directory, tour20_segments_output)
    print("..... Done! (read segmentations)")

    tour20_userSummaries_directory = 'Tour20/Tour20-UserSummaries/Tour20-UserSummaries'
    tour20_userSummaries_output={}
    read_userSummaries_tour20_dataset(tour20_userSummaries_directory, tour20_userSummaries_output)
    print("..... Done! (read userSummaries)")
    
    return tour20_videos_output,tour20_segments_output,tour20_userSummaries_output








def generate_golden_summary_for_all_users_areas(user_summaries, gt_of_all_users_index, shot_repeat_threshold=1):
    golden_summary = {}
    # Iterate over users
    for (user, user_pairs),(user_i,user_pairs_i) in zip(user_summaries.items(),gt_of_all_users_index.items()):
        for (area, pairs),(area_i,pairs_i) in zip(user_pairs.items(),user_pairs_i.items()):
            # Merge pairs into golden summary
            if area not in golden_summary:
                golden_summary[area] = (pairs.copy(),pairs_i.copy())  # Use copy to create a new list
            else:
                golden_summary[area] += (pairs,pairs_i)  # Use + to concatenate without modifying the original list
    
    final_golden_summary = {}
    
    for area,summary in golden_summary.items():
        number_counts = {}

        # Count the occurrences of each number
        for shot_num,video_index in zip(summary[0],summary[1]):
            number_counts[(shot_num,video_index)] = (number_counts.get((shot_num,video_index), 0) + 1)

        # Select numbers that are repeated two or more times
        repeated_numbers = [(shot_num,video_index) for (shot_num,video_index), count in number_counts.items() if count >= shot_repeat_threshold]
        final_golden_summary[area]=repeated_numbers
    return final_golden_summary

def generate_golden_summary_for_all_users_areas_videos(user_summaries,golden_summary_limit=1):    
    # loop into each user video summary and combine them into single list 
    # -> to calc golden summary for each video
    combined_data_by_video = {}
    for user, user_pairs in user_summaries.items():
        for area, area_pairs in user_pairs.items():
            if area not in combined_data_by_video:
                combined_data_by_video[area] = {}
            for video, video_pairs in area_pairs.items():
                # Combine the data into a single list
                if video not in combined_data_by_video[area]:
                    combined_data_by_video[area][video] = []
                combined_data_by_video[area][video].extend(video_pairs)
    
    # -> now calc the golden summary for each video area - if the list has mension the segment more than one time add it to the golden summary of this video
    golden_summary = {}
    for area,area_pairs in combined_data_by_video.items():
        if area not in golden_summary:
                golden_summary[area] = {}
        for video,video_pairs in area_pairs.items():
            number_counts = {}

            # Count the occurrences of each number
            for num in video_pairs:
                number_counts[num] = number_counts.get(num, 0) + 1

            # Select numbers that are repeated two or more times
            repeated_numbers = [num for num, count in number_counts.items() if count >= golden_summary_limit]
            golden_summary[area][video]=repeated_numbers
    
    # test(remove it after finish) : return combined_data_by_video
    return golden_summary, combined_data_by_video


######## For UserSummaries browsing 
def extract_golden_summary_from_dataset_userSummaries_output(tour20_userSummaries_output):
    golden_summary_for_each_video_in_area={}
    gt_of_all_users={}
    gt_index_video_all_users={}

    for user, pairs in tour20_userSummaries_output.items():
        user_GTs = gt_of_all_users.get(user, {})  # Retrieve existing user1_GTs or initialize an empty dictionary
        user_GTs_video_index={}
        user_videos_in_area={}
        for area, area_pairs in pairs.items():
            videos_in_area={}
            for video, video_pairs in area_pairs.items():
                if video == 'GT' :
                    user_GTs[area] = video_pairs
                elif video=='GT-video-index':
                    user_GTs_video_index[area] = video_pairs
                else:
                    videos_in_area[video]=video_pairs
            user_videos_in_area[area]=videos_in_area
        gt_of_all_users[user]=user_GTs
        gt_index_video_all_users[user]=user_GTs_video_index
        golden_summary_for_each_video_in_area[user]=user_videos_in_area
    
    ################################################################################################################
    ############################################# Golden Summary Guide #############################################
#     print(gt_index_video_all_users['User1']['AT'])
#     print(gt_of_all_users['User1']['AT'])
    final_area_golden_summary   = generate_golden_summary_for_all_users_areas(gt_of_all_users,gt_index_video_all_users)    
    
    at_videos_golden_summary,combined_data_by_video    = generate_golden_summary_for_all_users_areas_videos(golden_summary_for_each_video_in_area)

    return final_area_golden_summary, at_videos_golden_summary, combined_data_by_video



################################################### Return frame_features for single video
def extract_shots_from_frames_with_exist_segmentation(video_segment, video_frames):
    shots = []
    for segment in video_segment:
        start_frame, end_frame = segment
        # Ensure the frames are within the valid range
        start_frame = max(0, min(start_frame, len(video_frames) - 1))
        end_frame = max(0, min(end_frame, len(video_frames)))
        # Extract frames for the current shot
        shot_frames = video_frames[start_frame:end_frame]
        shots.append(shot_frames)
    return shots

def summarize_video_by_golden_summary_color(video_path,video_segmentation,video_golden_summary):
    video_size = (224, 224)
    video_frames = video_to_frames(video_path, video_size)
    extracted_shots = extract_shots_from_frames_with_exist_segmentation(video_segmentation, video_frames)
    shots_choosed_with_golden_summary=[]
    for golden_shot_index in video_golden_summary:
        shots_choosed_with_golden_summary.append((golden_shot_index,extracted_shots[golden_shot_index-1]))
    frames_features_vector_for_extracted_shots=[]
    for shot_index,shot_frames in shots_choosed_with_golden_summary:
        frame_feature_color = [(frame,None)for frame in shot_frames]
        frames_features_vector_for_extracted_shots.append((shot_index,frame_feature_color))    
    return frames_features_vector_for_extracted_shots

def extract_color_videos_frames_features_final(area_name):
    tour20_videos_output,tour20_segments_output,tour20_userSummaries_output = upload_all_tour20_files()
    final_area_golden_summary, at_videos_golden_summary, combined_data_by_video = extract_golden_summary_from_dataset_userSummaries_output(tour20_userSummaries_output)

    print(f"Extract segmented videos from dataset of area {area_name}")
    area_videos = []
    for video in tour20_videos_output[area_name]:
        # print()
        # print(video['video_name'][0:3])
        # print()

        video_path           = video['video_path']
        video_segmentation   = tour20_segments_output[area_name][video['video_name'][0:3]]
        
        video_golden_summary = at_videos_golden_summary[area_name][video['video_name'][0:3]]  
        video_shots_frames_features = summarize_video_by_golden_summary_color(video_path,video_segmentation,video_golden_summary)
        area_videos.append((video['video_name'][0:3],video_shots_frames_features))
    return area_videos