from tools.files_tools import upload_pickle_object
from Objects import Tour20Vidoes,Shot
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math

def compine_two_list_of_tuples(list1,list2):
    result = []
    for i1,i2 in zip(list1,list2):
        result.append((i1,i2))
    return result

def retreive_users_summaires_from_tour20_dataset(all_shots:list[Shot],area_name):

    tour20_userSummaries = upload_pickle_object("metrics/tour20_userSummaries_output")
    
    user1_gt        = tour20_userSummaries['User1'][area_name]['GT']
    user1_gt_index  = tour20_userSummaries['User1'][area_name]['GT-video-index']
    

    user2_gt        = tour20_userSummaries['User2'][area_name]['GT']
    user2_gt_index  = tour20_userSummaries['User2'][area_name]['GT-video-index']

    user3_gt        = tour20_userSummaries['User3'][area_name]['GT']
    user3_gt_index  = tour20_userSummaries['User3'][area_name]['GT-video-index']
    
    user1_summary = compine_two_list_of_tuples(user1_gt_index,user1_gt)
    user1_summary_shots:list[Shot] = []
 
    user2_summary = compine_two_list_of_tuples(user2_gt_index,user2_gt)  
    user2_summary_shots:list[Shot] = []

    user3_summary = compine_two_list_of_tuples(user3_gt_index,user3_gt)
    user3_summary_shots:list[Shot] = []


    for shot in all_shots:
        if (shot.num_video, shot.num_shot) in user1_summary:
            user1_summary_shots.append(shot)
        if (shot.num_video, shot.num_shot) in user2_summary:
            user2_summary_shots.append(shot)
        if (shot.num_video, shot.num_shot) in user3_summary:
            user3_summary_shots.append(shot)

    # num_images = len(user1_summary_shots)
    # rows = math.ceil(num_images / 5)
    # plt.figure(figsize=(25, 5 * rows))
    # for i, shot in enumerate(user1_summary_shots):
    #     if i % 5 == 0:
    #         plt.figure(figsize=(25, 5))  # Set the figure size for each row of images
    #     plt.subplot(1, 5, i % 5 + 1)  # Subplot indices start at 1, not 0
    #     mid_frame_index = len(shot.frames) // 2

    #     plt.imshow(shot.frames[mid_frame_index].frame, cmap='gray')
    #     plt.axis('off')
    #     if (i + 1) % 5 == 0 or i == num_images - 1:
    #         plt.subplots_adjust(wspace=0.2, hspace=0.2)
    #         plt.show()
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.show()

    y_true = list(set(user1_summary + user2_summary + user3_summary))
    return y_true,user1_summary_shots, user2_summary_shots, user3_summary_shots



def print_evaluation_result(all_shots, area_name, shots_pred:list[Shot], test_name): 

    y_true,user1_summary_shots, user2_summary_shots, user3_summary_shots = retreive_users_summaires_from_tour20_dataset(all_shots ,area_name)
    
    #### plot user1_summary_shots as images (plot first image in each shot)
    
    # num_images = len(shots_pred[:len(user1_summary_shots)])
    # print(f"num_images = {num_images}")
    # rows = math.ceil(num_images / 5)
    # plt.figure(figsize=(25, 5 * rows))
    # for i, shot in enumerate(shots_pred[:num_images]):
    #     if i % 5 == 0:
    #         plt.figure(figsize=(25, 5))  # Set the figure size for each row of images
    #     plt.subplot(1, 5, i % 5 + 1)  # Subplot indices start at 1, not 0
    #     # calc mid frame
    #     mid_frame_index = len(shot.frames) // 2

    #     plt.imshow(shot.frames[mid_frame_index].frame, cmap='gray')
    #     plt.axis('off')
    #     if (i + 1) % 5 == 0 or i == num_images - 1:
    #         plt.subplots_adjust(wspace=0.2, hspace=0.2)
    #         plt.show()
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.show()

    precision_user1,recall_user1,f1_user1,accuracy_user1 = evaluate_predictions(shots_pred[:len(user1_summary_shots)],user1_summary_shots)
    precision_user2,recall_user2,f1_user2,accuracy_user2 = evaluate_predictions(shots_pred[:len(user2_summary_shots)],user2_summary_shots)
    precision_user3,recall_user3,f1_user3,accuracy_user3 = evaluate_predictions(shots_pred[:len(user3_summary_shots)],user3_summary_shots)

    print(f'|| \t\t\t\t_____________________________________________________________________')
    print( '|| ')
    print(f'|| user1\t\t\t{len(user1_summary_shots)}\t\t{precision_user1*100:.2f}\t\t{recall_user1*100:.2f}\t\t{f1_user1*100:.2f}\t\t{accuracy_user1*100:.2f}')
    print(f'|| user2\t\t\t{len(user2_summary_shots)}\t\t{precision_user2*100:.2f}\t\t{recall_user2*100:.2f}\t\t{f1_user2*100:.2f}\t\t{accuracy_user2*100:.2f}')
    print(f'|| user3\t\t\t{len(user3_summary_shots)}\t\t{precision_user3*100:.2f}\t\t{recall_user3*100:.2f}\t\t{f1_user3*100:.2f}\t\t{accuracy_user3*100:.2f}')
    print(f'|| f1 average = {(f1_user1+f1_user2+f1_user3)/3*100:.2f}')

def evaluate_predictions(shots_pred:list[Shot], user_summary_shots:list[Shot]):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for shot_pred in shots_pred:
        
        shot_features = shot_pred.get_features()
        flag_found = False
        for user_shot in user_summary_shots:
            similarity = cosine_similarity(shot_features, user_shot.get_features())[0][0]
            if similarity > 0.8:
                true_positive += 1
                flag_found = True
                break
        if not flag_found:
            false_positive += 1
    
    for user_shot in user_summary_shots:
        flag_found = False
        for shot_pred in shots_pred:
            

            similarity = cosine_similarity(shot_pred.get_features(), user_shot.get_features())[0][0]
            ### plot the first frame of the user_shot vs the first frame of the shot_pred in one figure, with the similarity value as seprated label, add label for each figure
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # mid_frame_index = len(shot_pred.frames) // 2
            # plt.imshow(shot_pred.frames[mid_frame_index].frame, cmap='gray')
            # plt.title(f'Pred Similarity = {similarity:.2f}')
            # plt.axis('off')
            # plt.subplot(1, 2, 2)
            # mid_frame_index = len(user_shot.frames) // 2
            # plt.imshow(user_shot.frames[mid_frame_index].frame, cmap='gray')
            # plt.title(f'User summary')
            # plt.axis('off')
            # plt.show()

            if similarity > 0.8:
                true_positive += 1
                ### plot the taken shot and print the similarity value
                # plt.figure()
                # plt.imshow(shot_pred.frames[0].frame, cmap='gray')
                # plt.title(f'Similarity = {similarity:.2f}')
                # plt.axis('off')
                # plt.show()

                flag_found = True
                break
        if not flag_found:
            false_negative += 1

    false_negative = len(user_summary_shots) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy_alternative = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

    return precision, recall, f1_score, accuracy_alternative
