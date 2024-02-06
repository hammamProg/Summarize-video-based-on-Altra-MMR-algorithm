from tools.files_tools import upload_pickle_object
from Objects import Tour20Vidoes,Shot
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def compine_two_list_of_tuples(list1,list2):
    result = []
    for i1,i2 in zip(list1,list2):
        result.append((i1,i2))
    return result

def calc_golden_summary(all_shots:list[Shot],area_name):

    userSummaries_output = upload_pickle_object("metrics/tour20_userSummaries_output")
    
    user1_gt        = userSummaries_output['User1'][area_name]['GT']
    user1_gt_index  = userSummaries_output['User1'][area_name]['GT-video-index']
    

    user2_gt        = userSummaries_output['User2'][area_name]['GT']
    user2_gt_index  = userSummaries_output['User2'][area_name]['GT-video-index']

    user3_gt        = userSummaries_output['User3'][area_name]['GT']
    user3_gt_index  = userSummaries_output['User3'][area_name]['GT-video-index']
    
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


    y_true = list(set(user1_summary + user2_summary + user3_summary))
    return y_true,user1_summary_shots, user2_summary_shots, user3_summary_shots

def calc_precision_recall_f1_accuracy(all_shots, area_name, shots_pred:list[Shot], test_name): 
    print(area_name)   
    y_true,user1_summary_shots, user2_summary_shots, user3_summary_shots = calc_golden_summary(all_shots ,area_name)
    

    precision_user1,recall_user1,f1_user1,accuracy_user1 = evaluate_predictions(shots_pred[:len(user1_summary_shots)],user1_summary_shots)
    precision_user2,recall_user2,f1_user2,accuracy_user2 = evaluate_predictions(shots_pred[:len(user2_summary_shots)],user2_summary_shots)
    precision_user3,recall_user3,f1_user3,accuracy_user3 = evaluate_predictions(shots_pred[:len(user3_summary_shots)],user3_summary_shots)


    # print(f'|| Golden Summary({test_name})\n|| \t\t\t\t{len(choosen_shots)}\t\t{precision*100:.2f}\t\t{recall*100:.2f}\t\t{f1*100:.2f}\t\t{accuracy*100:.2f}')
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
        
    false_negative = len(user_summary_shots) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy_alternative = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

    return precision, recall, f1_score, accuracy_alternative
