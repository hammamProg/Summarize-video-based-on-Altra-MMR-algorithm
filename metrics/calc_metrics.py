from tools.files_tools import upload_pickle_object
from Objects import Tour20Vidoes
from collections import defaultdict


def compine_two_list_of_tuples(list1,list2):
    result = []
    for i1,i2 in zip(list1,list2):
        result.append((i1,i2))
    return result

def calc_golden_summary(area_name:Tour20Vidoes):
    
    userSummaries_output = upload_pickle_object("metrics/tour20_userSummaries_output")
    
    user1_gt        = userSummaries_output['User1'][area_name]['GT']
    user1_gt_index  = userSummaries_output['User1'][area_name]['GT-video-index']
    

    user2_gt        = userSummaries_output['User2'][area_name]['GT']
    user2_gt_index  = userSummaries_output['User2'][area_name]['GT-video-index']

    user3_gt        = userSummaries_output['User3'][area_name]['GT']
    user3_gt_index  = userSummaries_output['User3'][area_name]['GT-video-index']
    
    user1_summary = compine_two_list_of_tuples(user1_gt_index,user1_gt)
    user2_summary = compine_two_list_of_tuples(user2_gt_index,user2_gt)  
    user3_summary = compine_two_list_of_tuples(user3_gt_index,user3_gt)

    y_true = list(set(user1_summary + user2_summary + user3_summary))
    return y_true,user1_summary,user2_summary,user3_summary

def calc_precision_recall_f1_accuracy(area_name, choosen_shots, test_name):    
    y_true,user1_summary,user2_summary,user3_summary = calc_golden_summary(area_name)
    
    y_pred = choosen_shots
    print(f"Top 22 shots :\n{y_pred[:len(user1_summary)]}\n")

    precision,recall,f1,accuracy = evaluate_predictions(y_pred,y_true)

    precision_user1,recall_user1,f1_user1,accuracy_user1 = evaluate_predictions(y_pred[:len(user1_summary)],user1_summary)
    precision_user2,recall_user2,f1_user2,accuracy_user2 = evaluate_predictions(y_pred[:len(user2_summary)],user2_summary)
    precision_user3,recall_user3,f1_user3,accuracy_user3 = evaluate_predictions(y_pred[:len(user3_summary)],user3_summary)

    print(f"user1_summary :\n{user1_summary}\n")
    print(f"user2_summary :\n{user2_summary}\n")
    print(f"user3_summary :\n{user3_summary}\n")

    print(f'|| Golden Summary({test_name})\n|| \t\t\t\t{len(choosen_shots)}\t\t{precision*100:.2f}\t\t{recall*100:.2f}\t\t{f1*100:.2f}\t\t{accuracy*100:.2f}')
    print(f'|| \t\t\t\t_____________________________________________________________________')
    print( '|| ')
    print(f'|| user1\t\t\t{len(user1_summary)}\t\t{precision_user1*100:.2f}\t\t{recall_user1*100:.2f}\t\t{f1_user1*100:.2f}\t\t{accuracy_user1*100:.2f}')
    print(f'|| user2\t\t\t{len(user2_summary)}\t\t{precision_user2*100:.2f}\t\t{recall_user2*100:.2f}\t\t{f1_user2*100:.2f}\t\t{accuracy_user2*100:.2f}')
    print(f'|| user3\t\t\t{len(user3_summary)}\t\t{precision_user3*100:.2f}\t\t{recall_user3*100:.2f}\t\t{f1_user3*100:.2f}\t\t{accuracy_user3*100:.2f}')
    print(f'|| f1 average = {(f1_user1+f1_user2+f1_user3)/3*100:.2f}')

def evaluate_predictions(y_prediction, y_true):
    tp = defaultdict(int)  # True Positives
    fp = defaultdict(int)  # False Positives
    fn = defaultdict(int)  # False Negatives

    # Convert y_true to a dictionary for easier lookup
    true_dict = defaultdict(int)
    for cls, count in y_true:
        true_dict[cls] += count

    # Calculate TP, FP, and FN
    for cls, count in y_prediction:
        if cls in true_dict:
            # Calculate TP and adjust FP if the predicted count exceeds the true count
            true_count = true_dict[cls]
            tp[cls] += min(count, true_count)
            if count > true_count:
                fp[cls] += count - true_count  # Ensure FP is not negative
            true_dict[cls] = max(0, true_count - count)  # Adjust true count, ensure it's not negative
        else:
            fp[cls] += count  # All predicted counts for classes not in true values are FP

    # Remaining counts in true_dict are FN
    for cls, count in true_dict.items():
        fn[cls] += count  # Ensure FN is not negative

    # Calculate precision, recall, and F1 score
    precision = sum(tp.values()) / (sum(tp.values()) + sum(fp.values())) if sum(tp.values()) + sum(fp.values()) > 0 else 0
    recall = sum(tp.values()) / (sum(tp.values()) + sum(fn.values())) if sum(tp.values()) + sum(fn.values()) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calculate an alternative form of accuracy
    total_counts = sum(tp.values()) + sum(fp.values()) + sum(fn.values())
    accuracy_alternative = sum(tp.values()) / total_counts if total_counts > 0 else 0

    return precision, recall, f1_score, accuracy_alternative
