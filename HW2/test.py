def calculate_precision_recall(actual, predicted):
    TP = 0
    FP = 0
    FN = 0
    
    for predicted_val in predicted:
        for actual_val in actual:
            if isinstance(actual_val, list):  # 如果 a 中的元素是列表（區間）
                if predicted_val in actual_val:  # 檢查 b_val 是否在區間內
                    TP += 1
                    break
            elif predicted_val == actual_val:  # 如果 a 中的元素是數值，直接比較
                TP += 1
                break

    FP = len(predicted) - TP
    FN = len(actual) - TP
            
            
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision, recall

# 假設列表
# 1st預測列表，包含數值和區間
# 2nd真實列表，只包含數值
a = [range(455, 479)]
b = [range(542, 579)]
c = [range(608, 645)]
d = [range(675, 697)]
e = [range(774, 799)]
f = [range(886, 887)]
predicted = [16, 93, 157, 355, 1237, 1401, 1555]
actual = [93, 157, 232, 314, 355, a, b, c, d, e, f, 1021, 1237, 1401, 1555]
precision, recall = calculate_precision_recall(actual, predicted)
print(f"Precision: {precision}")
print(f"Recall: {recall}")