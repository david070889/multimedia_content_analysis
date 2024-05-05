import librosa
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 保持原始採樣率
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    # return np.hstack((mfccs_mean, mfccs_std))  # 合併均值和標準差成為一個特徵向量
    return mfccs.mean(axis=1)

audio_files  = []
labels = []
for genre in genres:
    for filename in os.listdir(genre):
        if filename.endswith(".wav"):  # 確保處理 .wav 檔案
            file_path = os.path.join(genre, filename)
            audio_files.append(file_path)
            labels.append(genre)

features = np.array([extract_features(file) for file in audio_files])
print(features[0])

kf = StratifiedKFold(n_splits=5)
accuracy_scores = []

for train_index, test_index in kf.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
    
    # 創建 SVM 模型
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

    # 訓練模型
    svm_model.fit(X_train, y_train)
    
    # 預測測試集
    predictions = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

# 輸出平均準確率
average_accuracy = np.mean(accuracy_scores)
print(accuracy_scores)
print(f'Average classification accuracy with SVM: {average_accuracy}')