import librosa
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 保持原始採樣率
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    return np.hstack((mfccs_mean, mfccs_std))  # 合併均值和標準差成為一個特徵向量
    # return mfccs_mean

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
# 使用 StratifiedKFold 來保持每一類的比例
kf = StratifiedKFold(n_splits=5)
accuracy_scores = []

for train_index, test_index in kf.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
    
    # 創建 KNN 模型，這裡使用 k=3
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    
    # 訓練模型
    knn_model.fit(X_train, y_train)
    
    # 預測和計算準確率
    predictions = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

# 輸出平均準確率
average_accuracy = np.mean(accuracy_scores)
print(accuracy_scores)
print(f'Average classification accuracy: {average_accuracy}')
