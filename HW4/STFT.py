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


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # 頻譜質心
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # 頻譜下降點
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # 頻譜流
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # 過零率
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    # 聚合特徵
    features = np.hstack((np.mean(spectral_centroid), np.mean(spectral_rolloff), np.mean(spectral_flux), np.mean(zero_crossing_rate)))
    return features

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

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
# 使用 StratifiedKFold 來保持每一類的比例
kf = StratifiedKFold(n_splits=5)
accuracy_scores = []

for train_index, test_index in kf.split(features, labels_encoded):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = [labels_encoded[i] for i in train_index], [labels_encoded[i] for i in test_index]
    
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