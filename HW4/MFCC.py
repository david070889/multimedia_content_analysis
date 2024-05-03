import librosa
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 音樂檔案路徑
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


# 提取 MFCC 特徵
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)  # 取平均值作為特徵向量

# 載入數據
temp = []
data = []
labels = []
for genre in genres:
    for filename in os.listdir(genre):
        if filename.endswith(".wav"):  # 確保處理 .wav 檔案
            file_path = os.path.join(genre, filename)
            mfccs = extract_features(file_path)
            data.append(mfccs)
            labels.append(genre)

data = np.array(data)
labels = np.array(labels)

# 將標籤轉換為數字
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 5-Fold 交叉驗證
kf = StratifiedKFold(n_splits=5)
accuracys = []

for train_index, test_index in kf.split(data, labels_encoded):
    # 分割數據
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]

    # 訓練 GMM
    gmm = GaussianMixture(n_components=10, covariance_type='diag', max_iter=200, random_state=0)
    gmm.fit(X_train, y_train)

    # 預測
    y_pred = gmm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracys.append(accuracy)

# 輸出平均錯誤率
average_accuracy = np.mean(accuracys)
print(f'Average classification accuracy: {average_accuracy}')
print(accuracys)
