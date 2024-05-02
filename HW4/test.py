import librosa
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

# 音樂檔案路徑
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


# 提取 MFCC 特徵
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)  # 取平均值作為特徵向量

# 載入數據
train_data = []
test_data = []
train_labels = []
test_labels = []
train_files = []
test_files = []
for genre in genres:
    genre_files = [os.path.join(genre, f) for f in os.listdir(genre) if f.endswith(".wav")]
    random.shuffle(genre_files)  # 隨機打亂文件
    # 選取每個類別的 40 個片段作為訓練數據，剩餘的作為測試數據
    train_files += genre_files[:40]
    test_files += genre_files[40:]
    train_labels += [genre] * 40
    test_labels += [genre] * 10
    # for file_path in train_files + test_files:
    #     mfccs = extract_features(file_path)
    #     data.append(mfccs)
    #     labels.append(genre)
print(test_files)
for path in train_files:
    mfccs = extract_features(path)
    train_data.append(mfccs)

for path in test_files:
    mfccs = extract_features(path)
    test_data.append(mfccs)
    

train_data = np.array(train_data)
test_data = np.array(test_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 將標籤轉換為數字
le = LabelEncoder()
labels_encoded_train = le.fit_transform(train_labels)
labels_encoded_test = le.fit_transform(test_labels)

# 手動分割訓練數據和測試數據
X_train = train_data
X_test = test_data
y_train = labels_encoded_train
y_test = labels_encoded_test

# 訓練 GMM
gmm = GaussianMixture(n_components=len(genres), covariance_type='diag', max_iter=200, random_state=0)
gmm.fit(X_train, y_train)

# 預測
y_pred = gmm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 輸出錯誤率
print(f'Classification accuracy: {accuracy}')