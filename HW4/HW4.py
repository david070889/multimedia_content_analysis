import librosa
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def extract_features(file_path, feature):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # 頻譜質心
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # 頻譜下降點
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # 頻譜流
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # 過零率
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    if feature == 'MFCC':
        return np.hstack((mfccs_mean, mfccs_std))
    elif feature == 'STFT':
        return np.hstack((np.mean(spectral_centroid), np.mean(spectral_rolloff), np.mean(spectral_flux), np.mean(zero_crossing_rate)))
    elif feature == 'combined':
        return np.hstack((mfccs_mean, mfccs_std, np.mean(spectral_centroid), np.mean(spectral_rolloff), np.mean(spectral_flux), np.mean(zero_crossing_rate)))

def classification(features, labels, method : str):
    # 將標籤轉換為數字
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

# 5-Fold 交叉驗證
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracys = []

    for train_index, test_index in kf.split(features, labels_encoded):
        # 分割數據
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]

        # 訓練 GMM
        if method == 'GMM':
            gmm = GaussianMixture(n_components=5, covariance_type='diag', max_iter=200, random_state=0)
            gmm.fit(X_train, y_train)
            y_pred = gmm.predict(X_test)
        elif method == 'KNN':
            knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)) # 3 or 5
            knn_model.fit(X_train, y_train)
            y_pred = knn_model.predict(X_test)
        elif method == 'SVM':
            svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf')) # linear or rbf
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)

        # 預測
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracys.append(accuracy)
    average_accuracy = np.mean(accuracys)
    print(accuracys)
    print(f'Average classification accuracy with {method}: {average_accuracy}')

if __name__ == "__main__":
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    audio_files = []
    labels = []
    for genre in genres:
        for filename in os.listdir(genre):
            if filename.endswith(".wav"):  # 確保處理 .wav 檔案
                file_path = os.path.join(genre, filename)
                audio_files.append(file_path)
                labels.append(genre)
    
    features = np.array([extract_features(file, 'combined') for file in audio_files]) # function可替換 'MFCC', 'STFT', 'combined'
    classification(features, labels, 'GMM') # method 可替換 'SVM', 'KNN', 'GMM'