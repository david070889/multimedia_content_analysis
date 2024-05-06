# HW4 : music genre classification
## 環境
作業系統 : Win 11  
語言 : Python 3.11.8  
IDE : VScode  
套件 : librosa, scikit-learn, numPy, os  
## 特徵擷取
MFCC(13-dim)
STFT : spectral_centroid, spectral_rolloff, spectral_flux, zero_crossing_rate  
Combined(結合兩者)
### MFCC
```python=
import librosa
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```
### STFT
```python=
import librosa
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
```
## 方法
先擷取MFCC與STFT的這些features，再分別帶入GMM, SVM, KNN這幾個model中。  
### GMM
選用講義上表現比較好的GMM(5)。  
```python=
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5, covariance_type='diag', max_iter=200, random_state=0)
```

### KNN
分別選用n=3, n=5，觀察是否與講義上表現相同。  
```python=
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 # 3 or 5
knn_model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
```
### SVM
嘗試新的模型觀察是否能有更好的accuracy，並測試線性與非線性的表現是否有差異。
```python=
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# linear or rbf
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear')) 
```
### 5 fold cross validation
使用5-fold cross validation去檢驗accuracy，並在開使分割資料前先shuffle，打亂資料。  
```python=
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(features, labels_encoded):
    # 分割數據
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]
```
## 結果與討論
|feature |MFCC|STFT|Combined|
|-----|--------|---|---|
|GMM (n=5)|0.12|0.11|0.18|
|KNN (n=3)|0.66|0.46|0.67|
|KNN (n=5)|0.69|0.48|0.68|
|SVM ('linear')|0.68|0.49|0.72|
|SVM ('rbf')|0.74|0.52|0.74|  

在音樂類型分類的研究中，從特徵表現角度來看，短時傅立葉變換（STFT）的效果通常是最差的，其次是梅爾頻率倒譜係數（MFCC），而結合這兩種特徵的方法通常表現最好。然而，MFCC的表現與結合特徵（combined features）之間的差距很小，甚至在某些情況下，MFCC的效果略優於結合特徵，這表明MFCC 是一種非常重要且有效的特徵。

在使用高斯混合模型（GMM）時，效果未達預期，這可能是由於在特徵處理或模型參數設置方面存在差異，或是因為使用的dataset與演算法之間的兼容性問題。而 K-Nearest Neighbors（KNN）的表現略優於預期，尤其是當neighbor數設為5時，表現比設為3時更佳，這與講義中的結果一致。

在所有試驗的方法中，SVM展現出最佳的性能。這可能是因為在處理高維度特徵和探索類別間複雜的非線性關係方面，SVM特別有效。特別是當使用RBF kernel時，SVM的表現尤為突出。RBF通過映射數據到更高維的空間來尋找類別間的最佳分隔超平面，這使得SVM在原始空間中非線性可分的data上表現出色。

綜上所述，這些分析結果強調了在音樂類型分類任務中選擇適當特徵和模型的重要性，並證明了在面對複雜數據時，SVM特別是RBF kernel的SVM在提升模型泛化能力方面的有效性。

