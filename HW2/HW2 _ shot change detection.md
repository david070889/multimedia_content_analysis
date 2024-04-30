# HW2 : video shot change detection
## 環境
作業系統 : WIN 11  
語言 : Python 3.12  
IDE : VSCODE 1.87  
套件 : Matpoltlib, cv2, numPy, os  

## 擷取特徵
Gray scale histogram - histogram difference  
Edge detection - Edge Change Ratio  

## 方法
### Histogram difference
呼叫函式gray_hist_compare()，先將圖片轉灰階，  
```python=
grayImg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
```
統計其從0-255的每階像素數量，
```python=
hist1, bin_edge = np.histogram(data1, bins=256, range=(0, 255))
```
再與下一張圖片對比(next frame)，即可獲得similarity。
```python=
degree = 0
for i in range(len(hist1)): # 跑每一條 histogram
if hist1[i] != hist2[i]: # 不相同
    degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
else:degree = degree + 1
```
將similarity依照frame的前後順序，畫出相似性的變化圖。  
```python=
x_axis = list(range(len(hist_diff_arr)))  # 生成索引列表
y_axis = hist_diff_arr  # 數據列表
plt.plot(x_axis, y_axis, linestyle='-', color='b', label = 'similarity')
```

## Edge change ratio
呼叫函式ECR()，先將圖片轉換為灰階圖片，  
```python=
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
再使用Canny method找到圖片中明顯的edge，  
```python=
edge1 = cv2.Canny(gray_image, 100, 200)
```
對比前後兩張圖片的edge變化，計算出兩個圖片的difference。  
```python=
log_and1 = (edge1 & inverted2)
    log_and2 = (edge2 & inverted)
    
    pixels_sum_new = np.sum(edge1)
    pixels_sum_old = np.sum(edge2)

    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)

    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))
```
將difference為y軸，以時間為順序的frame為x軸，畫出difference的變化。  
```python=
x_axis = list(range(len(ecr_arr)))  # 生成索引列表
y_axis = ecr_arr  # 數據列表
plt.figure() 
plt.plot(x_axis, y_axis, linestyle='-', color='b', label = 'difference')
```
## 計算 precision, recall
在計算precision和recall的函式calculate_precision_recall()，因為在climate與ngc兩個txt檔案裏面都有整段區間的shot change，  
所以做了判斷式，判斷是否為integer，用於計算的PR。  
```python=
for predicted_val in predicted:
    for actual_val in actual:
        if isinstance(actual_val, list):  # 如果 a 中的元素是列表（區間）
            if predicted_val in actual_val:  # 檢查 b_val 是否在區間內
                TP += 1
                break
        elif predicted_val == actual_val:  # 如果 a 中的元素是數值，直接比較
            TP += 1
            break
            
```


## 結果與討論
### Histogram difference
呼叫gray_hist_compare()，返回數組後，因為數值為similarity，所以以小於即為threshold的數值為判定的shot change，且我們開始尋找threshold讓precision和recall都有相對好的表現。  
|||
|--|--|
|![hd_news](https://hackmd.io/_uploads/B1ydUlJ1C.jpg)|![hd_climate](https://hackmd.io/_uploads/By1yteJJ0.jpg)|
|![hd_ngc](https://hackmd.io/_uploads/Sy8qnxk10.jpg)||

如下表所示，將news與climate的threshold訂為<=0.8，可以獲得較好的Precision和Recall，並且由上圖能看出看出經過gray_hist_compare()後shot change的frame都有非常明顯的difference。  
但是ngc的表現在這裡的表現並不是太好，將threshold定在相對較下面的位置也只能勉強讓precision和recall的表現維持平衡。  
我認為這邊ngc表現不好的原因為，畫面太常出現單一色調，轉換成灰階後容易偏白或是偏黑，造成similarity的大起大落。  

下方為統計三種資料的表格，執行時間為單純呼叫gray_hist_compare()計算出的時間。
|file|news|ngc|climate|
|----|---|---|-------|
|threshold|0.8|0.45|0.8|
|precision|0.875|0.41|0.75|
|recall|1.0|0.5|0.6|
|excution time(s)|5.5|15|32|

### Edge change ratio
呼叫ECR()，返回的數組為difference，且我們開始尋找threshold讓precision和recall都有相對好的表現。  
|||
|--|--|
|![ed_news](https://hackmd.io/_uploads/Hkj9Y-1JR.jpg)|![ed_climate](https://hackmd.io/_uploads/SJNmqbkkR.jpg)|
|![ed_ngc](https://hackmd.io/_uploads/H1W5aZkyC.jpg)||

觀察上面三張圖，唯一較為正常的只有climate image，difference能夠有足夠的鑑別度。news的前面與後面的frame變化都非常劇烈，我猜測偵測到的edge對於分辨shot change來說還是太少，使得變化過大。

整體來說，我設計的edge change ratio完全比不上簡單的hist difference，而我認為原因有兩點。第一點為我用Canny method作為edge detection只有灰階，可能不足以分辨讓返回的數組有足夠的辨識性。第二點為，相較news與climate，ngc的畫面多以景色為主，特徵不是過少就是過多，使得單一的thresholds難以分辨出真正的shot change。  

|file|news|ngc|climate|
|----|---|---|-------|
|threshold|0.25|0.25|0.12|
|precision|0.5|0.04|1|
|recall|0.29|0.11|0.33|
|excution time|3.1|7.2|15.5|

這邊的edge change ratio唯一的好處在於execution time只有histogram difference的一半，原因可能在於以canny method為主的detection大大減少了計算量，卻可能也是使得最後的結果不堪用的原因，之後嘗試以RGB做偵測或許能使ECR更穩定。  






