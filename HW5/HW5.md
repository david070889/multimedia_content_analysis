---
title: HW5

---

# HW5
Q56121052 黃書堯  
## 環境
作業系統 : WIN 11  
語言 : Python 3.11  
IDE : Jupyter Lab(colab)  
套件 : Numpy, torch, matplotlib, torchvision  
## 模型 - 1
第一部分完全參照alexnet模型，
```python=
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):  # 更改類別數為 3
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 輸出層改為 3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

並且優化器分別使用adam與SDG，learning rate為0.001開始，而loss fuction則使用``torch.nn.CrossEntropyLoss()``。


| Column 1 | loss | accuracy |
| -------- | -------- | -------- |
| Adam     | ![image](https://hackmd.io/_uploads/HkrjnHuQ0.png)| ![image](https://hackmd.io/_uploads/H140nBO70.png)|
|SDG|![image](https://hackmd.io/_uploads/BkW7irOQC.png)|![image](https://hackmd.io/_uploads/Hk3-6HdXC.png)|

## 模型 - 2
根據上面顯示的結果，我認為alexnet對於3000張圖片的資料集還是有點過大，所以我以alexnet為模板，減少了兩個conv layer與一個fully connected layer，希望能看到效能的上升。  
```pythob=
class SimplifiedAlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SimplifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一個卷積層
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二個卷積層
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三個卷積層
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # 第一個全連接層
            nn.Dropout(),
            nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # 第二個全連接層
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
```
| Column 1 | loss | accuracy |
| -------- | -------- | -------- |
| Adam     | ![image](https://hackmd.io/_uploads/HyZ-28dXC.png)| ![image](https://hackmd.io/_uploads/B1cQhLdQR.png)|
|SDG|![image](https://hackmd.io/_uploads/rk8_WDdmA.png)|![image](https://hackmd.io/_uploads/BJv9bw_XC.png)|  

我們能看到Adam優化器的結果與modified之前的Alexnet非常類似，可能同樣為資料不足或epoch不夠。
而SDG優化器就不有了不小的進步，從剛開始的training loss就不斷降低直到最後，但是validation loss再第10 epoch後就開始不斷上升。train_accuracy也是不斷上升到最後有非常好的表現，但是validation loss則是稍微上升後就持平。造成以上現象的原因我認為可能是overfitting，因為需要訓練的參數因為刪除許多層數後大幅度減少。

## 加入learning rate scheduler
鑒於剛剛的simplified Alexnet可能有overfitting，所以我在多嘗試一個learning rate scheduler的方法。  
``scheduler = StepLR(optimizer, step_size=15, gamma=0.1)``  
觀察前面圖片，可看出大約10多個epoch後在validation set的表現開始衰退，所以這裡我設定每15個epoch讓learning rate乘以0.1。  

| | loss | accuracy |
| -------- | -------- | -------- |
| simplified alexnet with scheduler |![image](https://hackmd.io/_uploads/Hy-NHYOQC.png)|![image](https://hackmd.io/_uploads/S1YIrFdX0.png)|

結果這次的結果與我設想的差了更多，變成前面與Adam優化器類似的結果。所以結果我猜測這邊常常因為進入了一個saddle point出不來，造成有好幾張圖片都呈現震盪，但是卻無法獲得更好表現的模樣。