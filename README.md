## Offline Signature Verification with Siamese Network


In that project Siamse Convolutional Neural Network Model is used for offline signature verifcation. It takes two signature pair for verification (Genuine-Forge).
The feature vector produced by both sides of the Siamese network is measured by a similarity metric containing Euclidean distance. This similarity metric is the most preferred Constrastive loss function, described below.



![Constr](https://user-images.githubusercontent.com/25572428/87487976-d19e1f80-c647-11ea-8182-a9d0fdf125f5.PNG)

```python
def contrastive_loss(l, y_pred):
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2
    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)
```

### Model illustration
![Siamese](https://user-images.githubusercontent.com/25572428/87487785-41f87100-c647-11ea-8f40-ec7d694625fa.png)



### Dataset and Preprocessing 
____
* The model is trained on [CEDAR dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar )
* Images were grouped in pairs of genuine and forged images
* Images were converted to grayscale, inverted and scaled down to 0 or up to 255 with OTSU Thresholding
* Images were resized to 155x220 then normalized

##### Original Image

##### After Preprocessing
![image](https://user-images.githubusercontent.com/25572428/87489668-4410fe80-c64c-11ea-953d-d4b93d5c87a6.png)

|Original Image | After Preprocessing| 
| ------------- |:-------------:|
|![image](https://user-images.githubusercontent.com/25572428/87489667-41aea480-c64c-11ea-870c-3c3c7b861117.png)|![image](https://user-images.githubusercontent.com/25572428/87489668-4410fe80-c64c-11ea-953d-d4b93d5c87a6.png)|


### Training
The model is trained on Google Colab using Keras which is **Siamese_network_forColab.ipynb**. The model and best weights are saved for the prediction

## Description
A Restful Web Service to decide  given signatures are real or forged. A service accept 


##### Run the Restful Web Services

```
python webservice.py
```




### Code Requirements 
---------------
```
pip install -r requirements.txt
```

### Docker Image
---





#### This is my final year project and also main subject of my thesis. 
### [Thesis](https://github.com/CantOkan/ENG402_Final_Year_Project/files/4841294/CAN.OKAN.TASKIRAN100042773.pdf)

### [Presentation](https://github.com/CantOkan/ENG402_Final_Year_Project/files/4841293/CanOkanTaskiran_2.Sunum.pdf)




### References
---------------------------
#### 1.[SigNet: Convolutional Siamese Network for Writer Independent Offline SignatureVerification](https://arxiv.org/pdf/1707.02131.pdf)
#### 2.[OfflineSignatureVerification](https://github.com/Aftaab99/OfflineSignatureVerification)
