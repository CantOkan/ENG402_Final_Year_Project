## Offline Signature Verification with Siamese Network

![Siamese](https://user-images.githubusercontent.com/25572428/87487785-41f87100-c647-11ea-8f40-ec7d694625fa.png)

In that project Siamse Convolutional Neural Network Model is used for offline signature verifcation. It takes two signature pair for verification (Genuine-Forge).
The feature vector produced by both sides of the Siamese network is measured by a similarity metric containing Euclidean distance. This similarity metric is the most preferred Constrastive loss function, described below.


```
## formula 
```
```python
def contrastive_loss(l, y_pred):
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2
    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)
```



### Dataset and Preprocessing 
____
The model is trained on [CEDAR dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar )

##### Run the Restful Web Services

```
python webservice.py
```




### Code Requirements 
---------------
```
pip install -r requirements.txt
```





#### This is my final year project and also main subject of my thesis. 
### [Thesis](https://github.com/CantOkan/ENG402_Final_Year_Project/files/4841294/CAN.OKAN.TASKIRAN100042773.pdf)

### [Presentation](https://github.com/CantOkan/ENG402_Final_Year_Project/files/4841293/CanOkanTaskiran_2.Sunum.pdf)




### References
---------------------------
#### 1.[SigNet: Convolutional Siamese Network for Writer Independent Offline SignatureVerification](https://arxiv.org/pdf/1707.02131.pdf)
#### 2.[OfflineSignatureVerification](https://github.com/Aftaab99/OfflineSignatureVerification)
