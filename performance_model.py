# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:33:38 2020

@author: canok
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 03:27:45 2020

@author: canok
"""


import numpy as np
import keras
from PIL import Image
import os 
import glob
import cv2
from keras.preprocessing import image
import numpy as np
from natsort import natsorted, ns
import pandas as pd



######################### DataGenerator


current_dir = os.path.dirname(__file__)


path_org=os.path.join(current_dir,'signatures/full_org')
path_frog= os.path.join(current_dir,'signatures/full_forg')

def get_dataframe():
  forg_images=[]
  for indis in range(1,56):#55 different classes   #56
    if(indis==41):
      continue
    else:
        for j in range(1,25): #25
            #path=os.path.join(path_frog,'/forgeries_'+str(indis)+'_'+str(j)+'.png')
            path=(path_frog+'/forgeries_'+str(indis)+'_'+str(j)+'.png')
              #img=cv2.imread(path)
            # p.append(path)
            forg_images.append(path)
      
    
  org_images=[]
  for indis in range(1,56):#55 different classes   #56
    if(indis==41):
      continue
    else:
      for j in range(1,25): #24 farklı gerçek-gerçek veya gerçek-sahte imza çifti
          #path=os.path.join(path_org,'/original_'+str(indis)+'_'+str(j)+'.png')
          path=(path_org+'/original_'+str(indis)+'_'+str(j)+'.png')
          #img=cv2.imread(path)
          #p.append(path)

          org_images.append(path)  

  no_of_ppl = len(org_images)//24 #her kisi icin 24 sample var
    #find the number of class  
  
  raw_data = {"sing_1":[], "sign_2":[], "label":[]}
  
  for i in range(no_of_ppl):
    i1_batch_1 = []
    i1_batch_2 = []
    i2_batch = []

    start = i*24
    end = (i+1)*24
    
    for j in range(start,end): 
      i1_batch_1.append(org_images[j])
      i1_batch_2.append(org_images[j])
      raw_data["label"].append(1)#0

    temp_rot = (i1_batch_1[-12:]+i1_batch_1[:-12])
    i1_batch_1.extend(i1_batch_2)

    for elem in temp_rot:
      i2_batch.append(elem)

    for j in range(start,end): 
      i2_batch.append(forg_images[j])
      raw_data["label"].append(0)#1

    raw_data["sing_1"].extend(i1_batch_1)
    raw_data["sign_2"].extend(i2_batch)
  df = pd.DataFrame(raw_data, columns = ["sing_1","sign_2","label"])
  df=df.reindex(df.index)
  return df






from sklearn.model_selection import train_test_split


def get_dataset():
  df = get_dataframe()
  #print(df.shape)
  
  train_set, val_set = train_test_split(df,test_size=0.3,random_state=0)
  
  return train_set, val_set

ds_train,ds_val = get_dataset()
print(len(ds_val))


import numpy as np
import keras
from PIL import Image
import cv2








class SignatureSequence(keras.utils.Sequence):
    
    def __init__(self, df, batch_size, dim):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.labels = df["label"]
        
        self.on_epoch_end()

    def __len__(self):
        s_df=self.df.shape[0]
        n=np.floor(s_df/self.batch_size)
        return int(n)

    def __getitem__(self, indis):
         #indexes
        batches = self.indises[indis*self.batch_size:(indis+1)*self.batch_size]
        items = [self.df.iloc[k] for k in batches]
        part1,part2 = self.generator(items)
        return part1,part2

    def on_epoch_end(self):
        self.indises = np.arange(self.df.shape[0])
        

    def generator(self, items):
        part_1 = np.empty((self.batch_size, *self.dim,1))#working with gray images
        part_2 = np.empty((self.batch_size, *self.dim,1))#working with gray images
        label = np.empty((self.batch_size), dtype=int)
        
        for i in range(len(items)):
            #image 1
            signature_1 = cv2.imread(items[i]["sing_1"])
           
            resized_signature = cv2.resize(signature_1,(220,155))
            gray_signature=cv2.cvtColor(resized_signature, cv2.COLOR_BGR2GRAY)
            ret,thr_img = cv2.threshold(gray_signature, 0, 255, cv2.THRESH_OTSU)
            normalized_signature=thr_img/255
            signature_expanded = normalized_signature[:, :, np.newaxis]
            signature_1=np.array(signature_expanded)

            #image 2
            signature_2 = cv2.imread(items[i]["sign_2"])
            
            resized_signature = cv2.resize(signature_2,(220,155))
            gray_signature=cv2.cvtColor(resized_signature, cv2.COLOR_BGR2GRAY)
            ret,thr_img = cv2.threshold(gray_signature, 0, 255, cv2.THRESH_OTSU)
            normalized_signature=thr_img/255
            signature_expanded = normalized_signature[:, :, np.newaxis]
            signature_2=np.array(signature_expanded)
            

            
          
            label[i] = items[i]["label"]
            part_1[i,] = signature_1
            part_2[i,] = signature_2 

        return [part_1 ,part_2], label

###DataGenerator




import numpy as np
import keras
from PIL import Image
import cv2





img_width= 155
img_height = 220


dim=(img_width,img_height)
batch_size=64
train_datagen = SignatureSequence(ds_train,batch_size,dim)
validation_datagen = SignatureSequence(ds_val,batch_size,dim)

###DataGenerator






from keras.models import load_model
from keras import backend as K


def contrastive_loss(l, y_pred):
    #l=label(y_true)
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2

    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)



mod = load_model('siyam_model.h5',custom_objects={'contrastive_loss':contrastive_loss})



y_pred = mod.predict_generator(validation_datagen)#, steps=25

print("count  pred:"+str(len(y_pred)))
ds_labels=validation_datagen.labels[0:768]
print("count of labels: "+str(len(ds_labels)))







from keras.models import load_model
from keras import backend as K


def contrastive_loss(l, y_pred):
    #l=label(y_true)
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2

    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)



mod = load_model('siyam_RMSProp_model.h5',custom_objects={'contrastive_loss':contrastive_loss})



y_pred = mod.predict_generator(validation_datagen)#, steps=25
ds_labels=validation_datagen.labels[0:768]





def accuracy_hesapla(y_pred, y_true):
    '''Compute ROC accuracy with a range of thresholds on distances.
    '''
    df_max = np.max(y_pred)
    df_min = np.min(y_pred)
    number_of_sim = np.sum(y_true == 1)
    number_of_diff = np.sum(y_true == 0)
   
    step = 0.01
    max_acc = 0
    best_thresh=0
    
    true_pos_rate_list=[]
    for distance in np.arange(df_min, df_max+step, step):
        idx1 = y_pred.ravel() <= distance
        idx2 = y_pred.ravel() > distance
       
        true_pos_rate = float(np.sum(y_true[idx1] == 1)) / number_of_sim
        true_neg_rate  = float(np.sum(y_true[idx2] == 0)) / number_of_diff
        acc = 0.5 * (true_pos_rate + true_neg_rate )       
#       
        
        true_pos_rate_list.append(true_pos_rate)
        if (acc > max_acc):
            max_acc, best_thresh = acc, distance
           
    return max_acc, best_thresh,true_pos_rate_list


max_acc,threshold,tpr_list=accuracy_hesapla(y_pred,ds_labels)

print("Max Accuracy:"+str(max_acc))
print("Best thresh:"+str(threshold))


from sklearn import metrics as m
#threshold=0.5203162277571391
confusionmatrix = m.confusion_matrix(ds_labels, y_pred <threshold)

print('false acceptance', confusionmatrix[0, 1] / confusionmatrix[0, :].sum())
print('false rejection', confusionmatrix[1, 0] / confusionmatrix[1, :].sum())
print('accuracy', m.accuracy_score(ds_labels, y_pred < threshold))





from sklearn.metrics import classification_report

print(classification_report(ds_labels, y_pred <threshold))













