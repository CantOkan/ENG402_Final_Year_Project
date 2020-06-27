# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:35:29 2020

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
import random
import cv2

current_dir = os.path.dirname(__file__)


path_org=os.path.join(current_dir,'signatures/full_org')
path_frog= os.path.join(current_dir,'signatures/full_forg')


count_of_person=55
number_of_sample=24


def create_dataframe():
  fake_sign=[]
  for indis in range(1,count_of_person+1):#55 different classes   #56
      for j in range(1,number_of_sample+1): #25
          path=os.path.join(current_dir,'signatures/full_forg/forgeries_'+str(indis)+'_'+str(j)+'.png')
            #img=cv2.imread(path)
           # p.append(path)
          fake_sign.append(path)
     
    
  real_sign=[]
  for indis in range(1,count_of_person+1):#55 different classes   #56
     for j in range(1,number_of_sample+1): #25
         path=os.path.join(current_dir,'signatures/full_org/original_'+str(indis)+'_'+str(j)+'.png')
         #img=cv2.imread(path)
         #p.append(path)

         real_sign.append(path)  
     
    
    #her kisi icin 24 sample var
    #Sınıf saysı =55 
  

  raw_data = {"sing_1":[], "sign_2":[], "label":[]}
  
  for kisi in range(count_of_person):
 
    real_signs_1=[]
    real_signs_2=[]
    fake_signs_1=[]
    
    indis_start = kisi*24
    indis_end = (kisi+1)*24
    
    for sample in range(indis_start,indis_end): 
      real_signs_1.append(real_sign[sample])
      real_signs_2.append(real_sign[sample])
      raw_data["label"].append(1) 
      
      #etkiet 1 gerçek imza
      #label 1 represents the geniune pair
   
    real_signs_1.extend(real_signs_2)

    for sign in real_signs_2:
      fake_signs_1.append(sign)

    for j in range(indis_start,indis_end): 
      fake_signs_1.append(os.path.join(path_frog,fake_sign[j]))
      raw_data["label"].append(0)
       #etkiet 0 sahte imzaları temsil etmektedir

    raw_data["sing_1"].extend(real_signs_1) #real-real pairs
    raw_data["sign_2"].extend(fake_signs_1) #fake-fake pairs
  df = pd.DataFrame(raw_data, columns = ["sing_1","sign_2","label"])
  return df


from sklearn.model_selection import train_test_split


def train_val_dataset():
  data_frame = create_dataframe()
  print(data_frame.shape)
  
  data_frame=data_frame.reindex(np.random.permutation(data_frame.index))
  
  train_set, val_set = train_test_split(data_frame,test_size=0.3,random_state=0)
  
  return train_set, val_set

train_set,val_set = train_val_dataset()
print(len(val_set))


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
        np.random.shuffle(self.indises)

    def generator(self, items):
        part_1 = np.empty((self.batch_size, *self.dim,1))#working with gray images
        part_2 = np.empty((self.batch_size, *self.dim,1))#working with gray images
        label = np.empty((self.batch_size), dtype=int)
        
        for i in range(len(items)):
            #image 1
            signature_1 = cv2.imread(items[i]["sign_1"])
           
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






dim=(155,220)
batch_size=64


data_train = SignatureSequence(train_set,batch_size,dim)
data_validation = SignatureSequence(val_set,batch_size,dim)


########  Network ###############################

from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Input, Dropout, Flatten
from keras.models import Model
from keras.models import Sequential

from keras.layers import Lambda


def network_model():
  input_shape=(155,220,1)
  in_imgLeft = Input(shape=input_shape, name="left_image")
  in_imgRight = Input(shape=input_shape, name="right_image")  

  model = Sequential()
  #1st Conv layer
  model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
  model.add( MaxPooling2D(pool_size=(3,3)) )
  #2nd Conv layer
  model.add( Conv2D(64, (3, 3), activation="relu") )
  model.add( MaxPooling2D(pool_size=(3,3),strides=(2,2)) )
  
  #3rd Conv layer
  model.add( Conv2D(128, (3, 3), activation="relu") )
  #4st Conv layer
  model.add( Conv2D(64, (3, 3), activation="relu") )
  #5st Conv layer
  model.add( Conv2D(128, (3, 3), activation="relu") )

  model.add( MaxPooling2D(pool_size=(3,3),strides=(2,2)) )

  model.add( Dropout(0.3) )
  #Fully Connected Layer
  model.add( Flatten() )
  
  model.add( Dense(256, activation="relu") )
  model.add( Dropout(0.5) )

  model.add( Dense(256, activation="relu") )

  
  left_branch = model(in_imgLeft)
  right_branch = model(in_imgRight)

  cal = Lambda(euclidean_distance,output_shape=output_shape)
  distance=cal([left_branch, right_branch])

  model = Model([in_imgLeft, in_imgRight], distance)

  return model


from keras import backend as K

def euclidean_distance(vectors):
    xi, yi = vectors
    return K.sqrt(K.sum(K.square(xi - yi), axis=1, keepdims=True))

def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(l, y_pred):
    #l=label(y_true)
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2
    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)
  
  
def accuracy(y_true, y_pred):    
    #y_true.dtype)
    casted_ytrue=K.cast(y_pred < 0.5, 'float32')
    return K.mean(K.equal(y_true,casted_ytrue ))


from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop,SGD

model = network_model()

#   RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08) best

opt=RMSprop()
model.compile(loss=contrastive_loss, optimizer=opt,metrics=[accuracy])


checkpointer = ModelCheckpoint(filepath="best_weights_for_Network.hdf5", 
                               monitor = 'val_accuracy',
                               verbose=1, 
                               save_best_only=True)

history=model.fit_generator(generator=data_train,validation_data=data_validation, epochs=20,steps_per_epoch=56, validation_steps=24
                            ,callbacks=[checkpointer])

model.save("siyam_RMSProp_model.h5")