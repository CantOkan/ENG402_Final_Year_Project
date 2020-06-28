# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:10:53 2020

@author: canok
"""
import os 
import glob
import cv2
from keras.preprocessing import image
import numpy as np
import random

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image

from keras.models import load_model
from keras import backend as K



def contrastive_loss(l, y_pred):
    #l=label(y_true)
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2

    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)


img_height = 155
img_width = 220


dim=(img_width,img_height)

def preprocess_image(img):
    resized_signature = cv2.resize(img,(220,155))
    gray_signature=cv2.cvtColor(resized_signature, cv2.COLOR_BGR2GRAY)
    ret,thr_img = cv2.threshold(gray_signature, 0, 255, cv2.THRESH_OTSU)
    normalized_signature=thr_img/255
    signature_expanded = normalized_signature[:, :, np.newaxis]
    
    return signature_expanded


mod = load_model('siyam_RMSProp_model.h5',custom_objects={'contrastive_loss':contrastive_loss})

#kendi_model.h5 ,siyam_RMSProp_model.h5

root=Tk()

root.title("Signature Classification")
root.geometry("800x800")





btn_width=20
btn_height=1

pic_width=400
pic_height=400
def open1():
    global image2
    root.filename=filedialog.askopenfilename(initialdir='/signatures',title='Select Image',filetypes=(("png files","*.png"),("all files","*.*")))
    img=Image.open(root.filename)
    print(img)
    resized_img=img.resize((pic_width, pic_height), Image.ANTIALIAS)
    image2=ImageTk.PhotoImage(resized_img)
    my_image_box2=Label(image=image2).place(x=pic_width+20,y=50)
    
    global img2
    img2=cv2.imread(root.filename)
    

def open():
    global image1
    root.filename=filedialog.askopenfilename(initialdir='/signatures',title='Select Image',filetypes=(("png files","*.png"),("all files","*.*")))
    img=Image.open(root.filename)
    print(img)
    resized_img=img.resize((pic_width, pic_height), Image.ANTIALIAS)
    image1=ImageTk.PhotoImage(resized_img)
    
    my_image_box=Label(image=image1).place(x=0,y=50)
    
    global img1
    img1=cv2.imread(root.filename)



def get_predict():
    global img1

    global img2

    
    pre_image=preprocess_image(img1)
    pre_image2=preprocess_image(img2)

        
    img_1=np.expand_dims(pre_image,axis=0)
    img_2=np.expand_dims(pre_image2,axis=0)
    
    y_pred = mod.predict([img_1,img_2])
    
    y1=(np.round(y_pred))
    print(y1)
    if (y1[0][0] == 0):
      print('The second signature is genuine \n')
      label="İkinci imza gerçek"
      color="green"
    elif(y1[0][0]>=1):
      print('The second signature is forged \n')
      label="İkinci imza sahte"
      color="red"
      
    
    Label(root,text=label,fg=color).place(x=pic_width,y=pic_height+200)
    





my_btn=Button(root,text="Open Signature",height=btn_height,width=btn_width,command=open).place(x=100,y=0)
my_btn2=Button(root,text="Open Signature2",width=btn_width,height=btn_height,command=open1).place(x=pic_width+200,y=0)




my_btn3=Button(root,text="Verify",width=btn_width,height=btn_height,command=get_predict).place(x=pic_width-20,y=pic_height+100)

#my_btn3=Button(root,text="Verify",command=get_predict).grid(row=15, column=5) 

root.mainloop()