from flask import Flask,jsonify,render_template,request
import requests
import json
from keras.models import load_model
from keras import backend as K

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization

from keras.preprocessing import image
import os
import io
from PIL import Image

import cv2
import numpy as np

app=Flask("__name__")


img_height = 155
img_width = 220
dim=(img_width,img_height)



def contrastive_loss(l, y_pred):
    #l=label(y_true)
    margin = 1
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin_square = (K.maximum(margin - y_pred, 0))**2

    # α,β= ½ 
    return K.mean((l * (y_pred)**2)*l + (1 - l)* margin_square)

def preprocess_image(img):
    resized_signature = cv2.resize(img,(220,155))
    gray_signature=cv2.cvtColor(resized_signature, cv2.COLOR_BGR2GRAY)
    ret,thr_img = cv2.threshold(gray_signature, 0, 255, cv2.THRESH_OTSU)
    normalized_signature=thr_img/255
    signature_expanded = normalized_signature[:, :, np.newaxis]
    
    return signature_expanded




# load model

#model=load_model(MODEL_PATH)


@app.route("/")
def verify():
    return render_template("upload.html")

@app.route("/upload",methods=['POST'])
def upload():
    image_pair=[]
    target=os.path.join(APP_ROOT,'static/')
    
    print(target)
    if(not os.path.isdir(target)):
        os.mkdir(target)
    
    i=0
    for file in request.files.getlist("file"):
        print(file)
        image_name=file.filename

        #destination="/".join([target,image_name])
        destination="/".join([target,image_name])
        print(destination)
        image_pair.append(image_name)

        file.save(destination)
        i+=1
    print(image_pair)
    return render_template("verify.html",image_file1=image_pair[0],image_file2=image_pair[1],pred="deneme")


from base64 import b64decode



APP_ROOT=os.path.dirname(os.path.abspath(__file__))

MODEL_PATH='/model/siyam_RMSProp_model.h5'


def get_model():
    global model
    path=APP_ROOT+MODEL_PATH
    model=load_model(path,custom_objects={'contrastive_loss':contrastive_loss})
    return model




def get_predict(img1,img2):
    
   
    pre_image=preprocess_image(img1)
    pre_image2=preprocess_image(img2)

    
        
    img_1=np.expand_dims(pre_image,axis=0)
    img_2=np.expand_dims(pre_image2,axis=0)

    model =get_model()

    y_pred = model.predict([img_1,img_2])
    print(y_pred)
    y1=(np.round(y_pred))
    print(y1)
    if (y1[0][0] == 0):
      print('The second signature is genuine \n')
      return "genuine"
      
    elif(y1[0][0]>=1):
      print('The second signature is forged \n')
      return "forged"
   



@app.route("/deneme",methods=['GET','POST'])
def deneme():

    dataDict=request.get_json()
    person_name=dataDict["name"]
        
    code=dataDict["image1"] #image enceoded(base64) 
    decode1=b64decode(code)
    code2=dataDict["image2"] #image enceoded(base64) 
    decode2=b64decode(code2)


    Image1=Image.open(io.BytesIO(decode1)).convert('RGB')
    Image1 = np.array(Image1) 

    #preprocess_image1=preprocess_image(Image1)

    Image2=Image.open(io.BytesIO(decode2)).convert('RGB')
    Image2=np.array(Image2)
    
    result=get_predict(Image1,Image2)

    #preprocess_image2=preprocess_image(Image2)

    """
    with open(  dataDict["name"]+"_1" +".png","wb") as f:#open file with person name
        f.write(b64decode(code)) #decode and write as a image

    with open(  dataDict["name"]+"_2" +".png","wb") as f:#open file with person name
        f.write(b64decode(code2)) #decode and write as a image
    """
    
    response={
        'Name':person_name,
        "Result":result
    }

    return jsonify(response)
    

if __name__=="__main__":
    app.run(debug=True)