from flask import Flask,jsonify,render_template,request
import requests
import json
from keras.models import load_model
from keras import backend as K
import os

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

"""

MODEL_PATH='model/siyam_RMSProp_model.h5'
def get_model():
    global model
    model=load_model(MODEL_PATH)
    print("model loaded")

"""


# load model

#model=load_model(MODEL_PATH)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

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


if __name__=="__main__":
    app.run(debug=True)