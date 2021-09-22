from __future__ import division, print_function
# coding=utf-8
import csv

import sys
import os
import glob
import re
import numpy as np
import json
import tensorflow as tf
from  tensorflow.keras.preprocessing import image
from PIL import Image
import requests
# import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
from flask import Flask, jsonify, request
import urllib.request
import urllib
import cv2
app = Flask(__name__)

def model_predict(img_path, model):
    #print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"    
    
    print(preds)
    return preds
MODEL_PATH ='model_inception.h5'

# Load the  trained model
model = load_model(MODEL_PATH)
@app.route('/',methods=['POST','GET'])
def fun():              
    
    #get the http request from flutter 
    request_data = request.data
    request_data = json.loads(request_data.decode('utf-8'))
    userurl = request_data['url']
    #userurl represents the url of image stored in firebase storage which is uploaded in mobile 
    print(userurl)
    #conversition of the url to image
    response = requests.get(userurl, stream=True)
    img = Image.open(response.raw)
    #save the image
    img.save('3.jpg')

    img_path='3.jpg'
    #print(img_path)
    img = image.load_img(img_path,target_size=(224, 224))
      
    preds = model_predict(img_path, model)
    result=preds
    return result 

 # if os.path.exists("2.jpg"):
    #      os.remove("2.jpg")
  #print(request_data['https://firebasestorage.googleapis.com/v0/b/cip1-5e4a5.appspot.com/o/dis_leaf%20(5)_iaip.jpg?alt=media&token=902e5ba4-d445-4930-a53d-3e2278d779fc'])
    #   print("HAI")
      #model_predict(img_path,model)
      
# resp = urllib.request.urlopen(url)
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# img = cv2.imdecode(image, cv2.IMREAD_COLOR)
#url1= "https://firebasestorage.googleapis.com/v0/b/cip1-5e4a5.appspot.com/o/dis_leaf%20(5)_iaip.jpg?alt=media&token=902e5ba4-d445-4930-a53d-3e2278d779fc"
# response = requests.get(url, stream=True)
# img = Image.open(response.raw)
# plt.imshow(img)
#im = Image.open(urllib.request.urlopen(url))

# req = urllib.request.urlopen(url)
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# img = cv2.imdecode(arr, -1)


# from io import BytesIO
# response = requests.get(url)
# img = Image.open(BytesIO(response.content))
# img1=
#
# resp = urllib.request.urlopen(url)
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# img = cv2.imdecode(image, cv2.IMREAD_COLOR)

# img_path='3.jpg'
# print(img_path)

# img = image.load_img(img_path,target_size=(224, 224))
#  return image.load_img(path, grayscale=grayscale, color_mode=color_mode
         
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
