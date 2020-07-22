import tensorflow as tf
import flask
import tensorflow.keras as keras
import numpy as np
import re
import base64
import cv2
from gtts import gTTS
from flask import send_file
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
app = flask.Flask(__name__)

@app.route('/show_mnist')
def show_fashion():
  return flask.render_template('/index.html')

@app.route('/mnist/', methods =['POST'])
def fashion():
  
  imgData = flask.request.get_data()

  convertImage(imgData)
  x=cv2.imread('output.png')
  x=np.invert(x)

  x=cv2.resize(x,(28,28))
  x=x[np.newaxis,:,:,0:1]
  x= x.astype('float64') 
  
  model= keras.models.load_model("model_5_acc_99.h5")
  result=model.predict_classes(x)
  
  target_names=['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

  ans=target_names[result[0]]
  speech=gTTS(ans) //converting text to speech
  speech.save("hello.mp3")
  
  return str(ans)
@app.route('/audio/hello.mp3',methods=['GET'])
@nocache
def audio():
  path='hello.mp3'
  return send_file(path,mimetype="audio/mpeg", as_attachment=True, attachment_filename="hello.mp3")

def convertImage(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(imgstr))

