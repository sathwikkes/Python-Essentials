import tensorflow as tf
from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel


MODEL = tf.saved_model.load('./saved/')
label_names= ['_background_noise_', 'backward' ,'bed', 'bird' ,'cat' ,'dog', 'down', 'eight',
  'five', 'follow' ,'forward', 'four' ,'go' ,'happy', 'house', 'learn', 'left',
  'marvin', 'nine', 'no', 'off', 'on' ,'one', 'right', 'seven' ,'sheila', 'six',
  'stop', 'three' ,'tree', 'two', 'up' ,'visual', 'wow', 'yes', 'zero']
app = FastAPI()




@app.get('/')
async def index():
    return {"Message": "This is Index"}


@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    prediction = MODEL(file.filename)
    class_ids = tf.argmax(prediction["predictions"], axis=-1)
    class_names = label_names[int(class_ids)]
    return {"prediction": class_names}