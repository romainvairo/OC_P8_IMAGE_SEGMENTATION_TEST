# ------------------------- Imports Libraries ------------------------------

from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse
import uvicorn
import base64
import numpy as np
import os
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import OneHotMeanIoU
from tensorflow.keras import backend as K

# --------------------------------------------------------------------------




# ----------------------------- Functions ----------------------------------

def dice_coeff(y_true, y_pred):
    """ Dice coefficient

    :param y_true : true values
    :param y_pred : predicted values

    :return score : return the Dice coefficient """

    smooth = 0.001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score

def dice_loss(y_true, y_pred):
    """ Dice loss metric

    :param y_true : true values
    :param y_pred : predicted values

    :return loss : return the score of the Dice loss metric """

    loss = 1 - dice_coeff(y_true, y_pred)

    return loss

def total_loss(y_true, y_pred):
    """ total loss function

    :param y_true : true values
    :param y_pred : predicted values

    :return loss : return the score of the total loss function """

    loss = categorical_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))

    return loss

def IoU(y_true, y_pred):
    """ IOU metric

    :param y_true : true values
    :param y_pred : predicted values

    :return result : return the score of the IOU metric """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - intersection
    result = intersection / denominator

    return result

def segmentation_color(real_mask):
    """ segmentation_color function

    :param real_mask : real mask

    :return real_mask : return transform 35 categories in 8 categories """
    labels_color = np.array([[0, 0, 0], [206, 26, 26], [2237, 165, 21], [132, 132, 132], [31, 161, 135], [255, 0, 255], [98, 200, 122], [187, 7, 247]])
    real_mask = labels_color[real_mask]
    real_mask = real_mask.astype(np.uint8)
    return real_mask

# --------------------------------------------------------------------------




# --------------------------- API Creation ---------------------------------

app = FastAPI()

# --------------------------------------------------------------------------




# ------------------------------ Model -------------------------------------

model = load_model("unet_mini_not_augmented.h5", custom_objects={'dice_coeff' : dice_coeff,
'mean_iou' : OneHotMeanIoU(num_classes=8, name='mean_iou'), 'IoU' : IoU}, compile = True)

# --------------------------- POST Request ---------------------------------

@app.get('/')
def index():
    '''
    Test de l'API
    '''
    return {'message': 'Bonjour, ceci est un test'}

@app.post("/predict_mask/")
async def predict_mask(file: UploadFile = File()):

    # Get the original image with (1, 256, 256, 3) shape
    contents = await file.read()
    conversion_np_array = np.fromstring(contents, np.uint8)
    
    # Decode image using Pillow (PIL)
    img = Image.open(BytesIO(conversion_np_array))
    img_resized = img.resize((256, 256))
    img_np = np.array(img_resized)
    img_batch = np.expand_dims(img_np, axis=0)

    # Predict mask with model and assign color to it
    predicted_mask = model.predict(img_batch)
    predicted_mask = predicted_mask.reshape(1,256,256,8)[0,:,:,:]
    predicted_mask = np.array(np.argmax(predicted_mask, axis=2), dtype='uint8')
    predicted_mask = segmentation_color(predicted_mask)

    # Create image/predicted mask with alpha parameter
    alpha_image = Image.blend(Image.fromarray(predicted_mask.astype(np.uint8)), Image.fromarray(img_np.astype(np.uint8)), 0.5)

    # Encode original image/predicted mask to PNG format
    buffer_mask = BytesIO()
    alpha_image.save(buffer_mask, format='PNG')
    buffer_mask.seek(0)

    # Encode original image/predicted mask in base64 format
    mask = base64.b64encode(buffer_mask.read()).decode("utf-8")
    
    # Return a Response in JSON format
    return JSONResponse(content={"prediction": mask})

# --------------------------------------------------------------------------




# ----------------------- Application's Running ----------------------------

if __name__ == '__main__': 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# --------------------------------------------------------------------------