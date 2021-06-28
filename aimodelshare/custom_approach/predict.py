#Image Classification Prediction Runtime Code

import cv2
import os
import numpy as np
import json
import base64
import imghdr
import six

from load_model import load_model
model = load_model('runtime_model.onnx', filetype='onnx')

def preprocessor(data, shape=(192, 192)):

    "Resize a color image and min/max transform the image"
    img = cv2.imread(data) # Read in image from filepath.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in images in order of blue green and red, we reverse the order for ML.
    img = cv2.resize(img, shape) # Change height and width of image.
    img = img / 255.0 # Min-max transform.

    # Resize all the images...
    X = np.array(img) # Converting to numpy array.
    X = np.expand_dims(X, axis=0) # Expand dims to add "1" to object shape [1, h, w, channels].
    X = np.array(X, dtype=np.float32) # Final shape for onnx runtime.

    return X

def predict_classes(x): 
    if x.shape[-1] > 1:
        return x.argmax(axis=-1)
    else:
        return (x > 0.5).astype("int32")

def predict(image_path):

    labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # Generate prediction using preprocessed input data
    print("The model expects input shape:", model.get_inputs()[0].shape)

    input_data = preprocessor(image_path)
    
    input_name = model.get_inputs()[0].name

    res = model.run(None, {input_name: input_data})
  
    #extract predicted probability for all classes, extract predicted label
    prob = res[0]

    prediction_index=predict_classes(prob)

    result=list(map(lambda x: labels[x], prediction_index))

    return result

def handler(event, context):

    # Load base64 encoded image stored within "data" key of event dictionary
    body = event["body"]
    if isinstance(event["body"], six.string_types):
        body = json.loads(event["body"])

    bodydata = body['data']

    # Extract image file extension (e.g.-jpg, png, etc.)
    sample = base64.decodebytes(bytearray(bodydata, "utf-8"))

    for tf in imghdr.tests:
        image_file_type = tf(sample, None)
        if image_file_type:
            break
    image_file_type = image_file_type

    if(image_file_type == None):
        print("This file is not an image, please submit an image base64 encoded image file.")

    temp_file = "imagetopredict."+image_file_type

    # Save image to local file, read into session, and preprocess image with preprocessor function
    with open(temp_file, "wb") as fh:
        fh.write(base64.b64decode(bodydata))

    result = predict(temp_file)

    os.remove(temp_file)

    return result