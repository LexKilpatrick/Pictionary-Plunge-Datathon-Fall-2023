from taipy.gui import Gui
import taipy
import matlab.engine
import matlab as ml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from engineio.async_drivers import gevent

path = None
prediction = "Waiting for input..."
w = "160px"
h = "160px"

md = '''# Pictionary Plunge 
Upload a picture, wait for it to appear and your prediction will be made below.

&emsp;

<|{path}|file_selector|label=Upload picture|on_action=createPrediction|extensions=.png,.jpg|> 

<|{path}|image|width={w}|height={h}|hover_text=Your Image!|> 

Prediction: <|{prediction}|>.
'''

def binarizeArr(path):
    eng = matlab.engine.start_matlab()
    RGB = eng.imread(path)
    GRY = eng.rgb2gray(RGB)
    arrHeight = 500
    arrWidth = 500
    targetSize = np.array([float(arrHeight), float(arrWidth)])
    SIZ = eng.resize(GRY, targetSize)

    height = len(SIZ)
    width = len(SIZ[0])

    binarizeImage = np.ndarray(shape=(height, width))
    for row in range(height):
        for col in range(width):
            value = SIZ[row][col] #All R, G, and B values are the same color/value
            if (value >= 128): #White
                binarizeImage[row][col] = 1
            else: #Black
                binarizeImage[row][col] = 0

    print("Binarized Array Created")
    return binarizeImage

def createPrediction(state):
    state.prediction = "Loading..."
    BW = binarizeArr(state.path) #numpy array of 1s and 0s
    problem = False

    #Connect to ML here
    model = tf.keras.models.load_model("model.h5")
    try:
        state.prediction = model.predict(BW)
    except:
        problem = True
    if (problem):
        state.prediction = "Error! Ran out of memory?"
    print("Create Prediction Done")

Gui(md).run()