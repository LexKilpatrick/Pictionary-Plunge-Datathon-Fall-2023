from taipy.gui import Gui
import matlab.engine
import matlab as ml
import imageio.v3 as iio
import numpy as np

path = None

md = '''# Pictionary Plunge 
<|{path}|file_selector|label=Upload picture|on_action=createPrediction|extensions=.png,.jpg|> 

<|{path}|image|on_action=clickCheck|width=128px|height=128px|>
'''

def binarizeArr(path):
    eng = matlab.engine.start_matlab()
    RGB = eng.imread(path)
    GRY = eng.rgb2gray(RGB)
    targetSize = np.array([256.0, 256.0])
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
    return binarizeImage

def createPrediction(state):
    global path
    path = state.path
    BW = binarizeArr(path) #numpy array of 1s and 0s
    #Connect to ML here

Gui(md).run()