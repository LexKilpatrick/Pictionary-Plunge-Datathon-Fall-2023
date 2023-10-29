import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
path = "C:/Users/datga/Desktop/profile pic folder/tanya being suspiscious of blue hair.png"
RGB = eng.imread(path)
GRY = eng.rgb2gray(RGB)
targetSize = np.array([256.0, 256.0])
SIZ = eng.resize(GRY, targetSize)
print(np.shape(SIZ))
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

print(binarizeImage)