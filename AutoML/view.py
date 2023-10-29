from DataStream import DataStream
from matplotlib import pyplot as plt
import numpy as np

stream=DataStream(1000)
images=stream.get_random_data()

for image in images:
    plt.title(list(stream.fileNames.keys())[list(stream.fileNames.values()).index(image[1])])
    plt.imshow(np.array(image[0], dtype='float'), cmap='gray')
    plt.show()