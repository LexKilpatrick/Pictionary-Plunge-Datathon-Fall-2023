import os
import random
import json
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

import numpy as np

import numpy as np
from scipy.ndimage import zoom

def crop_doodle(image, target_size=(500, 500)):
    # Find the rows and columns that contain 1s
    rows = np.any(image == 1, axis=1)
    cols = np.any(image == 1, axis=0)
    
    # Get the minimum and maximum row indices and the minimum and maximum column indices
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop the image
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]
    
    # Calculate the resizing factors for rows and columns
    row_factor = target_size[0] / cropped_image.shape[0]
    col_factor = target_size[1] / cropped_image.shape[1]
    
    # Resize the cropped image to the target size
    resized_image = zoom(cropped_image, (row_factor, col_factor), order=1)  # Using bilinear interpolation (order=1)
    
    return resized_image



def draw_line(arr, start, end):
    """Draw a line in the array from start to end using Bresenham's algorithm."""
    
    x1, y1 = start
    x2, y2 = end
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
    
    # Rotate the line if needed
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    
    # Swap start and end points if necessary and retrieve the swapped values
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    
    # Recalculate the differences
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    
    # Iterate over bounding box generating points between start and end
    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        arr[coord[0], coord[1]] = 1
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    
    return arr


class DataStream:
    def __init__(self, num_image) -> None:
        self.data_path = 'AutoML/TrainingData/'
        self.allData = {i:self.get_data(file, num_image) for i, file in enumerate(os.listdir(self.data_path))}
        # self.fileNames={file[file.rindex('_')+1:file.index('.')]:i for i, file in enumerate(os.listdir(self.data_path))}
        self.fileNames={file[:file.index('.')]:i for i, file in enumerate(os.listdir(self.data_path))}

    def get_data(self, file: str, num_image):
        count=0
        with open(self.data_path+file) as data:
            for line in data:
                if count>num_image:
                    break
                dataset = json.loads(line)
                count+=1
                yield dataset

    def get_random_data(self):
        dataRemaining=list(range(0,len(self.allData)))
        while True:
            arr=np.zeros((2001,2001))
            randomChoice=random.choice(dataRemaining)
            data=(self.allData[randomChoice])
            try:
                image=next(data)
            except StopIteration:
                dataRemaining.remove(randomChoice)
                if len(dataRemaining)==0:
                    break
                else:
                    continue
            
            # if not i

            strokes=image['strokes']
            for stroke in strokes:
                x,y=stroke
                x=[round(x) for x in x]
                y=[round(y) for y in y]
                coords=list(zip(x,y))
                # for i in range(1, len(coords)):
                #     p1=coords[i-1]
                #     p2=coords[i]
                #     draw_line(arr, p1, p2)
                for coord in coords:
                    arr[coord[0]][coord[1]]=1
            
            arr=crop_doodle(arr)
            yield (np.rot90(arr, k=3),self.fileNames[image['category']])


# stream=DataStream()

# for image in stream.get_random_data():

#     print(image)
    # plt.imshow(np.array(image[0], dtype='float').reshape((256,256)), cmap='gray')
    # plt.show()
