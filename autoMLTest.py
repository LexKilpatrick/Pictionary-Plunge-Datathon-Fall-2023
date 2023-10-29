import ast
import pandas as pd
from io import StringIO
import numpy as np

def get_data(file: str, length: int = 0): #maybe pick random lines until length reached
    dataset = []
    with open(file + ".ndjson") as data:
        #for line in data:
        if (length > 0):
            for i in range(length):
                line = data.readline()
                dataset += [pd.read_json(StringIO(line))] #removes error
        else:
            for line in data:
                dataset += [pd.read_json(StringIO(line))] #removes error
    return dataset

dataset = pd.read_json("bucketSample.ndjson", lines=True)
ids = np.arange(0,np.shape(dataset)[0])
truth = np.random.randint(2, size=np.shape(dataset)[0])
both = np.transpose(np.stack([ids, truth]))
print(np.shape(both))
print(np.shape(dataset))

import flaml as fla

automl = fla.AutoML()
automl.fit(X_train=dataset, y_train=both, task="classification", time_budget=25)


