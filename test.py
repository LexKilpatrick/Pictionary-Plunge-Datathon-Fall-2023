import ast
import pandas as pd
from io import StringIO

def get_data(file: str, length: int): #maybe pick random lines until length reached
    dataset = []
    with open(file + ".ndjson") as data:
        #for line in data:
        for i in range(length):
            line = data.readline()
            dataset = [pd.read_json(StringIO(line))] #removes error
            print(dataset)

get_data("bucket", 5)