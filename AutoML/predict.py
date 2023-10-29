from keras.models import load_model
import numpy as np

class model_load:
    def __init__(self, model_path) -> None:
        self.path=model_path
        self.model=load_model(self.path)

    def predict(self, arr):
        return self.model.predict(np.array([arr]))
    
