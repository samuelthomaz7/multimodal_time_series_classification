import pandas as pd

class GetDummiesLabels():

    def __init__(self, X_raw, y_raw, metadata = None):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.metadata = metadata
        self.step_name = 'GetDummiesLabels'

    
    def fit(self):
        pass

    def transform(self):
        self.y_dummies = pd.get_dummies(self.y_raw).astype(int).values
        return self.X_raw, self.y_dummies