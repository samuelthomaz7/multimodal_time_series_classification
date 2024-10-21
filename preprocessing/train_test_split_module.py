from sklearn.model_selection import train_test_split


class TrainTestSplit():

    def __init__(self, X_raw, y_raw, metadata = None, random_state = 42):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.metadata = metadata
        self.step_name = 'TrainTestSplit'
        self.random_state = random_state

    
    def fit(self):
        pass

    def transform(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_raw, self.y_raw, random_state=self.random_state, train_size= 0.8)
        return self.X_train, self.X_test, self.y_train, self.y_test