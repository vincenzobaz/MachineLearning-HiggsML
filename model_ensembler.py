import numpy as np

class Model_Ensembler:

    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model
        self.can_predict = False

    def train(self, y_train, x_train):
        x_tr = x_train.copy()

        # Split data into two parts
        half_index = x_tr.shape[0] / 2
        x_train_half_1 = x_tr[:half_index]
        x_train_half_2 = x_tr[half_index:]

        # Train first stage models
        for model in self.models:
            model.train(y_train[half_index:], x_train_half_1)

        # Predict values on second half of the train data
        stage_0_predictions = np.array([model.predict(x_train_half_2) for model in self.models]).T

        # Feed those predictions to the meta model
        self.meta_model.train(y_train[:half_index], stage_0_predictions)

        self.can_predict = True

    def predict(self, x_test):
        if not self.can_predict:
            raise Exception('Trying to predict before training')

        stage_0_predictions = np.array([model.predict(x_test) for model in self.models]).T
        meta_predictions = self.meta_model.predict(stage_0_predictions)

        return meta_predictions

    def predict_label(self, x_test):
        if not self.can_predict:
            raise Exception('Trying to predict before training')

        stage_0_predictions = np.array([model.predict(x_test) for model in self.models]).T
        meta_predictions = self.meta_model.predict_label(stage_0_predictions)

        return meta_predictions
