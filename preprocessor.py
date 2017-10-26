class EmptyPreprocessor:
    def preprocess_train(self, x_train):
        return x_train

    def preprocess_test(self, x_test):
        return x_test

class Preprocessor:

    def __init__(self, preprocess_train, preprocess_test):
        """Class constructor

        Keyword arguments:
        preprocess_train -- function used to preprocess x_train. This function should
            return dependencies used in preprocess_test
        preprocess_test -- function used to preprocess x_test. This function takes
            a matrix and dependencies as argument
        """
        self.preprocess_train_f = preprocess_train
        self.preprocess_test_f = preprocess_test
        self.can_preprocess_test = False

    def preprocess_train(self, x_train):
        processed_x_train, dependency = self.preprocess_train_f(x_train)
        self.test_process_dependency = dependency
        self.can_preprocess_test = True
        return processed_x_train

    def preprocess_test(self, x_test):
        if not self.can_preprocess_test:
            raise Exception('Trying to preprocess x_test before x_train')
        return self.preprocess_test_f(x_test, self.test_process_dependency)
