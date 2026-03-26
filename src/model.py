from sklearn.linear_model import LogisticRegression

class LoanModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000,
                                        class_weight='balanced')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)