from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000,
                                        class_weight='balanced')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
class SVMModel:
    def __init__(self):
        self.model = SVC(
                        class_weight='balanced', 
                        random_state=42                    
                    )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)