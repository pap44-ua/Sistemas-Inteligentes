import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Clase DecisionStump
class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = None
        self.threshold = None
        self.polarity = None
        
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = None
        self.polarity = 1 if np.random.rand() < 0.5 else -1

    def fit(self, X, Y, weights):
        feature_values = X[:, self.feature_index]
        self.threshold = np.median(feature_values)

        predictions = np.ones(len(X))
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1

        error = np.sum(weights * (predictions != Y)) / np.sum(weights)

        return predictions, error

    def predict(self, X):
        if self.threshold is None:
            raise RuntimeError("Threshold not defined. Fit the model before prediction.")

        predictions = np.ones(len(X))
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions


# Clase AdaboostBinario adaptada con detección de sobreentrenamiento
class AdaboostBinario:
    def __init__(self, n_clf=5, early_stopping_rounds=None, validation_size=0.2):
        self.n_clf = n_clf
        self.clfs = []
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=42)
        n_samples, n_features = X_train.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        best_val_accuracy = -1
        rounds_since_best = 0

        for _ in range(self.n_clf):
            clf = DecisionStump(n_features)
            predictions, error = clf.fit(X_train, y_train, w)

            if error > 0.5:
                error = 1 - error
                clf.polarity *= -1

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            w *= np.exp(-alpha * y_train * predictions)
            w /= np.sum(w)

            clf.alpha = alpha
            self.clfs.append(clf)

            # Evaluar en el conjunto de validación
            val_predictions = self.predict_values(X_val)
            val_accuracy = accuracy_score(y_val, np.argmax(val_predictions, axis=1))

            # Verificar si hay sobreentrenamiento
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                rounds_since_best = 0
            else:
                rounds_since_best += 1

            if self.early_stopping_rounds is not None and rounds_since_best >= self.early_stopping_rounds:
                print(f"Deteniendo entrenamiento debido a sobreentrenamiento detectado en la ronda {_}")
                break

    def predict_values(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_clf))

        for i, clf in enumerate(self.clfs):
            predictions[:, i] = clf.alpha * clf.predict(X)

        return predictions

    def predict(self, X):
        predictions = self.predict_values(X)
        return np.argmax(predictions, axis=1)


# Clase AdaboostMulticlase
class AdaboostMulticlase:
    def __init__(self, n_clf=5, early_stopping_rounds=None, validation_size=0.2):
        self.n_clf = n_clf
        self.clfs = []
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size

    def fit(self, X_train, Y_train):
        self.clfs = []
        n_classes = len(np.unique(Y_train))

        for i in range(n_classes):
            print(f"Training Adaboost for class {i}...")
            Y_train_bin = np.where(Y_train == i, 1, -1)
            adaboost = AdaboostBinario(n_clf=self.n_clf, early_stopping_rounds=self.early_stopping_rounds, validation_size=self.validation_size)
            adaboost.fit(X_train, Y_train_bin)
            self.clfs.append(adaboost)

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.clfs)
        predictions = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            predictions[:, i] = self.clfs[i].predict_values(X).sum(axis=1)  # Sumamos las predicciones ponderadas

        return np.argmax(predictions, axis=1)


# Función para cargar MNIST y adaptada para AdaboostMulticlase
def load_MNIST_for_multiclass():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    return X_train, Y_train, X_test, Y_test


# Función para entrenar y evaluar AdaboostMulticlase con detección de sobreentrenamiento
def train_and_evaluate_multiclass(T=10, validation_size=0.2, early_stopping_rounds=None):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    adaboost_multiclass = AdaboostMulticlase(n_clf=T, early_stopping_rounds=early_stopping_rounds, validation_size=validation_size)
    adaboost_multiclass.fit(X_train, Y_train)

    Y_pred = adaboost_multiclass.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Test Accuracy of AdaboostMulticlase with T={T}, validation_size={validation_size}, early_stopping_rounds={early_stopping_rounds}: {test_accuracy * 100:.2f}%")

    return test_accuracy


# Función para entrenar y evaluar el clasificador Adaboost con DecisionTreeClassifier (profundidad 0)
def train_and_evaluate_adaboost_with_decision_tree(T=10, max_pixels_per_iteration=None):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    # Parámetros relevantes para DecisionTreeClassifier con profundidad 0
    base_clf = DecisionTreeClassifier(max_depth=1)
    if max_pixels_per_iteration is not None:
        base_clf.max_features = max_pixels_per_iteration
    
    # Crear el clasificador Adaboost
    adaboost_clf = AdaBoostClassifier(estimator=base_clf, n_estimators=T, algorithm='SAMME')

    # Entrenar el clasificador Adaboost
    adaboost_clf.fit(X_train, Y_train)

    # Evaluar en el conjunto de prueba
    Y_pred = adaboost_clf.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Test Accuracy of AdaboostClassifier with DecisionTree (T={T}, max_pixels_per_iteration={max_pixels_per_iteration}): {test_accuracy * 100:.2f}%")

    return test_accuracy


# Función para experimentar con diferentes valores de T y porcentajes de división entre entrenamiento y validación
def experimenta_T_multiclase(T_values, validation_sizes, early_stopping_rounds=None):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    accuracies = []

    for validation_size in validation_sizes:
        for T in T_values:
            print(f"\nExperimentando con T={T}, validation_size={validation_size}...")
            acc = train_and_evaluate_multiclass(T, validation_size=validation_size, early_stopping_rounds=early_stopping_rounds)
            accuracies.append(acc)

    return accuracies

if __name__ == "__main__":
    # Entrenamiento y evaluación de Adaboost multiclase
    print("\nEntrenamiento y evaluación de Adaboost multiclase con detección de sobreentrenamiento:")
    train_and_evaluate_multiclass(T=10, validation_size=0.2, early_stopping_rounds=3)

    # Experimentación con diferentes valores de T y porcentajes de división entre entrenamiento y validación para Adaboost multiclase
    T_values = [5, 10, 15, 20, 25]
    validation_sizes = [0.1, 0.2, 0.3]
    early_stopping_rounds = 3
    print("\nExperimentación con diferentes valores de T, validation_size y detección de sobreentrenamiento:")
    experimenta_T_multiclase(T_values, validation_sizes, early_stopping_rounds)

    # Entrenamiento y evaluación de Adaboost con DecisionTreeClassifier (profundidad 0)
    print("\nEntrenamiento y evaluación de Adaboost con DecisionTreeClassifier (profundidad 0):")
    train_and_evaluate_adaboost_with_decision_tree(T=10)

   
