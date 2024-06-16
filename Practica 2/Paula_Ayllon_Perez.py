import logging, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Cargar los datos de MNIST
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)

def show_image(imagen, title):
    plt.figure()
    plt.suptitle(title)
    plt.imshow(imagen, cmap="Greys")
    plt.show()

for i in range(3):
    title = f"Mostrando imagen X_train[{i}] -- Y_train[{i}] = {Y_train[i]}"
    show_image(X_train[i], title)

def plot_X(X, title):
    plt.title(title)
    plt.plot(X)
    plt.show()

fila, columna = 10, 10
features_fila_col = X_train[:, fila, columna]
print(len(np.unique(features_fila_col)))
title = f"Valores en ({fila}, {columna})"
plot_X(features_fila_col, title)



class DecisionStump:
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1 if np.random.rand() > 0.5 else -1

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        predictions = np.ones(X.shape[0])
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions

class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.alphas = []
        self.classifiers = []

    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for t in range(self.T):
            min_error = float('inf')
            best_classifier = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                misclassified = w * (predictions != Y)
                error = np.sum(misclassified)

                if error < min_error:
                    min_error = error
                    best_classifier = stump

            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - min_error) / (min_error + EPS))
            predictions = best_classifier.predict(X)
            w *= np.exp(-alpha * Y * predictions)
            w /= np.sum(w)

            self.alphas.append(alpha)
            self.classifiers.append(best_classifier)

            if verbose:
                print(f'Classifier {t+1}/{self.T}, Error: {min_error}, Alpha: {alpha}')

    def predict(self, X):
        classifier_preds = np.array([alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.classifiers)])
        y_pred = np.sum(classifier_preds, axis=0)
        return np.sign(y_pred)

if __name__ == "__main__":
    # Convertir las etiquetas a {-1, 1}
    Y_train_binary = np.where(Y_train == 9, 1, -1)
    Y_test_binary = np.where(Y_test == 9, 1, -1)
    
    # Aplanar las imágenes de 28x28 a un vector de 784 características
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    
    # Entrenar el modelo Adaboost
    adaboost = Adaboost(T=10, A=10)
    adaboost.fit(X_train_flat, Y_train_binary, verbose=True)
    
    # Obtener las predicciones del modelo
    train_predictions = adaboost.predict(X_train_flat)
    test_predictions = adaboost.predict(X_test_flat)
    
    # Calcular la precisión
    train_accuracy = np.mean(train_predictions == Y_train_binary)
    test_accuracy = np.mean(test_predictions == Y_test_binary)
    
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
