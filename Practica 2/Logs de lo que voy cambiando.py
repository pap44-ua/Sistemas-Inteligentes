import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow import keras

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


# Clase AdaboostBinario adaptada
class AdaboostBinario:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump(n_features)
            predictions, error = clf.fit(X, y, w)

            if error > 0.5:
                error = 1 - error
                clf.polarity *= -1

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            clf.alpha = alpha
            self.clfs.append(clf)

    def predict_values(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_clf))

        for i, clf in enumerate(self.clfs):
            predictions[:, i] = clf.alpha * clf.predict(X)

        return predictions

    def predict(self, X):
        predictions = self.predict_values(X)
        return np.argmax(predictions, axis=1)


# Función para cargar MNIST y adaptada para AdaboostBinario multiclase
def load_MNIST_for_adaboost_multiclass(subset_size=None):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    if subset_size is not None:
        X_train = X_train[:subset_size]
        Y_train = Y_train[:subset_size]
        X_test = X_test[:subset_size]
        Y_test = Y_test[:subset_size]
    return X_train, Y_train, X_test, Y_test


# Función para cargar MNIST y adaptada para AdaboostBinario
def load_MNIST_for_adaboost(subset_size=None):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    if subset_size is not None:
        X_train = X_train[:subset_size]
        Y_train = Y_train[:subset_size]
        X_test = X_test[:subset_size]
        Y_test = Y_test[:subset_size]
    return X_train, Y_train, X_test, Y_test

# Función para tareas 1A y 1B adaptada para AdaboostBinario
def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=False):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(subset_size=1000)
    Y_train_bin = np.where(Y_train == clase, 1, -1)
    Y_test_bin = np.where(Y_test == clase, 1, -1)

    adaboost = AdaboostBinario(n_clf=T)
    adaboost.fit(X_train, Y_train_bin)
    train_accuracy = np.mean(adaboost.predict(X_train) == Y_train_bin)
    test_accuracy = np.mean(adaboost.predict(X_test) == Y_test_bin)

    if verbose:
        print(f"Tasas acierto (train, test): {train_accuracy * 100:.2f}%, {test_accuracy * 100:.2f}%")
    return train_accuracy, test_accuracy


# Función para experimentar con diferentes valores de T y A
def experimenta_T_A():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(subset_size=1000)
    Y_train_bin = np.where(Y_train == 5, 1, -1)
    Y_test_bin = np.where(Y_test == 5, 1, -1)

    T_values = [5, 10, 15, 20, 25, 30]  # Valores posibles para T
    A_values = [int(900 // T) for T in T_values if 900 % T == 0]  # Valores posibles para A, asegurando T * A <= 900

    best_accuracy = 0.0
    optimal_T = None
    optimal_A = None

    for T in T_values:
        for A in A_values:
            start_time = time.time()
            adaboost = AdaboostBinario(n_clf=T)
            adaboost.fit(X_train, Y_train_bin)
            end_time = time.time()
            y_test_pred = adaboost.predict(X_test)
            test_accuracy = np.mean(y_test_pred == Y_test_bin)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                optimal_T = T
                optimal_A = A

            print(f"T={T}, A={A}: Test accuracy={test_accuracy}, Time={end_time - start_time}")

    print(f"\nOptimal T={optimal_T}, A={optimal_A} with Best Test accuracy={best_accuracy}")

    return optimal_T, optimal_A


# Función para entrenar y evaluar el clasificador Adaboost multiclase
def tareas_1D_adaboost_multiclase(T, A):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost_multiclass(subset_size=1000)

    n_classes = 10
    adaboost_classifiers = []
    train_accuracies = []
    test_accuracies = []

    for clase in range(n_classes):
        Y_train_bin = np.where(Y_train == clase, 1, -1)
        Y_test_bin = np.where(Y_test == clase, 1, -1)

        adaboost = AdaboostBinario(n_clf=T)
        adaboost.fit(X_train, Y_train_bin)

        adaboost_classifiers.append(adaboost)

        train_accuracy = np.mean(adaboost.predict(X_train) == clase)
        test_accuracy = np.mean(adaboost.predict(X_test) == clase)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Clase {clase}: Train Accuracy={train_accuracy * 100:.2f}%, Test Accuracy={test_accuracy * 100:.2f}%")

    mean_train_accuracy = np.mean(train_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)

    print(f"\nMean Train Accuracy across all classes: {mean_train_accuracy * 100:.2f}%")
    print(f"Mean Test Accuracy across all classes: {mean_test_accuracy * 100:.2f}%")

    return mean_test_accuracy


if __name__ == "__main__":
    import time

    # Entrenamiento y evaluación de Adaboost binario para la clase 5
    print("Adaboost binario para la clase 5:")
    tareas_1A_y_1B_adaboost_binario(clase=5, T=10, A=10)

    # Experimentación con diferentes valores de T y A
    print("\nExperimentación con diferentes valores de T y A:")
    experimenta_T_A()

    # Entrenamiento y evaluación del clasificador Adaboost multiclase
    print("\nClasificador Adaboost multiclase para MNIST:")
    tareas_1D_adaboost_multiclase(T=10, A=10)

-------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


# Clase AdaboostBinario adaptada
class AdaboostBinario:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump(n_features)
            predictions, error = clf.fit(X, y, w)

            if error > 0.5:
                error = 1 - error
                clf.polarity *= -1

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            clf.alpha = alpha
            self.clfs.append(clf)

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
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X_train, Y_train):
        self.clfs = []
        n_classes = len(np.unique(Y_train))

        for i in range(n_classes):
            print(f"Training Adaboost for class {i}...")
            Y_train_bin = np.where(Y_train == i, 1, -1)
            adaboost = AdaboostBinario(n_clf=self.n_clf)
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


# Función para entrenar y evaluar AdaboostMulticlase
def train_and_evaluate_multiclass(T=10):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    adaboost_multiclass = AdaboostMulticlase(n_clf=T)
    adaboost_multiclass.fit(X_train, Y_train)

    Y_pred = adaboost_multiclass.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    

    print(f"Test Accuracy of AdaboostMulticlase with T={T}: {test_accuracy * 100:.2f}%")
    

    return test_accuracy


# Función para experimentar con diferentes valores de T
def experimenta_T_multiclase(T_values):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    accuracies = []
    

    for T in T_values:
        print(f"\nExperimentando con T={T}...")
        acc = train_and_evaluate_multiclass(T)
        accuracies.append(acc)
        

    return accuracies

def load_MNIST_for_multiclass_with_PCA(n_components=None):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca = X_train
        X_test_pca = X_test

    return X_train_pca, Y_train, X_test_pca, Y_test


# Función para entrenar y evaluar AdaboostMulticlase con PCA
def train_and_evaluate_multiclass_with_PCA(T=10, n_components=None):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass_with_PCA(n_components=n_components)

    adaboost_multiclass = AdaboostMulticlase(n_clf=T)
    adaboost_multiclass.fit(X_train, Y_train)

    Y_pred = adaboost_multiclass.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Test Accuracy of AdaboostMulticlase with T={T} and PCA components={n_components}: {test_accuracy * 100:.2f}%")

    return test_accuracy


# Función para experimentar con diferentes valores de T y PCA components
def experimenta_T_PCA(T_values, pca_components):
    accuracies = []

    for n_components in pca_components:
        for T in T_values:
            print(f"\nExperimentando con T={T} y PCA components={n_components}...")
            acc = train_and_evaluate_multiclass_with_PCA(T, n_components)
            accuracies.append(acc)

    return accuracies



if __name__ == "__main__":
    # Entrenamiento y evaluación de Adaboost multiclase
    print("\nEntrenamiento y evaluación de Adaboost multiclase:")
    train_and_evaluate_multiclass(T=10)

    # Experimentación con diferentes valores de T
    T_values = [5, 10, 15, 20, 25]
    print("\nExperimentación con diferentes valores de T:")
    experimenta_T_multiclase(T_values)
    
    
     # Entrenamiento y evaluación de Adaboost multiclase con PCA
    print("\nEntrenamiento y evaluación de Adaboost multiclase con PCA:")
    train_and_evaluate_multiclass_with_PCA(T=10, n_components=50)  # Ejemplo con 50 componentes PCA

    # Experimentación con diferentes valores de T y PCA components
    T_values = [5, 10, 15, 20, 25]
    pca_components = [20, 30, 40, 50]  # Ejemplo de diferentes números de componentes PCA
    print("\nExperimentación con diferentes valores de T y PCA components:")
    experimenta_T_PCA(T_values, pca_components)


#Va bastante bien la verdad -> el de arriba de la linea
    
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Siento que va rarete

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

    # Experimentación con diferentes valores de T y porcentajes de división entre entrenamiento y validación
    T_values = [5, 10, 15, 20, 25]
    validation_sizes = [0.1, 0.2, 0.3]  # Ejemplo de diferentes tamaños de validación
    early_stopping_rounds = 3
    print("\nExperimentación con diferentes valores de T, validation_size y detección de sobreentrenamiento:")
    experimenta_T_multiclase(T_values, validation_sizes, early_stopping_rounds)

--------------------------------------------------------------------------------------------------------------------------------------------------
#Esta va bn

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


# Clase AdaboostBinario adaptada
class AdaboostBinario:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump(n_features)
            predictions, error = clf.fit(X, y, w)

            if error > 0.5:
                error = 1 - error
                clf.polarity *= -1

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            clf.alpha = alpha
            self.clfs.append(clf)

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
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X_train, Y_train):
        self.clfs = []
        n_classes = len(np.unique(Y_train))

        for i in range(n_classes):
            print(f"Training Adaboost for class {i}...")
            Y_train_bin = np.where(Y_train == i, 1, -1)
            adaboost = AdaboostBinario(n_clf=self.n_clf)
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


# Función para entrenar y evaluar AdaboostMulticlase
def train_and_evaluate_multiclass(T=10):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    adaboost_multiclass = AdaboostMulticlase(n_clf=T)
    adaboost_multiclass.fit(X_train, Y_train)

    Y_pred = adaboost_multiclass.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    

    print(f"Test Accuracy of AdaboostMulticlase with T={T}: {test_accuracy * 100:.2f}%")
    

    return test_accuracy


# Función para experimentar con diferentes valores de T
def experimenta_T_multiclase(T_values):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass()

    accuracies = []
    

    for T in T_values:
        print(f"\nExperimentando con T={T}...")
        acc = train_and_evaluate_multiclass(T)
        accuracies.append(acc)
        

    return accuracies

def load_MNIST_for_multiclass_with_PCA(n_components=None):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca = X_train
        X_test_pca = X_test

    return X_train_pca, Y_train, X_test_pca, Y_test


# Función para entrenar y evaluar AdaboostMulticlase con PCA
def train_and_evaluate_multiclass_with_PCA(T=10, n_components=None):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_multiclass_with_PCA(n_components=n_components)

    adaboost_multiclass = AdaboostMulticlase(n_clf=T)
    adaboost_multiclass.fit(X_train, Y_train)

    Y_pred = adaboost_multiclass.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Test Accuracy of AdaboostMulticlase with T={T} and PCA components={n_components}: {test_accuracy * 100:.2f}%")

    return test_accuracy


# Función para experimentar con diferentes valores de T y PCA components
def experimenta_T_PCA(T_values, pca_components):
    accuracies = []

    for n_components in pca_components:
        for T in T_values:
            print(f"\nExperimentando con T={T} y PCA components={n_components}...")
            acc = train_and_evaluate_multiclass_with_PCA(T, n_components)
            accuracies.append(acc)

    return accuracies



if __name__ == "__main__":
    # Entrenamiento y evaluación de Adaboost multiclase
    print("\nEntrenamiento y evaluación de Adaboost multiclase:")
    train_and_evaluate_multiclass(T=10)

    # Experimentación con diferentes valores de T
    T_values = [5, 10, 15, 20, 25]
    print("\nExperimentación con diferentes valores de T:")
    experimenta_T_multiclase(T_values)
    
    
     # Entrenamiento y evaluación de Adaboost multiclase con PCA
    print("\nEntrenamiento y evaluación de Adaboost multiclase con PCA:")
    train_and_evaluate_multiclass_with_PCA(T=10, n_components=50)  # Ejemplo con 50 componentes PCA

    # Experimentación con diferentes valores de T y PCA components
    T_values = [5, 10, 15, 20, 25]
    pca_components = [20, 30, 40, 50]  # Ejemplo de diferentes números de componentes PCA
    print("\nExperimentación con diferentes valores de T y PCA components:")
    experimenta_T_PCA(T_values, pca_components)

#Esta es anterior a la de arriba
    
    ------------------------------------------------------------------------------------------------------------------------------


