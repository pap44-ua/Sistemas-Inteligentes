import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column >= self.threshold] = -1
        return predictions

class AdaboostBinario:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    error = sum(w[y != predictions])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)

class AdaboostMulticlase:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.models = []

    def fit(self, X, y):
        self.models = []
        for clase in np.unique(y):
            y_bin = np.where(y == clase, 1, -1)
            model = AdaboostBinario(n_clf=self.n_clf)
            model.fit(X, y_bin)
            self.models.append(model)

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        return np.argmax(model_preds, axis=0)

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

def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=True):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(subset_size=1000)
    Y_train_bin = np.where(Y_train == clase, 1, -1)
    Y_test_bin = np.where(Y_test == clase, 1, -1)

    adaboost = AdaboostBinario(n_clf=T)
    print("Starting training Adaboost...")
    adaboost.fit(X_train, Y_train_bin)
    print("Finished training Adaboost.")
    train_accuracy = np.mean(adaboost.predict(X_train) == Y_train_bin)
    test_accuracy = np.mean(adaboost.predict(X_test) == Y_test_bin)

    if verbose:
        print(f"Tasas acierto (train, test): {train_accuracy * 100:.2f}%, {test_accuracy * 100:.2f}%")
    return train_accuracy, test_accuracy

def experimenta_T_A():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(subset_size=1000)
    Y_train_bin = np.where(Y_train == 5, 1, -1)
    Y_test_bin = np.where(Y_test == 5, 1, -1)

    T_values = [5, 10, 15, 20]
    A_values = [10, 20, 30, 40]
    train_accuracies = []
    test_accuracies = []
    times = []

    for T in T_values:
        for A in A_values:
            start_time = time.time()
            adaboost = AdaboostBinario(n_clf=T)
            print(f"Training Adaboost with T={T}, A={A}...")
            adaboost.fit(X_train, Y_train_bin)
            end_time = time.time()
            y_train_pred = adaboost.predict(X_train)
            y_test_pred = adaboost.predict(X_test)
            train_accuracy = np.mean(y_train_pred == Y_train_bin)
            test_accuracy = np.mean(y_test_pred == Y_test_bin)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            times.append(end_time - start_time)
            print(f"T={T}, A={A}: Train accuracy={train_accuracy}, Test accuracy={test_accuracy}, Time={end_time - start_time}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(times, train_accuracies, label='Train Accuracy')
    plt.plot(times, test_accuracies, label='Test Accuracy')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(T_values, times, label='Time')
    plt.xlabel('T')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.show()

def train_and_evaluate_multiclase():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(subset_size=1000)

    adaboost_multiclase = AdaboostMulticlase(n_clf=10)
    print("Starting training Adaboost Multiclase...")
    adaboost_multiclase.fit(X_train, Y_train)
    print("Finished training Adaboost Multiclase.")

    y_train_pred = adaboost_multiclase.predict(X_train)
    y_test_pred = adaboost_multiclase.predict(X_test)

    train_accuracy = np.mean(y_train_pred == Y_train)
    test_accuracy = np.mean(y_test_pred == Y_test)
    print(f"Tasa de acierto en entrenamiento: {train_accuracy}, Tasa de acierto en test: {test_accuracy}")

if __name__ == "__main__":
    import time

    # Entrenamiento y evaluación de Adaboost binario para la clase 5
    print("Adaboost binario para la clase 5:")
    tareas_1A_y_1B_adaboost_binario(clase=5, T=10, A=10)

    # Experimentación con diferentes valores de T y A
    print("\nExperimentación con diferentes valores de T y A:")
    experimenta_T_A()

    # Entrenamiento y evaluación de Adaboost multiclase
    print("\nAdaboost multiclase:")
    train_and_evaluate_multiclase()
