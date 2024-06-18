#1A

import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaboostBinario:
    def __init__(self, T=5, A=20):
        self.T = T  # Número de iteraciones del Adaboost
        self.A = A  # Número de stump a probar en cada iteración
        self.alphas = []  # Pesos de los stumps
        self.stumps = []  # Lista de los stumps

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))  # Inicialización de los pesos

        for _ in range(self.T):
            min_error = float('inf')
            best_stump = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = sum(w[y != predictions])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump
            
            epsilon = 1e-10  # Para evitar división por cero
            alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            best_stump.alpha = alpha
            self.stumps.append(best_stump)

            predictions = best_stump.predict(X)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        y_pred = np.sign(np.sum(stump_preds, axis=0))
        return y_pred

def load_MNIST_for_adaboost():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

def balance_data(X, y, class_label):
    pos_idx = np.where(y == class_label)[0]
    neg_idx = np.where(y != class_label)[0]
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:len(pos_idx)]
    new_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(new_idx)
    return X[new_idx], np.where(y[new_idx] == class_label, 1, -1)

def train_and_evaluate_adaboost():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifiers = []
    for i in range(10):
        print(f"Entrenando clasificador para la clase {i}")
        X_balanced, y_balanced = balance_data(X_train, Y_train, i)
        classifier = AdaboostBinario(T=10, A=10)
        classifier.fit(X_balanced, y_balanced)
        classifiers.append(classifier)

    for i in range(10):
        print(f"Evaluando clasificador para la clase {i}")
        y_test_binary = np.where(Y_test == i, 1, -1)
        y_pred = classifiers[i].predict(X_test)
        accuracy = np.mean(y_pred == y_test_binary)
        print(f"Tasa de acierto para la clase {i}: {accuracy:.4f}")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(f"Matriz de confusión para la clase {i}:\n{cm}")

def main():
    train_and_evaluate_adaboost()

if __name__ == "__main__":
    main()

------------------------------------------------------------------------------------------------------------------------------

#1B

import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaboostBinario:
    def __init__(self, T=5, A=20):
        self.T = T  # Número de iteraciones del Adaboost
        self.A = A  # Número de stump a probar en cada iteración
        self.alphas = []  # Pesos de los stumps
        self.stumps = []  # Lista de los stumps

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))  # Inicialización de los pesos

        for _ in range(self.T):
            min_error = float('inf')
            best_stump = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = sum(w[y != predictions])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump
            
            epsilon = 1e-10  # Para evitar división por cero
            alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            best_stump.alpha = alpha
            self.stumps.append(best_stump)

            predictions = best_stump.predict(X)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        y_pred = np.sign(np.sum(stump_preds, axis=0))
        return y_pred

def load_MNIST_for_adaboost():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

def balance_data(X, y, class_label):
    pos_idx = np.where(y == class_label)[0]
    neg_idx = np.where(y != class_label)[0]
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:len(pos_idx)]
    new_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(new_idx)
    return X[new_idx], np.where(y[new_idx] == class_label, 1, -1)

def train_and_evaluate_adaboost():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifiers = []
    for i in range(10):
        print(f"Entrenando clasificador para la clase {i}")
        X_balanced, y_balanced = balance_data(X_train, Y_train, i)
        classifier = AdaboostBinario(T=10, A=10)
        classifier.fit(X_balanced, y_balanced)
        classifiers.append(classifier)

    for i in range(10):
        print(f"Evaluando clasificador para la clase {i}")
        y_test_binary = np.where(Y_test == i, 1, -1)
        y_pred = classifiers[i].predict(X_test)
        accuracy = np.mean(y_pred == y_test_binary)
        print(f"Tasa de acierto para la clase {i}: {accuracy:.4f}")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(f"Matriz de confusión para la clase {i}:\n{cm}")

def experiment_with_parameters():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    T_values = [1, 5, 10, 20]
    A_values = [1, 5, 10, 20]

    results_T_fixed = []
    results_A_fixed = []

    for T in T_values:
        accs = []
        times = []
        for _ in range(5):  # Ejecutar 5 veces para promediar
            start_time = time.time()
            classifier = AdaboostBinario(T=T, A=10)
            X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
            classifier.fit(X_balanced, y_balanced)
            y_test_binary = np.where(Y_test == 0, 1, -1)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test_binary, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_T_fixed.append((T, np.mean(accs), np.mean(times)))

    for A in A_values:
        accs = []
        times = []
        for _ in range(5):  # Ejecutar 5 veces para promediar
            start_time = time.time()
            classifier = AdaboostBinario(T=10, A=A)
            X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
            classifier.fit(X_balanced, y_balanced)
            y_test_binary = np.where(Y_test == 0, 1, -1)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test_binary, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_A_fixed.append((A, np.mean(accs), np.mean(times)))

    # Gráficas
    plt.figure(figsize=(12, 6))
    
    # T fijo y variando A
    plt.subplot(1, 2, 1)
    A_vals, accs, times = zip(*results_A_fixed)
    plt.plot(A_vals, accs, label='Accuracy')
    plt.plot(A_vals, times, label='Execution Time')
    plt.xlabel('A')
    plt.ylabel('Value')
    plt.title('T fixed at 10, varying A')
    plt.legend()

    # A fijo y variando T
    plt.subplot(1, 2, 2)
    T_vals, accs, times = zip(*results_T_fixed)
    plt.plot(T_vals, accs, label='Accuracy')
    plt.plot(T_vals, times, label='Execution Time')
    plt.xlabel('T')
    plt.ylabel('Value')
    plt.title('A fixed at 10, varying T')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Exploración de combinaciones
    best_accuracy = 0
    best_T = None
    best_A = None
    for T in range(1, 61):
        for A in range(1, 61):
            if T * A > 3600:
                continue
            accs = []
            for _ in range(5):
                classifier = AdaboostBinario(T=T, A=A)
                X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
                classifier.fit(X_balanced, y_balanced)
                y_test_binary = np.where(Y_test == 0, 1, -1)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test_binary, y_pred)
                accs.append(accuracy)
            avg_accuracy = np.mean(accs)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_T = T
                best_A = A

    print(f"Best combination T={best_T}, A={best_A} with accuracy={best_accuracy}")

def main():
    # Primero entrenamos y evaluamos los clasificadores
    train_and_evaluate_adaboost()
    # Luego hacemos los experimentos con los parámetros T y A
    experiment_with_parameters()

if __name__ == "__main__":
    main()
    
    ------------------------------------------------------ tbn es 1B
    import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

# Clase para el Decision Stump
class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

# Clase para el Adaboost binario
class AdaboostBinario:
    def __init__(self, T=5, A=20):
        self.T = T  # Número de iteraciones del Adaboost
        self.A = A  # Número de stump a probar en cada iteración
        self.alphas = []  # Pesos de los stumps
        self.stumps = []  # Lista de los stumps

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))  # Inicialización de los pesos

        for _ in range(self.T):
            min_error = float('inf')
            best_stump = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = sum(w[y != predictions])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump
            
            epsilon = 1e-10  # Para evitar división por cero
            alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            best_stump.alpha = alpha
            self.stumps.append(best_stump)

            predictions = best_stump.predict(X)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        y_pred = np.sign(np.sum(stump_preds, axis=0))
        return y_pred

# Cargar el dataset MNIST
def load_MNIST_for_adaboost():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

# Balancear los datos para una clase específica
def balance_data(X, y, class_label):
    pos_idx = np.where(y == class_label)[0]
    neg_idx = np.where(y != class_label)[0]
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:len(pos_idx)]
    new_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(new_idx)
    return X[new_idx], np.where(y[new_idx] == class_label, 1, -1)

# Entrenar y evaluar Adaboost para todas las clases
def train_and_evaluate_adaboost():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifiers = []
    for i in range(10):
        print(f"Entrenando clasificador para la clase {i}")
        X_balanced, y_balanced = balance_data(X_train, Y_train, i)
        classifier = AdaboostBinario(T=10, A=10)
        classifier.fit(X_balanced, y_balanced)
        classifiers.append(classifier)

    for i in range(10):
        print(f"Evaluando clasificador para la clase {i}")
        y_test_binary = np.where(Y_test == i, 1, -1)
        y_pred = classifiers[i].predict(X_test)
        accuracy = np.mean(y_pred == y_test_binary)
        print(f"Tasa de acierto para la clase {i}: {accuracy:.4f}")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(f"Matriz de confusión para la clase {i}:\n{cm}")

# Experimentar con los parámetros T y A y graficar los resultados
def experiment_with_parameters():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    T_values = [1, 5, 10, 20]
    A_values = [1, 5, 10, 20]

    results_T_fixed = []
    results_A_fixed = []

    for T in T_values:
        accs = []
        times = []
        for _ in range(5):  # Ejecutar 5 veces para promediar
            start_time = time.time()
            classifier = AdaboostBinario(T=T, A=10)
            X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
            classifier.fit(X_balanced, y_balanced)
            y_test_binary = np.where(Y_test == 0, 1, -1)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test_binary, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_T_fixed.append((T, np.mean(accs), np.mean(times)))

    for A in A_values:
        accs = []
        times = []
        for _ in range(5):  # Ejecutar 5 veces para promediar
            start_time = time.time()
            classifier = AdaboostBinario(T=10, A=A)
            X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
            classifier.fit(X_balanced, y_balanced)
            y_test_binary = np.where(Y_test == 0, 1, -1)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test_binary, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_A_fixed.append((A, np.mean(accs), np.mean(times)))

    # Gráficas
    plt.figure(figsize=(12, 6))
    
    # T fijo y variando A
    plt.subplot(1, 2, 1)
    A_vals, accs, times = zip(*results_A_fixed)
    plt.plot(A_vals, accs, label='Accuracy')
    plt.plot(A_vals, times, label='Execution Time')
    plt.xlabel('A')
    plt.ylabel('Value')
    plt.title('T fixed at 10, varying A')
    plt.legend()

    # A fijo y variando T
    plt.subplot(1, 2, 2)
    T_vals, accs, times = zip(*results_T_fixed)
    plt.plot(T_vals, accs, label='Accuracy')
    plt.plot(T_vals, times, label='Execution Time')
    plt.xlabel('T')
    plt.ylabel('Value')
    plt.title('A fixed at 10, varying T')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Exploración de combinaciones
    best_accuracy = 0
    best_T = None
    best_A = None
    for T in range(1, 61):
        for A in range(1, 61):
            if T * A > 3600:
                continue
            accs = []
            for _ in range(5):
                classifier = AdaboostBinario(T=T, A=A)
                X_balanced, y_balanced = balance_data(X_train, Y_train, 0)
                classifier.fit(X_balanced, y_balanced)
                y_test_binary = np.where(Y_test == 0, 1, -1)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test_binary, y_pred)
                accs.append(accuracy)
            avg_accuracy = np.mean(accs)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_T = T
                best_A = A

    print(f"Best combination T={best_T}, A={best_A} with accuracy={best_accuracy}")

# Función principal que ejecuta ambos experimentos
def main():
    # Primero entrenamos y evaluamos los clasificadores (Parte 1A)
    train_and_evaluate_adaboost()
    # Luego hacemos los experimentos con los parámetros T y A (Parte 1B)
    experiment_with_parameters()

if __name__ == "__main__":
    main()


-----------------------------------------1C-----------------------------------------------------------------
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

# Clase para el Decision Stump
class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

# Clase para el Adaboost binario
class AdaboostBinario:
    def __init__(self, T=5, A=20):
        self.T = T  # Número de iteraciones del Adaboost
        self.A = A  # Número de stump a probar en cada iteración
        self.alphas = []  # Pesos de los stumps
        self.stumps = []  # Lista de los stumps

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))  # Inicialización de los pesos

        for _ in range(self.T):
            min_error = float('inf')
            best_stump = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = sum(w[y != predictions])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump
            
            epsilon = 1e-10  # Para evitar división por cero
            alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            best_stump.alpha = alpha
            self.stumps.append(best_stump)

            predictions = best_stump.predict(X)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        y_pred = np.sign(np.sum(stump_preds, axis=0))
        return y_pred

# Clase para el Adaboost multiclase
class AdaboostMulticlase:
    def __init__(self, T=10, A=10):
        self.T = T
        self.A = A
        self.classifiers = []

    def fit(self, X, y):
        self.classifiers = []
        for i in range(10):
            print(f"Entrenando clasificador para la clase {i}")
            X_balanced, y_balanced = self.balance_data(X, y, i)
            classifier = AdaboostBinario(T=self.T, A=self.A)
            classifier.fit(X_balanced, y_balanced)
            self.classifiers.append(classifier)

    def predict(self, X):
        clf_preds = np.array([classifier.predict(X) for classifier in self.classifiers])
        return np.argmax(clf_preds, axis=0)
    
    def balance_data(self, X, y, class_label):
        pos_idx = np.where(y == class_label)[0]
        neg_idx = np.where(y != class_label)[0]
        np.random.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]
        new_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(new_idx)
        return X[new_idx], np.where(y[new_idx] == class_label, 1, -1)

# Cargar el dataset MNIST
def load_MNIST_for_adaboost():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

# Función para entrenar y evaluar el Adaboost multiclase
def train_and_evaluate_adaboost_multiclase():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifier = AdaboostMulticlase(T=10, A=10)
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Tasa de acierto para el clasificador multiclase: {accuracy:.4f}")
    cm = confusion_matrix(Y_test, y_pred)
    print(f"Matriz de confusión para el clasificador multiclase:\n{cm}")

# Función de experimentación para Adaboost multiclase
def experiment_with_parameters_multiclase():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    T_values = [1, 5, 10, 20]
    A_values = [1, 5, 10, 20]

    results_T_fixed = []
    results_A_fixed = []

    for T in T_values:
        accs = []
        times = []
        for _ in range(3):  # Ejecutar 3 veces para promediar
            start_time = time.time()
            classifier = AdaboostMulticlase(T=T, A=10)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_T_fixed.append((T, np.mean(accs), np.mean(times)))

    for A in A_values:
        accs = []
        times = []
        for _ in range(3):  # Ejecutar 3 veces para promediar
            start_time = time.time()
            classifier = AdaboostMulticlase(T=10, A=A)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
            times.append(time.time() - start_time)
        results_A_fixed.append((A, np.mean(accs), np.mean(times)))

    # Gráficas
    plt.figure(figsize=(12, 6))
    
    # T fijo y variando A
    plt.subplot(1, 2, 1)
    A_vals, accs, times = zip(*results_A_fixed)
    plt.plot(A_vals, accs, label='Accuracy')
    plt.plot(A_vals, times, label='Execution Time')
    plt.xlabel('A')
    plt.ylabel('Value')
    plt.title('T fixed at 10, varying A (Multiclase)')
    plt.legend()

    # A fijo y variando T
    plt.subplot(1, 2, 2)
    T_vals, accs, times = zip(*results_T_fixed)
    plt.plot(T_vals, accs, label='Accuracy')
    plt.plot(T_vals, times, label='Execution Time')
    plt.xlabel('T')
    plt.ylabel('Value')
    plt.title('A fixed at 10, varying T (Multiclase)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Función principal que ejecuta el experimento
def main():
    
    # Primero entrenamos y evaluamos los clasificadores (Parte 1A)
    #train_and_evaluate_adaboost()
    # Luego hacemos los experimentos con los parámetros T y A (Parte 1B)
    #experiment_with_parameters()
    #Parte 1C
    train_and_evaluate_adaboost_multiclase()
    experiment_with_parameters_multiclase()
    
    print("FIN")

if __name__ == "__main__":
    main()



