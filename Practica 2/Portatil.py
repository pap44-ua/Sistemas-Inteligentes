import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

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
            predictions[X_column >= self.threshold] = -1
        return predictions

# Clase para el Adaboost binario
class AdaboostBinario:
    def __init__(self, T=50, A=70):  # Ajustado para que T*A <= 3600
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

# Nueva clase AdaboostBinario con detección de sobreentrenamiento
class AdaboostBinarioConDeteccion(AdaboostBinario):
    def __init__(self, T=50, A=70):
        super().__init__(T, A)

    def fit(self, X, y):
        # Dividir en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_samples, n_features = X_train.shape
        w = np.full(n_samples, (1 / n_samples))  # Inicialización de los pesos
        
        best_accuracy = 0
        best_stumps = []
        best_alphas = []

        for t in range(self.T):
            min_error = float('inf')
            best_stump = None
            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X_train)
                error = sum(w[y_train != predictions])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump
            
            epsilon = 1e-10  # Para evitar división por cero
            alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            best_stump.alpha = alpha
            self.stumps.append(best_stump)

            predictions = best_stump.predict(X_train)
            w *= np.exp(-alpha * y_train * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)

            # Validar el modelo
            y_val_pred = self.predict(X_val)
            accuracy = accuracy_score(y_val, y_val_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_stumps = self.stumps.copy()
                best_alphas = self.alphas.copy()
            else:
                print(f"Deteniendo el entrenamiento en la iteración {t+1} debido a sobreentrenamiento")
                self.stumps = best_stumps
                self.alphas = best_alphas
                break


# Clase para el Adaboost multiclase
class AdaboostMulticlase:
    def __init__(self, T=50, A=70):  # Ajustado para que T*A <= 3600
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

# Función para balancear los datos (extraída para reutilización)
def balance_data(X, y, class_label):
    pos_idx = np.where(y == class_label)[0]
    neg_idx = np.where(y != class_label)[0]
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:len(pos_idx)]
    new_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(new_idx)
    return X[new_idx], np.where(y[new_idx] == class_label, 1, -1)

# Cargar el dataset MNIST
def load_MNIST_for_adaboost():
    print("Cargando el dataset MNIST")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print("Dataset MNIST cargado")
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

# Función para entrenar y evaluar el Adaboost multiclase
def train_and_evaluate_adaboost_multiclase():
    print("Entrenando y evaluando Adaboost multiclase")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifier = AdaboostMulticlase(T=50, A=70)  # Ajustado para que T*A <= 3600
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Tasa de acierto para el clasificador multiclase: {accuracy:.4f}")
    cm = confusion_matrix(Y_test, y_pred)
    print(f"Matriz de confusión para el clasificador multiclase:\n{cm}")

# Función de experimentación para Adaboost multiclase
def experiment_with_parameters_multiclase():
    print("Experimentando con parámetros multiclase")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    T_values = [  70, 90]  # Ajustados para que T*A <= 3600
    A_values = [10, 20 ]  # Ajustados para que T*A <= 3600

    results_T_fixed = []
    results_A_fixed = []

    for T in T_values:
        accs = []
        for _ in range(3):  # Ejecutar 3 veces para promediar
            classifier = AdaboostMulticlase(T=T, A=20)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
        results_T_fixed.append((T, np.mean(accs)))

    for A in A_values:
        accs = []
        for _ in range(5):  # Ejecutar 3 veces para promediar
            classifier = AdaboostMulticlase(T=90, A=A)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
        results_A_fixed.append((A, np.mean(accs)))

    # Gráficas de Accuracy
    plt.figure(figsize=(12, 6))
    
    # T fijo y variando A
    plt.subplot(1, 2, 1)
    A_vals, accs = zip(*results_A_fixed)
    plt.plot(A_vals, accs, label='Precision')
    plt.xlabel('A')
    plt.ylabel('Precision')
    plt.title('T ajustado a 50, variando A (Multiclase)')
    plt.legend()

    # A fijo y variando T
    plt.subplot(1, 2, 2)
    T_vals, accs = zip(*results_T_fixed)
    plt.plot(T_vals, accs, label='Precision')
    plt.xlabel('T')
    plt.ylabel('Precision')
    plt.title('A ajustado a 50, variando T (Multiclase)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_and_evaluate_sklearn_adaboost(X_train, y_train, X_test, y_test, n_estimators=50, max_depth=1, max_features=None):
    """
    Entrena un clasificador AdaboostClassifier utilizando DecisionTreeClassifier con la configuración dada
    y evalúa su rendimiento en el conjunto de prueba.
    """
    # Crear el clasificador débil (Decision Stump)
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    
    # Crear el clasificador Adaboost usando el algoritmo SAMME
    adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, algorithm='SAMME')
    
    # Entrenar el clasificador
    adaboost.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = adaboost.predict(X_test)
    
    # Calcular y devolver la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy

def experiment_with_sklearn_adaboost_parameters(X_train, y_train, X_test, y_test):
    """
    Experimenta con diferentes configuraciones de parámetros para el clasificador Adaboost
    y busca la mejor tasa de acierto posible.
    """
    best_accuracy = 0
    best_params = {}
    
    # Experimentar con diferentes valores de n_estimators y max_depth
    for n_estimators in [10, 50, 100]:
        for max_depth in [1]:  # max_depth=1 para simular un Decision Stump
            for max_features in [None, 'sqrt', 'log2']:
                print(f"Evaluating: n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}")
                accuracy = train_and_evaluate_sklearn_adaboost(X_train, y_train, X_test, y_test,
                                                               n_estimators=n_estimators, max_depth=max_depth,
                                                               max_features=max_features)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'max_features': max_features
                    }
    
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Parameters: {best_params}")
    return best_params


# Aplicar PCA para reducir dimensionalidad
def apply_pca(X_train, X_test, n_components=50):
    print("Aplicando PCA")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("PCA aplicado")
    return X_train_pca, X_test_pca

# Función para entrenar y evaluar el Adaboost multiclase con PCA
def train_and_evaluate_adaboost_multiclase_pca():
    print("Entrenando y evaluando Adaboost multiclase con PCA")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=50)
    
    classifier = AdaboostMulticlase(T=50, A=70)  # Ajustado para que T*A <= 3600
    classifier.fit(X_train_pca, Y_train)

    y_pred = classifier.predict(X_test_pca)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Tasa de acierto para el clasificador multiclase con PCA: {accuracy:.4f}")
    cm = confusion_matrix(Y_test, y_pred)
    print(f"Matriz de confusión para el clasificador multiclase con PCA:\n{cm}")

# Función de experimentación para Adaboost multiclase con PCA
def experiment_with_parameters_multiclase_pca():
    print("Experimentando con parámetros multiclase con PCA")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=50)

    T_values = [10, 40, 70]  # Ajustados para que T*A <= 3600
    A_values = [10, 20, 30]  # Ajustados para que T*A <= 3600

    results_T_fixed = []
    results_A_fixed = []

    for T in T_values:
        accs = []
        for _ in range(3):  # Ejecutar 3 veces para promediar
            classifier = AdaboostMulticlase(T=T, A=50)
            classifier.fit(X_train_pca, Y_train)
            y_pred = classifier.predict(X_test_pca)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
        results_T_fixed.append((T, np.mean(accs)))

    for A in A_values:
        accs = []
        for _ in range(5):  # Ejecutar 3 veces para promediar
            classifier = AdaboostMulticlase(T=50, A=A)
            classifier.fit(X_train_pca, Y_train)
            y_pred = classifier.predict(X_test_pca)
            accuracy = accuracy_score(Y_test, y_pred)
            accs.append(accuracy)
        results_A_fixed.append((A, np.mean(accs)))

    # Gráficas
    plt.figure(figsize=(12, 6))
    
    # T fijo y variando A
    plt.subplot(1, 2, 1)
    A_vals, accs = zip(*results_A_fixed)
    plt.plot(A_vals, accs, label='Precision')
    plt.xlabel('A')
    plt.ylabel('Precision')
    plt.title('T ajustado a 50, variando A (Multiclase with PCA)')
    plt.legend()

    # A fijo y variando T
    plt.subplot(1, 2, 2)
    T_vals, accs = zip(*results_T_fixed)
    plt.plot(T_vals, accs, label='Precision')
    plt.xlabel('T')
    plt.ylabel('Precision')
    plt.title('A ajustado a 50, variando T (Multiclase with PCA)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_and_evaluate_adaboost_multiclase_stop_overfitting():
    print("Entrenando y evaluando Adaboost multiclase con detección de sobreentrenamiento")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    classifier = AdaboostMulticlase(T=50, A=90)  # Ajustado para que T*A <= 3600
    classifier.classifiers = [AdaboostBinarioConDeteccion(T=50, A=90) for _ in range(10)]
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Tasa de acierto para el clasificador multiclase con detección de sobreentrenamiento: {accuracy:.4f}")
    cm = confusion_matrix(Y_test, y_pred)
    print(f"Matriz de confusión para el clasificador multiclase con detección de sobreentrenamiento:\n{cm}")


def experiment_with_parameters_stop_overfitting():
    print("Experimentando con parámetros para detección de sobreentrenamiento")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    T_values = [ 40, 90]  # Ajustados para que T*A <= 3600
    validation_ratios = [0.1,0.3, 0.4,0.5]

    results = []

    for validation_ratio in validation_ratios:
        for T in T_values:
            accs = []
            for _ in range(5):  # Ejecutar 3 veces para promediar
                classifier = AdaboostMulticlase(T=T, A=50)
                classifier.classifiers = [AdaboostBinarioConDeteccion(T=T, A=50) for _ in range(10)]
                classifier.fit(X_train, Y_train)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(Y_test, y_pred)
                accs.append(accuracy)
            results.append((validation_ratio, T, np.mean(accs)))

    # Gráficas
    plt.figure(figsize=(12, 6))

    for validation_ratio in validation_ratios:
        subset = [(T, acc) for (vr, T, acc) in results if vr == validation_ratio]
        T_vals, accs = zip(*subset)
        plt.plot(T_vals, accs, label=f'Ratio de validacion de precision {validation_ratio}')

    plt.xlabel('T')
    plt.ylabel('Precision')
    plt.title('Variando T para diferentes ratios')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_and_evaluate_sklearn_adaboost_deep_tree(X_train, y_train, X_test, y_test, n_estimators=50, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None):
    """
    Entrena un clasificador AdaboostClassifier utilizando DecisionTreeClassifier con la configuración dada
    y evalúa su rendimiento en el conjunto de prueba.
    """
    print(f"Entrenando AdaboostClassifier con n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, max_features={max_features}")
    
    # Crear el clasificador débil (árbol de decisión con profundidad > 0)
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
    
    # Crear el clasificador Adaboost usando el algoritmo SAMME
    adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, algorithm='SAMME')
    
    # Entrenar el clasificador
    print("Entrenando el clasificador...")
    adaboost.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    print("Prediciendo en el conjunto de prueba...")
    y_pred = adaboost.predict(X_test)
    
    # Calcular y devolver la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión obtenida: {accuracy}")
    return accuracy

def experiment_with_deep_tree_parameters(X_train, y_train, X_test, y_test):
    """
    Experimenta con diferentes configuraciones de parámetros para el clasificador Adaboost utilizando árboles de decisión con profundidad > 0
    y busca la mejor tasa de acierto posible.
    """
    print("Comenzando experimentos con diferentes configuraciones de parámetros para AdaboostClassifier con árboles de decisión profundos")
    best_accuracy = 0
    best_params = {}
    
    # Experimentar con diferentes valores de n_estimators, max_depth, min_samples_split, min_samples_leaf y max_features
    for n_estimators in [50, 100, 200]:
        for max_depth in [3, 5, 7]:
            for min_samples_split in [2, 5, 10]:
                for min_samples_leaf in [1, 2, 4]:
                    for max_features in [None, 'sqrt', 'log2']:
                        print(f"Evaluando configuración: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, max_features={max_features}")
                        accuracy = train_and_evaluate_sklearn_adaboost_deep_tree(X_train, y_train, X_test, y_test,
                                                                                n_estimators=n_estimators, max_depth=max_depth,
                                                                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                                                max_features=max_features)
                        print(f"Precisión para configuración actual: {accuracy}")
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features
                            }
                            print(f"Nueva mejor precisión encontrada: {best_accuracy}")
                            print(f"Nuevos mejores parámetros: {best_params}")
    
    print(f"Mejor precisión obtenida: {best_accuracy}")
    print(f"Mejores parámetros encontrados: {best_params}")
    return best_params, best_accuracy



def main():
    
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    # Primero entrenamos y evaluamos los clasificadores (Parte 1A)
#     train_and_evaluate_adaboost()
    
    # Luego hacemos los experimentos con los parámetros T y A (Parte 1B)
#     experiment_with_parameters()
    
    # Parte 1C
#     train_and_evaluate_adaboost_multiclase()
#     experiment_with_parameters_multiclase()
    
    # Parte 1D
#     train_and_evaluate_adaboost_multiclase_pca()
#     experiment_with_parameters_multiclase_pca()
#     
    # Parte 1E
#     train_and_evaluate_adaboost_multiclase_stop_overfitting()
#     experiment_with_parameters_stop_overfitting()
    
    # Parte 2A
#     train_and_evaluate_sklearn_adaboost(X_train, Y_train, X_test, Y_test)
#     experiment_with_sklearn_adaboost_parameters(X_train, Y_train, X_test, Y_test)
    
        # Parte 2B
    best_params_2b, best_accuracy_2b = experiment_with_deep_tree_parameters(X_train, Y_train, X_test, Y_test)
    print(f"Mejores parámetros recomendados: {best_params_2b} con tasa de acierto={best_accuracy_2b:.4f}")
    
    print("FIN")

if __name__ == "__main__":
    main()
