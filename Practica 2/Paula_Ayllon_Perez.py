import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.alpha = None
        self.direction = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Iterar sobre cada característica y encontrar el mejor umbral
        for feature_index in range(n_features):
            feature_values = np.unique(X[:, feature_index])
            for threshold in feature_values:
                p = 1
                prediction = np.ones(n_samples)
                prediction[X[:, feature_index] < threshold] = -1
                
                # Calcular el error ponderado
                error = np.sum(weights[y != prediction])
                
                # Guardar el mejor umbral encontrado hasta ahora
                if error < min_error:
                    min_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.direction = p if np.mean(prediction == y) > 0.5 else -1

    def predict(self, X):
        n_samples = X.shape[0]
        prediction = np.ones(n_samples)
        prediction[X[:, self.feature_index] < self.threshold] = -1
        return self.direction * prediction

class AdaboostBinario:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1/n_samples)
        
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            
            # Calcular el error ponderado
            predictions = stump.predict(X)
            err = np.sum(weights[y != predictions])
            
            # Calcular el peso del clasificador
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            self.estimators.append(stump)
            self.alphas.append(alpha)
            
            # Actualizar los pesos
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        n_samples = X.shape[0]
        results = np.zeros(n_samples)
        
        for stump, alpha in zip(self.estimators, self.alphas):
            results += alpha * stump.predict(X)
        
        return np.sign(results)
def prepare_class_data(class_label, X_train, Y_train):
    # Filtrar ejemplos de la clase específica y del resto
    indices_class = np.where(Y_train == class_label)[0]
    indices_resto = np.where(Y_train != class_label)[0]
    
    # Seleccionar tantos ejemplos de la clase como de la clase "resto"
    sample_size = min(len(indices_class), len(indices_resto))
    selected_indices = np.concatenate([indices_class[:sample_size], indices_resto[:sample_size]])
    
    X_class = X_train[selected_indices]
    Y_class = np.zeros(2 * sample_size)
    Y_class[:sample_size] = 1  # Etiquetas para la clase específica
    Y_class[sample_size:] = -1  # Etiquetas para el resto de clases
    
    return X_class, Y_class




def train_and_evaluate_adaboost_for_all_classes(X_train, Y_train, X_test, Y_test, n_estimators=1):
    num_classes = 10
    all_confusion_matrices = []

    for class_label in range(num_classes):
        print(f"Entrenando Adaboost para la clase {class_label}...")

        # Preparar los datos para la clase específica y el "resto"
        X_class, Y_class = prepare_class_data(class_label, X_train, Y_train)
        
        # Crear y entrenar el clasificador AdaboostBinario
        adaboost = AdaboostBinario(n_estimators=n_estimators)
        
        print("Entrenando AdaboostBinario")
        try:
            adaboost.fit(X_class.reshape(len(X_class), -1), Y_class)
        except Exception as e:
            print(f"JAJA no que ha pasao: {e}")

        print("AdaboostBinario entrenamiento terminado")

        # Predecir en el conjunto de test
        predictions = adaboost.predict(X_test.reshape(len(X_test), -1))
        
        # Calcular métricas
        accuracy = accuracy_score((Y_test == class_label).astype(np.float64), (predictions == 1).astype(np.float64))
        confusion_mat = confusion_matrix((Y_test == class_label).astype(int), (predictions == 1).astype(int))  # Aquí está el cambio
        
        print(f"Accuracy para la clase {class_label}: {accuracy:.2%}")
        print("Confusion Matriz:")
        print(confusion_mat)
        print()
        
        all_confusion_matrices.append(confusion_mat)


if __name__ == "__main__":
    import logging, os
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow import keras

    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Cargar los datos de MNIST
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    try:
        # Entrenar y evaluar AdaboostBinario para todas las clases
        train_and_evaluate_adaboost_for_all_classes(X_train, Y_train, X_test, Y_test)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise  # Re-lanzar la excepción para obtener más información si es necesario

