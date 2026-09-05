# Sistemas Inteligentes

Repositorio con las prácticas realizadas para la asignatura **Sistemas Inteligentes** (Grado en Ingeniería Informática). Incluye implementaciones de técnicas clásicas de IA (satisfacción de restricciones) y de aprendizaje automático (boosting).

##  Estructura del repositorio

```
Sistemas-Inteligentes-main/
├── Practica 1/          # CSP: resolución de crucigramas
│   ├── P1Plantilla/      # Plantilla base del ejercicio
│   ├── marina/           # Entrega/variante de una compañera de prácticas
│   ├── enunciadoP1.pdf   # Enunciado de la práctica
│   └── Practica1Julio.pdf
└── Practica 2/          # Aprendizaje automático: Adaboost sobre MNIST
    ├── practica2.py
    ├── prac2.py / Paula_Ayllon_Perez.py / Portatil.py
    ├── Practica 2 SI.docx
    └── enunciados y transparencias (PDF)
```

##  Práctica 1 — Resolución de crucigramas con CSP

Implementación de un **crucigrama como Problema de Satisfacción de Restricciones (CSP)**, con interfaz gráfica en `pygame`.

- **`tablero.py`**: representa la rejilla del crucigrama (filas × columnas).
- **`variable.py`**: modela cada hueco (horizontal/vertical) como una variable del CSP, con su dominio de palabras posibles.
- **`dominio.py`**: gestiona los conjuntos de palabras candidatas por longitud.
- **`restriccion.py`**: define las restricciones de coincidencia de letras entre variables que se cruzan.
- **`forwardchecking.py`**: implementa el algoritmo **Forward Checking**, podando dominios al asignar una palabra a una variable.
- **`AC3.py`**: implementa el algoritmo de consistencia de arcos **AC-3**.
- **`main.py`**: interfaz gráfica (pygame + tkinter) con botones para ejecutar FC, AC3 y resetear el tablero.

La carpeta `marina/` contiene una versión/entrega paralela del mismo ejercicio.

##  Práctica 2 — Adaboost sobre el dataset MNIST

Implementación desde cero del algoritmo **Adaboost** (boosting de clasificadores débiles tipo *decision stump*), aplicado a la clasificación de dígitos manuscritos del dataset **MNIST**.

- **`DecisionStump`**: clasificador débil basado en un umbral sobre un único píxel/característica.
- **`AdaboostBinario`**: combina varios *decision stumps* mediante boosting para clasificación binaria (ej. "¿es un 5 o no?").
- **`AdaboostMulticlase`**: extiende el binario a las 10 clases de dígitos mediante estrategia *uno-contra-el-resto*.
- **`experimenta_T_A()`**: experimentos variando el número de clasificadores débiles (T) para estudiar precisión vs. tiempo de entrenamiento, con gráficas (`matplotlib`).
- Uso de `tensorflow.keras` únicamente para cargar el dataset MNIST.

Existen varias variantes del ejercicio de distintos compañeros (`prac2.py`, `Paula_Ayllon_Perez.py`, `Portatil.py`, `Logs de lo que voy cambiando.py`, etc.), además del enunciado y transparencias de la práctica en PDF.

##  Tecnologías utilizadas

- **Python**
- `pygame` y `tkinter` (interfaz gráfica de la Práctica 1)
- `numpy`, `tensorflow`/`keras`, `matplotlib` (Práctica 2)

##  Ejecución

**Práctica 1** (crucigrama):
```bash
cd "Practica 1/P1Plantilla/Fuente"
python main.py
```

**Práctica 2** (Adaboost/MNIST):
```bash
cd "Practica 2"
python practica2.py
```

##  Licencia

Proyecto académico sin licencia específica — uso educativo.
