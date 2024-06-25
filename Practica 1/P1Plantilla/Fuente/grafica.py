import matplotlib.pyplot as plt

# Datos de las pruebas
pruebas = ['3x3 Pequeño Diccionario', '4x4 Pequeño Diccionario', '3x3 Grande Diccionario', '4x4 Grande Diccionario']
tiempos_fc = [0.0, 0.00099921, 0.00101327, 0.001998424]
tiempos_ac3 = [0.00197362, 0.00200057029, 0.00602650, 0.0069785]

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(pruebas, tiempos_fc, label='Forward Checking (FC)', marker='o')
plt.plot(pruebas, tiempos_ac3, label='AC3', marker='o')

# Añadir etiquetas y título
plt.xlabel('Prueba')
plt.ylabel('Tiempo (s)')
plt.title('Comparación de Tiempos entre FC y AC3')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig('comparacion_tiempos.png')
plt.show()