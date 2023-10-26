def forward_checking(tablero, almacen):
    # Crear una lista de variables, donde cada variable representa una palabra en el crucigrama
    variables = []

    for fila in range(tablero.getAlto()):
        for columna in range(tablero.getAncho()):
            if tablero.getCelda(fila, columna) == LLENA:
                # Si la celda está marcada como LLENA, la variable representa una palabra
                # Encuentra la longitud de la palabra (número de celdas en la dirección horizontal o vertical)
                longitud_palabra = 0
                palabra_actual = []
                if columna > 0 and tablero.getCelda(fila, columna - 1) == LLENA:
                    # La palabra es horizontal
                    while columna + longitud_palabra < tablero.getAncho() and tablero.getCelda(fila, columna + longitud_palabra) == LLENA:
                        palabra_actual.append(tablero.getCelda(fila, columna + longitud_palabra))
                        longitud_palabra += 1
                else:
                    # La palabra es vertical
                    while fila + longitud_palabra < tablero.getAlto() and tablero.getCelda(fila + longitud_palabra, columna) == LLENA:
                        palabra_actual.append(tablero.getCelda(fila + longitud_palabra, columna))
                        longitud_palabra += 1
                
                if longitud_palabra > 1:  # Solo considerar palabras de más de una letra
                    variables.append((palabra_actual, longitud_palabra))
    
    # Lógica de Forward Checking
    def forward_check(variables):
        if not variables:
            return True  # Todas las variables han sido asignadas
        
        palabra_actual, longitud_actual = variables[0]
        variables_restantes = variables[1:]
        
        # Obtener el dominio de palabras válidas para la palabra actual
        dominio = []
        for dom in almacen:
            if dom.tam == longitud_actual:
                dominio.extend(dom.getLista())
        
        # Iterar a través del dominio de la palabra actual
        for palabra in dominio:
            # Asignar la palabra a la variable actual
            for i in range(longitud_actual):
                if columna > 0 and tablero.getCelda(fila, columna - 1) == LLENA:
                    tablero.setCelda(fila, columna + i, palabra[i])
                else:
                    tablero.setCelda(fila + i, columna, palabra[i])
            
            # Verificar si la asignación actual es consistente con las restricciones
            restricciones_cumplidas = True
            # Implementa aquí la lógica para verificar las restricciones con las otras variables
            
            if restricciones_cumplidas:
                # Si es consistente, continuar con las variables restantes
                if forward_check(variables_restantes):
                    return True
                
            # Si no es consistente, deshacer la asignación y probar con la siguiente palabra
            for i in range(longitud_actual):
                if columna > 0 and tablero.getCelda(fila, columna - 1) == LLENA:
                    tablero.setCelda(fila, columna + i, VACIA)
                else:
                    tablero.setCelda(fila + i, columna, VACIA)
        
        return False  # No se encontró una asignación consistente
    
    # Llamar a la función de Forward Checking
    if forward_check(variables):
        # Se encontró una asignación consistente
        print("Solución encontrada:")
        return True
    else:
        print("No se encontró solución.")
        return False

# Llamar a la función de Forward Checking desde tu programa principal
if pulsaBotonFC(pos, anchoVentana, altoVentana):
    forward_checking(tablero, almacen)