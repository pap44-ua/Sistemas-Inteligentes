from main import *
from tablero import *
from dominio import *
from variable import *
from restriccion import *
import time
from collections import deque


def start_ac3(almacen, tablero, varHor, varVer):
#     print("Inicializando AC3")
    start_time = time.time()
    # Inicialización de dominios y restricciones
    for var in varHor:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())
        
        # Actualiza el dominio según las letras preinsertadas
        for i in range(var.coorInicio[1], var.coorFin[1] + 1):
            letra = tablero.getCelda(var.coorInicio[0], i)
            if letra != VACIA and letra != LLENA:
                var.dominio = [palabra for palabra in var.getDominio() if palabra[i - var.coorInicio[1]] == letra]
#                 print(f"Dominio actualizado para {var.getNombre()} considerando la letra '{letra}': {var.getDominio()}")

        for a in range(var.coorInicio[1], var.coorFin[1] + 1):
            coorBusq = (var.coorInicio[0], a)
            varY = buscarVar(varVer, coorBusq)
            if varY.getCoorIni() == (-1, -1):
                continue

            newRestriccion = Restriccion(varY, (coorBusq[0] - varY.getCoorIni()[0], coorBusq[1] - var.getCoorIni()[1]))
            if newRestriccion.getY().getNombre() != -1:
                restricciones = var.getRestriccion()
                restricciones.append(newRestriccion)
                var.setRestriccion(restricciones)
#                 print(f"Se añadió una restricción entre {var.getNombre()} y {varY.getNombre()} en {coorBusq}")

    for var in varVer:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())

        # Actualiza el dominio según las letras preinsertadas
        for i in range(var.coorInicio[0], var.coorFin[0] + 1):
            letra = tablero.getCelda(i, var.coorInicio[1])
            if letra != VACIA and letra != LLENA:
                var.dominio = [palabra for palabra in var.getDominio() if palabra[i - var.coorInicio[0]] == letra]
#                 print(f"Dominio actualizado para {var.getNombre()} considerando la letra '{letra}': {var.getDominio()}")

        for a in range(var.coorInicio[0], var.coorFin[0] + 1):
            coorBusq = (a, var.coorInicio[1])
            varY = buscarVar(varHor, coorBusq)
            if varY.getCoorIni() == (-1, -1):
                continue

            newRestriccion = Restriccion(varY, (coorBusq[0] - var.getCoorIni()[0], coorBusq[1] - varY.getCoorIni()[1]))
            if newRestriccion.getY().getNombre() != -1:
                restricciones = var.getRestriccion()
                restricciones.append(newRestriccion)
                var.setRestriccion(restricciones)
#                 print(f"Se añadió una restricción entre {var.getNombre()} y {varY.getNombre()} en {coorBusq}")

    variables = varHor + varVer  # Procesar ambas variables

#     print_domains(variables, "Dominios antes de AC3:")
    ac3_result = AC3(variables)
#     print_domains(variables, "Dominios después de AC3:")

#     if ac3_result:
#         print("AC3 completado con éxito.")
#     else:
#         print("AC3 falló. No hay solución.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"AC3 completado en {execution_time} segundos. Resultado: {'Éxito' if ac3_result else 'Fallo'}")
    print(f"Tiempo de ejecución: {execution_time} segundos")
    return ac3_result

def AC3(variables):
    queue = deque()
    
    for var in variables:
        for res in var.getRestriccion():
            queue.append((var, res.getY()))
    
    while queue:
        (xi, xj) = queue.popleft()
        if revise(xi, xj):
            if not xi.getDominio():
                return False
            for res in xi.getRestriccion():
                if res.getY() != xj:
                    queue.append((res.getY(), xi))

    return True

def revise(xi, xj):
    revised = False
    for x in xi.getDominio()[:]:
        if not any(satisfies(x, y, xi, xj) for y in xj.getDominio()):
            xi.getDominio().remove(x)
            revised = True
#             print(f"Eliminado {x} del dominio de {xi.getNombre()} porque no hay valor consistente en {xj.getNombre()}.")
    return revised

def satisfies(x, y, xi, xj):
    for res in xi.getRestriccion():
        if res.getY() == xj:
            index_x = res.coor[1] if xi.horizontal() else res.coor[0]
            index_y = res.coor[0] if xi.horizontal() else res.coor[1]
            if x[index_x] != y[index_y]:
                return False
    return True

def print_domains(variables, message=""):
    print(message)
    for var in variables:
        print(f"Variable {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()}): {var.getDominio()}")
    print("\n")















def forward(var: Variable, tablero: Tablero) -> bool:
#     print(f"Forward checking para la variable: {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()})")
    for res in var.getRestriccion():
        dominio = res.getY().getDominio().copy()
#         print(f"Dominio inicial para {res.getY().getNombre()} ({res.getY().getCoorIni()} -> {res.getY().getCoorFin()}): {dominio}")
        index_var = res.coor[1] if var.horizontal() else res.coor[0]
        index_res = res.coor[0] if var.horizontal() else res.coor[1]
        for dom in dominio:
            if dom[index_res] != var.getPalabra()[index_var]:
                res.getY().getBorradas().append((dom, var.getNombre()))
                res.getY().getDominio().remove(dom)
#                 print(f"Dominio actualizado para {res.getY().getNombre()} después de eliminar {dom}: {res.getY().getDominio()}")

        if not res.getY().getDominio():
#             print(f"El dominio de la variable {res.getY().getNombre()} está vacío después del forward checking.")
            return False
    return True

def restaura(var: Variable):
#     print(f"Restaurando variable: {var.getNombre()}")
    for res in var.getRestriccion():
        elementos = res.getY().getBorradas().copy()
        for elemento in elementos:
            if var.getNombre() == elemento[1]:
                res.getY().getDominio().append(elemento[0])
                res.getY().getBorradas().remove(elemento)
#                 print(f"Dominio restaurado para {res.getY().getNombre()}: {res.getY().getDominio()}")
    var.setPalabra("")

def FC(variables, aux, tablero):
#     print("Iniciando Forward Checking")
    if not variables:
        return True

    primerVar = variables[0]
    dominio = primerVar.getDominio().copy()
#     print(f"Variable {primerVar.getNombre()} dominio inicial: {dominio}")

    for primeraPal in dominio:
        primerVar.setPalabra(primeraPal)
#         print(f"Asignando la palabra {primeraPal} a la variable {primerVar.getNombre()}")

        if forward(primerVar, tablero):
            if FC(variables[1:], aux + 1, tablero):
                return True
            else:
                restaura(primerVar)
        else:
            restaura(primerVar)

        dominio = primerVar.getDominio()
        dominio.remove(primeraPal)
        primerVar.getBorradasFC().append(primeraPal)
#         print(f"Dominio actualizado después de la eliminación: {dominio}")

    if aux != 0:
        primerVar.setDominio(primerVar.getBorradasFC())
        primerVar.getBorradasFC().clear()

    return False

def start(almacen, tablero, varHor, varVer):
#     print("Inicializando Forward Checking")
    
    start_time = time.time()
    
    # Proceso de inicialización de dominios y restricciones
    for var in varHor:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())
        
        # Actualizar dominio según las letras preinsertadas
        for i in range(var.coorInicio[1], var.coorFin[1] + 1):
            letra = tablero.getCelda(var.coorInicio[0], i)
            if letra != VACIA:
                var.dominio = [palabra for palabra in var.getDominio() if palabra[i - var.coorInicio[1]] == letra]
#                 print(f"Dominio actualizado para {var.getNombre()} considerando la letra '{letra}': {var.getDominio()}")

        for a in range(var.coorInicio[1], var.coorFin[1] + 1):
            coorBusq = (var.coorInicio[0], a)
            varY = buscarVar(varVer, coorBusq)
            if varY.getCoorIni() == (-1, -1):
                continue

            newRestriccion = Restriccion(varY, (coorBusq[0] - varY.getCoorIni()[0], coorBusq[1] - var.getCoorIni()[1]))
            if newRestriccion.getY().getNombre() != -1:
                restricciones = var.getRestriccion()
                restricciones.append(newRestriccion)
                var.setRestriccion(restricciones)
#                 print(f"Se añadió una restricción entre {var.getNombre()} y {varY.getNombre()} en {coorBusq}")

    for var in varVer:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())

        # Actualizar dominio según las letras preinsertadas
        for i in range(var.coorInicio[0], var.coorFin[0] + 1):
            letra = tablero.getCelda(i, var.coorInicio[1])
            if letra != VACIA:
                var.dominio = [palabra for palabra in var.getDominio() if palabra[i - var.coorInicio[0]] == letra]
#                 print(f"Dominio actualizado para {var.getNombre()} considerando la letra '{letra}': {var.getDominio()}")

        for a in range(var.coorInicio[0], var.coorFin[0] + 1):
            coorBusq = (a, var.coorInicio[1])
            varY = buscarVar(varHor, coorBusq)
            if varY.getCoorIni() == (-1, -1):
                continue

            newRestriccion = Restriccion(varY, (coorBusq[0] - var.getCoorIni()[0], coorBusq[1] - varY.getCoorIni()[1]))
            if newRestriccion.getY().getNombre() != -1:
                restricciones = var.getRestriccion()
                restricciones.append(newRestriccion)
                var.setRestriccion(restricciones)
#                 print(f"Se añadió una restricción entre {var.getNombre()} y {varY.getNombre()} en {coorBusq}")

    variables = varHor + varVer  # Procesar ambas variables

    fc = FC(variables, 0, tablero)
#     print("FORWARD CHECKING", fc)
    

    if fc:
        for var in varHor:
            fila, columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila, columna + a, var.getPalabra()[a])
        for var in varVer:
            fila, columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila + a, columna, var.getPalabra()[a])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"FORWARD CHECKING completado en {execution_time} segundos. Resultado: {'Éxito' if fc else 'Fallo'}")
    print(f"Tiempo de ejecución: {execution_time} segundos")
    return fc

# def imprimirTablero(tablero):
#     for fila in range(tablero.getAlto()):
#         for col in range(tablero.getAncho()):
#             print(f"{tablero.getCelda(fila, col)} ", end="")
#         print()


