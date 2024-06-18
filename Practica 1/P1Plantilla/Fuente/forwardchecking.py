from main import *
from tablero import *
from dominio import *
from variable import *
from restriccion import *

def forward(var: Variable, tablero: Tablero) -> bool:
    print(f"Forward checking for variable: {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()})")
    for res in var.getRestriccion():
        dominio = res.getY().getDominio().copy()
        print(f"Initial domain for {res.getY().getNombre()} ({res.getY().getCoorIni()} -> {res.getY().getCoorFin()}): {dominio}")
        index_var = res.coor[1] if var.horizontal() else res.coor[0]
        index_res = res.coor[0] if var.horizontal() else res.coor[1]
        for dom in dominio:
            if dom[index_res] != var.getPalabra()[index_var]:
                res.getY().getBorradas().append((dom, var.getNombre()))
                res.getY().getDominio().remove(dom)
                print(f"Updated domain for {res.getY().getNombre()} after removing {dom}: {res.getY().getDominio()}")

        if not res.getY().getDominio():
            print(f"Domain of variable {res.getY().getNombre()} is empty after forward checking.")
            return False
    return True

def restaura(var: Variable):
    print(f"Restoring variable: {var.getNombre()}")
    for res in var.getRestriccion():
        elementos = res.getY().getBorradas().copy()
        for elemento in elementos:
            if var.getNombre() == elemento[1]:
                res.getY().getDominio().append(elemento[0])
                res.getY().getBorradas().remove(elemento)
                print(f"Restored domain for {res.getY().getNombre()}: {res.getY().getDominio()}")
    var.setPalabra("")

def FC(variables, aux, tablero):
    print("Starting Forward Checking")
    if not variables:
        return True

    primerVar = variables[0]
    dominio = primerVar.getDominio().copy()
    print(f"Variable {primerVar.getNombre()} initial domain: {dominio}")

    for primeraPal in dominio:
        primerVar.setPalabra(primeraPal)
        print(f"Assigning word {primeraPal} to variable {primerVar.getNombre()}")

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
        print(f"Updated domain after removal: {dominio}")

    if aux != 0:
        primerVar.setDominio(primerVar.getBorradasFC())
        primerVar.getBorradasFC().clear()

    return False

def start(almacen, tablero, varHor, varVer):
    print("Initializing Forward Checking")
    for var in varHor:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())

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
                print(f"Added restriction between {var.getNombre()} and {varY.getNombre()} at {coorBusq}")

    for var in varVer:
        pos = busca(almacen, var.longitud())
        var.setDominio(almacen[pos].getLista())

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
                print(f"Added restriction between {var.getNombre()} and {varY.getNombre()} at {coorBusq}")

    variables = varHor + varVer  # Procesar ambas variables
    fc = FC(variables, 0, tablero)
    print("FORWARD CHECKING", fc)

    if fc:
        for var in varHor:
            fila, columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila, columna + a, var.getPalabra()[a])
        for var in varVer:
            fila, columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila + a, columna, var.getPalabra()[a])

    return fc

def imprimirTablero(tablero):
    for fila in range(tablero.getAlto()):
        for col in range(tablero.getAncho()):
            print(f"{tablero.getCelda(fila, col)} ", end="")
        print()
