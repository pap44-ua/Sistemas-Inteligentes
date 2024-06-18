from main import *

from tablero import *
from dominio import *
from variable import *
from restriccion import *

#Me falta tener en cuenta una variable de que solo sea una letra



from variable import Variable
from tablero import Tablero

def forward(var: Variable, tablero: Tablero) -> bool:
    print(f"Forward checking for variable: {var.getNombre()}")
    for res in var.getRestriccion():
        dominio = res.getY().getDominio().copy()
        print(f"Initial domain for {res.getY().getNombre()}: {dominio}")
        for dom in dominio:
            # Ajusta el índice para verificar la posición correcta del carácter
            if dom[abs(res.varY.coorInicio[0] - res.coor[0])] != var.getPalabra()[res.coor[1]]:
                res.getY().getBorradas().append((dom, var.getNombre()))
                res.getY().getDominio().remove(dom)
                print(f"Updated domain for {res.getY().getNombre()}: {res.getY().getDominio()}")

        if not res.getY().getDominio():
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

        
       


def start(almacen,tablero,varHor, varVer):
    
    #aux=0
    #palabras=[]

    #Guardamos todos los dominios en sus variables

    for var in varHor: #Establecemos todas las variables Horizontales

        pos=busca(almacen,var.longitud())
# 
        var.setDominio(almacen[pos].getLista())        

        for a in range(var.coorInicio[1], var.coorFin[1]+1):

            coorBusq=(var.coorInicio[0],a)

            varY=buscarVar(varVer,coorBusq)

            if(var.coorInicio[0]==-1):
                break
            
            newRestriccion=Restriccion(varY,(coorBusq[0],coorBusq[1]-var.coorInicio[1]) ) #Comprobar que esto se hace como quiero
            

            if newRestriccion.getY().getNombre() != -1:
                
#                 print()
#                 print("HORIZONTAL")
#                 print("NOMBRE:", var.getNombre())
#                 print("Restriccion")
#     #             print (newRestriccion.getPosX())
#                 print("NOMBRE:", newRestriccion.getY().getNombre())
#                 
                restricciones = var.getRestriccion()
                restricciones.append(newRestriccion)
                var.setRestriccion(restricciones)
                

      
    for var in varVer: #Establecemos todas las variables Verticales

        pos=busca(almacen,var.longitud())

        dominio=almacen[pos].getLista()
        var.setDominio(dominio)        


        for a in range(var.coorInicio[0], var.coorFin[0]+1):
#             print("VERTICAL")
#             print("el nombre de la variable de ahora",var.nombre)
#             print("El dominio",var.getDominio())
#             print("a la posicion que le toca")
#             print(a)
#             print("coorInicio variable")
#             print(var.coorInicio)
#             
            coorBusq=(a,var.coorInicio[1])

            varY=buscarVar(varHor,coorBusq)
            
            
            if(varY.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(varY,(coorBusq[0]-var.coorInicio[0],coorBusq[1])) #Comprobar que esto se hace como quiero

            if (newRestriccion.getY().getNombre()!=-1):
                restricciones = var.getRestriccion()  # Obtener la lista actual de restricciones
                restricciones.append(newRestriccion)  # Agregar la nueva restricción a la lista existente
                var.setRestriccion(restricciones)  # Asignar la lista modificada de restricciones a la variable
            


    variables = []
    variables.extend(varHor)
#     variables.extend(varVer)
            
    fc = FC(variables,0,tablero)
    print("FORWARD CHECKING", fc)
#Si todo va bn imprimimos
    if(fc):
        for var in varHor:
            fila,columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila,columna+a,var.getPalabra()[a])


    return fc
    
    
def imprimirTablero(tablero):
    for fila in range(tablero.getAlto()):
        for col in range(tablero.getAncho()):
            # Imprimir la celda y un espacio en la misma línea
            print(f"{tablero.getCelda(fila, col)} ", end="")
        # Saltar a la siguiente línea después de imprimir todas las celdas de la fila
        print()

    
