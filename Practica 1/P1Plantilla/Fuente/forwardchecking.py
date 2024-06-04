from main import *

from tablero import *
from dominio import *
from variable import *
from restriccion import *

#Me falta tener en cuenta una variable de que solo sea una letra



def restaura(var): #Me da error pq no se asigna una variable
    print("restaura restaura restaura restaura restaura restaura restaura restaura restaura restaura restaura restaura ")
    aux=0
    
#     print()
#     print("restaura")
    print()
    print()
    #Si el dominio de la restriccion Y se queda vacio hacemos esto
    #Borramos de la variable Y
    for res in var.getRestriccion():
        elementos=res.getY().getBorradas().copy()
        for elemento in elementos:
            if var.getNombre() == elemento[1]:
                print("ANTES:", res.getY().getDominio())
                res.getY().dominio.append(elemento[0])
                res.getY().borradas.remove(elemento)
                print("DESPUES:", res.getY().getDominio())

        aux=aux+1
    #Borramos de la variable X
        domi=var.setPalabra("")
    
    
        



def forward(var):
    print
    print("forward forward forward forward forward forward forward ")
    print("VARIABLE:",var.nombre)
#     print(var.getRestriccion())
    for res in var.getRestriccion() : #C3
        
        
            dominio = res.getY().getDominio().copy()
            print("DOMINIO INICIAL",dominio)
            print("Coordenada",res.coor)
            for dom in dominio:
                if(dom[res.coor[0]]!=var.getPalabra()[res.coor[1]]):
                    res.getY().getBorradas().append((dom,var.getNombre()))
                    
                    print("Coor restriccion: ",res.coor)
#                     print(res.getPosY())
                    domi=res.getY().getDominio()#AQUI TENGO QUE GUARDAR QN LA BORRA
                    domi.remove(dom)
                    print("Dominio que se queda: ", res.getY().getDominio())
                    print("Lista de borradas: ", res.getY().getBorradas())
#                     res.getY().setDominio(domi)
            if not res.getY().getDominio(): #Si la lista esta vacia se sale
                return False
    print()
    print("FORWARD TRUE")
    print()
    return True
    #Hasta aqui C4 parte 1


def FC(variables,aux):
    print()
    print("FC")
    print()
    
    borradas=[] #Tula de (palabra borrada , ID de quien lo ha borrado)
    primerVar=variables[0]
    dominio=variables[0].getDominio().copy()
  
    print()
    print("La primera variable: ",primerVar.nombre)
    print("Dominio asignado: ", primerVar.getDominio()) 
    
    for  primeraPal in dominio:
        

        
        primerVar.setPalabra(primeraPal)
        
        print("Palabra asignada: ", primerVar.getPalabra())
#         print("Nuevo dominio asignado: ", primerVar.getDominio())

        print()
        print("AUX: ", aux)
        print()
        if primerVar == variables[-1]:
            return True
        if forward(primerVar):
            aux = aux +1 #La primera variable seria 1
            
            if(FC(variables[1:],aux)):
                return True
            else:
                rest=restaura(primerVar)
            
                
        else:
            print()
            print("FORWARD FALSE")
            print()
            rest=restaura(primerVar)
            dominioo = primerVar.getDominio()  # Obten la lista de dominio
            dominioo.remove(primeraPal)
            primerVar.getBorradasFC().append(primeraPal)
            
        
    
        
    if aux !=0: #Si no es la primera variable
        print("JAJAN'T")
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
            
    fc = FC(variables,0)
    print("FORWARD CHECKING", fc)
#Si todo va bn imprimimos
    if(fc):
        for var in varHor:
            fila,columna = var.getCoorIni()
            for a in range(var.longitud()):
                tablero.setCelda(fila,columna+a,var.getPalabra()[a])
#             print("Variable",var)
# #             print("Variable coor inicio", var.getCoorIni())
# #             print("Variable coor final", var.getCoorFin())
#             print("Variable palabra", var.getPalabra())
#             for pos in range(var.getCoorIni()[0],var.getCoorFin()[0]+1):
#                 print("POSICION Y ",pos)
#                 for posi in range(var.getCoorIni()[1],var.getCoorFin()[1]+1):
#                     print("POSICION X ",posi)
#                     for letra in var.getPalabra():
#                         tablero.setCelda(pos,posi,letra)


    return fc
    
    
    
    
    
    
    
    
    
    
#     #while(True):
#     if(i==tablero.getAlto()-1):
#         break
    #else:

#     posAlmacen=busca(almacen,varHor[i].longitud())
#     palabra=almacen[posAlmacen].getLista()[0]
#     letras=[]
#     j=0
#     print(palabra)
#     
#     if(varHor[i].horizontal):
#         for letra in palabra:
#             letras.append(letra)
#         for a in range(varHor[i].coorInicio[1], varHor[i].coorFin[1]+1):
#             print(varHor[i].coorInicio)
#             print(varHor[i].coorFin)
#             if(j+1==len(letras)):
#                 break
#             print(a)
#             print (varHor[i].coorInicio[1])
#             tablero.setCelda(a,varHor[i].coorInicio[1],letras[j])
#             j=j+1
            #else:
                #for a in range(varHor[i].coorInicio[0], varHor[i].coorFin[0]+1):

