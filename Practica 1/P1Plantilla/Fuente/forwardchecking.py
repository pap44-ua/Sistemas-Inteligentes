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
                res.getY().getDominio().append(elemento[0])
                res.getY().getBorradas().remove(elemento)
                print("DESPUES:", res.getY().getDominio())
#                 restriccion=var.getRestriccion()[aux].getY().getDominio()
#                 restriccion.append(elemento[0])
#                 var.getRestriccion()[aux].getY().setDominio(restriccion)
            
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


def FC(variables):
    print()
    print("FC")
    print()
    aux = 0
    borradas=[] #Tula de (palabra borrada , ID de quien lo ha borrado)
    primerVar=variables[0]
    dominio=variables[0].getDominio().copy()
    print()
    print()
    for var in variables:
        print("VARIABLE LISTA",var.nombre)
        
    print()
    print()
    print("La primera variable: ",primerVar.nombre)
    print("Dominio asignado: ", primerVar.getDominio()) 
    
    for  primeraPal in dominio:
        
        
#         print("La primera variable: ",primerVar.getCoorFin())
        
        print("La primera palabra", primeraPal)
        if(primerVar.nombre == 0):
            rest=restaura(primerVar)
            
            
#         
       # Elimina 'primeraPal' de la lista 'dominio'

        print("LO QUE DEBERIA SALIR ", dominio)
        
        primerVar.setPalabra(primeraPal)
        
        #NO ESTA
        #Quitar P de todos los dominios restantes
        
        print("Palabra asignada: ", primerVar.getPalabra())
        print("Nuevo dominio asignado: ", primerVar.getDominio())
#         print()
        print()
        print("AUX: ", aux)
        print()
        if primerVar == variables[-1]:
            return True
        if forward(primerVar):
            aux = aux +1 #La primera variable seria 1
            
            if(FC(variables[1:])):
                
                return True
            
                
        else:
            print()
            print("FORWARD FALSE")
            print()
            rest=restaura(primerVar)
            dominioo = primerVar.getDominio()  # Obten la lista de dominio
            dominioo.remove(primeraPal)
            primerVar.getBorradasFC().append(primeraPal)
            
        
    
        
    if aux !=1: #Si no es la primera variable
        print("JAJAN'T")
        primerVar.setDominio(primerVar.getBorradas())
        
    return False
        
        
        
       
        
        
#         for varRes in varHor: #Compruebo de las siguientes variables . Seguramente pueda optimizarlo
#             if(varRes.longitud()==primerVar.longitud()): #Si la longitud es igual
#                 
#                 for lista in varRes.getDominio():
#                     if(lista==primeraPal):#Si la palabra es igual
#                         domi=varRes.getDominio()
#                         domi.pop(0)
#                         varRes.setDominio(domi) #La saco de la variable
#             for res in varRes.getRestriccion():
#                 print("Variable: ", varRes.getNombre())
#                 print("Restriccion en X: ",res.getPosX()," y en la pos Y: ", res.getPosY())
                          
        
        #Si forward(i,a)
            #si FC(i+1) return true
        #restaura i
    
   
       
    #Recorrer el dominio de una variable



def start(almacen,tablero,varHor, varVer):
    
    #aux=0
    #palabras=[]

    #Guardamos todos los dominios en sus variables

    for var in varHor: #Establecemos todas las variables Horizontales

        pos=busca(almacen,var.longitud())
#         print("la longitud de la variable",var.longitud())
#         print("el nombre de la variable de ahora",var.nombre)
        
#         print("posicion del almacen", pos)
#         #if(pos==-1): #No se que hay que hacer si es -1
#         print("El almacen",almacen[pos])
#         print("La lista del almacen", almacen[pos].getLista())
        var.setDominio(almacen[pos].getLista())        
        #print("El dominio",var.getDominio())
        for a in range(var.coorInicio[1], var.coorFin[1]+1):
#             print("La lista del almacen", almacen[pos].getLista())
#             print("a la posicion que le toca")
#             print(a)
#             print("coorInicio variable")
#             print(var.coorInicio)
            coorBusq=(var.coorInicio[0],a)
#             print("coor busqueda")
#             print(coorBusq)
            varY=buscarVar(varVer,coorBusq)
#             print("Variable Y",varY.getNombre())
            if(var.coorInicio[0]==-1):
                break
            
            
            #a devuelve la posicion que le corresponde
            newRestriccion=Restriccion(varY,coorBusq ) #Comprobar que esto se hace como quiero
            
#             
# #             print()
# #             print("HORIZONTAL")
# #             print("NOMBRE:", var.getNombre())
# #             print("Restriccion")
# #             print (newRestriccion.getPosX())
# #             print("NOMBRE:", newRestriccion.getY().getNombre())
# #             print(newRestriccion.getPosY())
# #             print ("Var Y")
# #             print(newRestriccion.getY())
# #             print(newRestriccion.getY().getDominio())
# #             
# #     
# #             print()
            restricciones = var.getRestriccion()  # Obtener la lista actual de restricciones
            restricciones.append(newRestriccion)  # Agregar la nueva restricción a la lista existente
#             var.setRestriccion(restricciones)  # Asignar la lista modificada de restricciones a la variable

            #var.setRestriccion(var.getRestriccion().append(newRestriccion))
#             
    for var in varVer: #Establecemos todas las variables Verticales

        pos=busca(almacen,var.longitud())
        #if(pos==-1): #No se que hay que hacer si es -1
#         print("pos almacen",almacen[pos])
        dominio=almacen[pos].getLista()
        var.setDominio(dominio)        
#         print("El dominio",var.getDominio())

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
            
#             print("coor siguiente")
#             print(coorBusq)
#             
            varY=buscarVar(varHor,coorBusq)
            
#             print("variable Y",varY.nombre)
            
            if(varY.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(varY,coorBusq) #Comprobar que esto se hace como quiero
            
#             print("Restriccion")
#             print (newRestriccion.getPosX())
#             print(newRestriccion.getPosY())
#             
            restricciones = var.getRestriccion()  # Obtener la lista actual de restricciones
            restricciones.append(newRestriccion)  # Agregar la nueva restricción a la lista existente
            var.setRestriccion(restricciones)  # Asignar la lista modificada de restricciones a la variable
            
#             print()
#             print("VERTICAL")
#             print("NOMBRE:", var.getNombre())
# #             print("Restriccion")
# #             print (newRestriccion.getPosX())
#             print("NOMBRE:", newRestriccion.getY().getNombre())
#             print(newRestriccion.getPosY())
#             print ("Var Y")
#             print(newRestriccion.getY())
#             print(newRestriccion.getY().nombre)
            #var.setRestriccion(var.getRestriccion().append(newRestriccion))
#             print("Añadida gucci")
#         for res in var.getRestriccion():
#             print("Las restricciones",res.getY())
    variables = []
    variables.extend(varHor)
    variables.extend(varVer)
            
    fc = FC(variables)
#     print("FORWARD CHECKING", fc)
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

