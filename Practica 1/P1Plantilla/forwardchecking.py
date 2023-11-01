from main import *

from tablero import *
from dominio import *
from variable import *
from restriccion import *

#Me falta tener en cuenta una variable de que solo sea una letra



def restaura(var): #Me da error pq no se asigna una variable
    aux=0
    borrados=var.getBorradas()
    print()
    print("restaura")
    print()
    print()
    #Si el dominio de la restriccion Y se queda vacio hacemos esto
    #Borramos de la variable Y
    for elemento in borrados:
        if var.getNombre() == elemento[1]:
            restriccion=var.getRestriccion()[aux].getY().getDominio()
#             print("La restriccion de la variable: ", var.getRestriccion()[aux])
#             print("La variable y de la restriccion de la variable: ", var.getRestriccion()[aux].getY())
#             print("Posicion Inicio varY: ", var.getRestriccion()[aux].getY().getCoorIni())
#             #print("El dominio de la variable y de la restriccion de la variable: ", var.getRestriccion()[aux].getY())
#             print()
#             print("La restriccion completa: ", restriccion)
#             print("El elemento: ", elemento)
#             print("La parte izq: ", elemento[0])
            restriccion.append(elemento[0])
            var.getRestriccion()[aux].getY().setDominio(restriccion)
            
        aux=aux+1
    #Borramos de la variable X
        domi=var.getDominio()
        domi.remove(var.getPalabra())
   
    
        



def forward(var,borradas):
    print
    print("forward")
    print(var.getRestriccion())
    for res in var.getRestriccion() : #C3
        
            #if(var.getNombre()[res.getPosX()]==res.getVarY().getNombre()[res.getPosY()]):
            dominio = res.getY().getDominio().copy()
            print(dominio)
            for dom in dominio:
                if(dom[res.getPosY()]!=var.getPalabra()[res.getPosX()]):
                    res.getY().getBorradas().append((dom,var.getNombre()))
                    print(res.getPosY())
                    print(res.getPosX())
                    
                    domi=res.getY().getDominio()#AQUI TENGO QUE GUARDAR QN LA BORRA
                    domi.remove(dom)
                    print("Dominio que se queda: ", res.getY().getDominio())
                    print("Lista de borradas: ", res.getY().getBorradas())
#                     res.getY().setDominio(domi)
                if not res.getY().getDominio(): #Si la lista esta vacia se sale
                    return False
                    
    return True
    #Hasta aqui C4 parte 1


def FC(variables):

    aux = 0
    borradas=[] #Tula de (palabra borrada , ID de quien lo ha borrado)

 
    for  var in variables:
        aux = aux +1 #La primera variable seria 1
        primerVar=var
        print("La primera variable: ",primerVar.getCoorIni())
        print("La primera variable: ",primerVar.getCoorFin())
        print("Dominio asignado: ", primerVar.getDominio()) 
        primeraPal=primerVar.getDominio()[0]
        print("La primera palabra", primeraPal)
        dominio = var.getDominio()  # Obten la lista de dominio
        dominio.remove(primeraPal)  # Elimina 'primeraPal' de la lista 'dominio'
        var.borradas.append((primeraPal,var.getNombre()))

        print("LO QUE DEBERIA SALIR ", dominio)
        
        primerVar.setPalabra(primeraPal)
        
        
        
        print("Palabra asignada: ", primerVar.getPalabra())
        print("Nuevo dominio asignado: ", primerVar.getDominio())
        print()
        
        if var == variables[-1]:
            return True
        if forward(var,borradas):
            if(FC(variables[1:])):
                return True
        rest=restaura(var,borradas)
            
        if  var.getDominio():
            domi= var.getDominio()[0]
            print("DOMIIIIIIIII", domi)
            var.setPalabra(domi)#Si no tiene dominio a lo mejor da error
            break
        elif aux !=1: #Si no es la primera variable
            
            if ()
            var.getDominio().append()
            print("Solo te queda esto")
            #var.setPalabra()
        else:
            return False
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
            coorBusq=(var.coorInicio[1],a)
#             print("coor siguiente")
#             print(coorBusq)
            varY=buscarVar(varVer,coorBusq)
#             print(varY)
            if(var.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(var.getCoorIni()[0],varY,a ) #Comprobar que esto se hace como quiero
#             print()
#             print("Restriccion")
#             print (newRestriccion.getPosX())
#             print(newRestriccion.getPosY())
#             print ("Var Y")
#             print(newRestriccion.getY())
#             print(newRestriccion.getY().getDominio())
#     
#             print()
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
        #print("El dominio",var.getDominio())
        for a in range(var.coorInicio[0], var.coorFin[0]+1):
#             print("El dominio",var.getDominio())
#             print("a la posicion que le toca")
#             print(a)
#             print("coorInicio variable")
#             print(var.coorInicio)
            coorBusq=(a,var.coorInicio[0])
#             print("coor siguiente")
#             print(coorBusq)
            varY=buscarVar(varHor,coorBusq )
#             print("variable Y",varY)
            if(var.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(a,varY,var.getCoorIni()[1]) #Comprobar que esto se hace como quiero
#             print("Restriccion")
#             print (newRestriccion.getPosX())
#             print(newRestriccion.getPosY())
            restricciones = var.getRestriccion()  # Obtener la lista actual de restricciones
            restricciones.append(newRestriccion)  # Agregar la nueva restricción a la lista existente
            var.setRestriccion(restricciones)  # Asignar la lista modificada de restricciones a la variable

            #var.setRestriccion(var.getRestriccion().append(newRestriccion))
#             print("Añadida gucci")
#         for res in var.getRestriccion():
#             print("Las restricciones",res.getY())
    variables = []
    variables.extend(varHor)
    variables.extend(varVer)
            
    fc = FC(variables)

#Si todo va bn imprimimos
    for var in variables:
        for pos in range(var.getCoorIni()[0],var.getCoorFin()[0]):
            for pos in range(var.getCoorIni()[1],var.getCoorFin()[1]):
                for letra in var.getPalabra():
                    Tablero.setCelda(pos[0],pos[1],letra)

    #Sacamos la primera palabra
    return fc

    #El C4 del mensaje que envio en el profesor ya no se hacerlo :_(

    return False
    
    
    
    
    
    
    
    
    
    
    
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

