from main import *
from tablero import *
from dominio import *
from variable import *
from restriccion import *

#Me falta tener en cuenta una variable de que solo sea una letra



def restaur():
    print("restaura")



def forward():
    print("forward")


def FC(varHor):


    for var in varHor:
        primerVar=var
        primeraPal=primerVar.getDominio()[0]
        primerVar.setDominio(primerVar.getDominio().pop(0))
        primerVar.setNombre(primeraPal)

        for varRes in varHor[1:]: #Compruebo de las siguientes variables . Seguramente pueda optimizarlo
            if(varRes.longitud()==primerVar.longitud()): #Si la longitud es igual
                for lista in varRes.getDominio():
                    if(lista==primeraPal):#Si la palabra es igual
                        varRes.setDominio(varRes.getDominio().pop(0)) #La saco de la variable
        for res in var.restriccion :
            


                
    #Recorrer el dominio de una variable



def start(almacen,tablero,varHor, varVer):
    i=0
    aux=0
    palabras=[]

    
    #Guardamos todos los dominios en sus variables

    for var in range (varHor): #Establecemos todas las variables

        pos=busca(almacen,var.longitud())
        #if(pos==-1): #No se que hay que hacer si es -1
        var.setDominio(almacen[pos][0])        

        for a in range(varHor[i].coorInicio[1], varHor[i].coorFin[1]+1):
            varY=buscarVar(varVer, var.coorInicio)
            if(var.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(a,varY, (varY.coorInicio[0]+1,varY.coorInicio[1])) #Comprobar que esto se hace como quiero
            var.set(var.getRestriccion().append(newRestriccion))
            #Mirar si hay alguna palabra que tenga esa letra en esa posicion
    fc = FC(varHor)
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
