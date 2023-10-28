from main import *
from tablero import *
from dominio import *
from variable import *
from restriccion import *

def start(almacen,tablero,varHor, varVer):
    i=0
    palabras=[]
    
    primerVar=varHor[0]
    primeraPal=primerVar[0].getDominio()[0]
    primerVar.setDominio(primerVar[0].getDominio().pop(0))
    
    palabras.append(primeraPal)
    
    primerVer=varVer[0]
    
    for a in range(varHor[i].coorInicio[1], varHor[i].coorFin[1]+1):
        varY=buscarVar(varVer, primerVar.coorInicio)
        newRestriccion=Restriccion(a,varY, (varY.coorInicio[0]+1,varY.coorInicio[1]))
        primerVar.set(primerVar.getRestriccion().append(newRestriccion))
        
    #El C4 del mensaje que envio en el profesor ya no se hacerlo :_(
    
    
    
    
    
    
    
    
    
    
    
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
