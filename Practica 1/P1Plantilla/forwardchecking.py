from main import *
from main import busca
from tablero import *
from dominio import *
from variable import *
from restriccion import *

#Me falta tener en cuenta una variable de que solo sea una letra



def restaura(borrados,):
    print("restaura")



def forward(var,borradas):
    print("forward")

    for res in var.restriccion : #C3
            #if(var.getNombre()[res.getPosX()]==res.getVarY().getNombre()[res.getPosY()]):
            dominio = res.getY().getDominio().copy()
            for dom in dominio:
                if(dom[res.getPosY()]!=var.getNombre(res.getPosX())):
                    borradas.append((dom,res.getY().getNombre()))   #AQUI TENGO QUE GUARDAR QN LA BORRA
                    res.getY().setDominio(res.getY().getDominio().remove(dom))
                if not res.getY().getDominio(): #Si la lista esta vacia se sale
                    return False
                    
    return True
    #Hasta aqui C4 parte 1


def FC(varHor):

    borradas=[] #Tula de (palabra borrada , ID de quien lo ha borrado)


    for index, var in enumerate(varHor):
        primerVar=var
        primeraPal=primerVar.getDominio()[0]
        primerVar.setDominio(primerVar.getDominio().pop(0))
        primerVar.setPalabra(primeraPal)

        for varRes in varHor: #Compruebo de las siguientes variables . Seguramente pueda optimizarlo
            if(varRes.longitud()==primerVar.longitud()): #Si la longitud es igual
                for lista in varRes.getDominio():
                    if(lista==primeraPal):#Si la palabra es igual
                        varRes.setDominio(varRes.getDominio().pop(0)) #La saco de la variable
        
        #Si forward(i,a)
            #si FC(i+1) return true
        #restaura i
        if var == varHor[-1]:
            return True
        if forward(var):
            print(f"si FC(i+1) retorna cierto")
            if(FC(varHor[1:])):
                return True
        rest=restaura()
            
    return False
       
    #Recorrer el dominio de una variable



def start(almacen,tablero,varHor, varVer):
    
    #aux=0
    #palabras=[]

    #Guardamos todos los dominios en sus variables

    for var in varHor: #Establecemos todas las variables Horizontales

        pos=busca(almacen,var.longitud())
        #if(pos==-1): #No se que hay que hacer si es -1
        var.setDominio(almacen[pos][0])        
        print(var.getDominio())
        for a in range(var.coorInicio[0], var.coorFin[0]+1):
            varY=buscarVar(varVer, var.coorInicio)
            if(var.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(a,varY, (varY.coorInicio[0]+1,varY.coorInicio[1])) #Comprobar que esto se hace como quiero
            var.set(var.getRestriccion().append(newRestriccion))
            #Mirar si hay alguna palabra que tenga esa letra en esa posicion

    for var in varVer: #Establecemos todas las variables Verticales

        pos=busca(almacen,var.longitud())
        #if(pos==-1): #No se que hay que hacer si es -1
        var.setDominio(almacen[pos][0])  #Guarda la lista de su tamaño      
        print(var.getDominio())
        for a in range(var.coorInicio[1], var.coorFin[1]+1):
            varX=buscarVar(varHor, var.coorInicio)
            if(var.coorInicio[0]==-1):
                break
            newRestriccion=Restriccion(a,varX, (varX.coorInicio[0]+1,varX.coorInicio[1])) #Comprobar que esto se hace como quiero
            var.set(var.getRestriccion().append(newRestriccion))
            #Mirar si hay alguna palabra que tenga esa letra en esa posicion

        
    fc = FC(varHor)

#Si todo va bn imprimimos
    for var in varHor:
        for pos in range(var.getCoorIni(),var.getCoorFin()):
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
