import pygame
import tkinter
from tkinter import *
from tkinter.simpledialog import *
from tkinter import messagebox as MessageBox
from tablero import *
from dominio import *
from variable import *
from pygame.locals import *
from forwardchecking import *
from AC3 import *

GREY=(190, 190, 190)
NEGRO=(100,100, 100)
BLANCO=(255, 255, 255)

MARGEN=5 #ancho del borde entre celdas
MARGEN_INFERIOR=60 #altura del margen inferior entre la cuadrícula y la ventana
TAM=60  #tamaño de la celda
FILS=5 #5 # número de filas del crucigrama
COLS=6#6 # número de columnas del crucigrama



LLENA='*' 
VACIA='-'

#########################################################################
# Detecta si se pulsa el botón de FC
######################################################################### 
def pulsaBotonFC(pos, anchoVentana, altoVentana):
    if pos[0]>=anchoVentana//4-25 and pos[0]<=anchoVentana//4+25 and pos[1]>=altoVentana-45 and pos[1]<=altoVentana-19:
        return True
    else:
        return False
    
######################################################################### 
# Detecta si se pulsa el botón de AC3
######################################################################### 
def pulsaBotonAC3(pos, anchoVentana, altoVentana):
    if pos[0]>=3*(anchoVentana//4)-25 and pos[0]<=3*(anchoVentana//4)+25 and pos[1]>=altoVentana-45 and pos[1]<=altoVentana-19:
        return True
    else:
        return False
    
######################################################################### 
# Detecta si se pulsa el botón de reset
######################################################################### 
def pulsaBotonReset(pos, anchoVentana, altoVentana):
    if pos[0]>=(anchoVentana//2)-25 and pos[0]<=(anchoVentana//2)+25 and pos[1]>=altoVentana-45 and pos[1]<=altoVentana-19:
        return True
    else:
        return False
    
######################################################################### 
# Detecta si el ratón se pulsa en la cuadrícula
######################################################################### 
def inTablero(pos):
    if pos[0]>=MARGEN and pos[0]<=(TAM+MARGEN)*COLS+MARGEN and pos[1]>=MARGEN and pos[1]<=(TAM+MARGEN)*FILS+MARGEN:        
        return True
    else:
        return False
    
######################################################################### 
# Busca posición de palabras de longitud tam en el almacen
######################################################################### 
def busca(almacen, tam):
    enc=False
    pos=-1
    i=0
    while i<len(almacen) and enc==False:
        if almacen[i].tam==tam: 
            pos=i
            enc=True
        i=i+1
    return pos
    
######################################################################### 
# Crea un almacen de palabras
######################################################################### 
def creaAlmacen():
    f= open('d0.txt', 'r', encoding="utf-8")
    lista=f.read()
    f.close()
    listaPal=lista.split()
    almacen=[]
   
    for pal in listaPal:        
        pos=busca(almacen, len(pal)) 
        if pos==-1: #no existen palabras de esa longitud
            dom=Dominio(len(pal))
            dom.addPal(pal.upper())            
            almacen.append(dom)
        elif pal.upper() not in almacen[pos].lista: #añade la palabra si no está duplicada        
            almacen[pos].addPal(pal.upper())           
    
    return almacen

######################################################################### 
# Imprime el contenido del almacen
######################################################################### 
def imprimeAlmacen(almacen):
    for dom in almacen:
        print (dom.tam)
        lista=dom.getLista()
        for pal in lista:
            print (pal, end=" ")
        print()


def sacarVariablesHor(tablero,ID):#DUDA: Mirar bn si reconoce las variables de 1
    #X columna
    #Y fila
    variables= []
    coorInicio=(0,0)
    for fila in range(0, tablero.getAlto()):
        coorInicio=(fila,0)
        for col in range( 0, tablero.getAncho()):
            coorInicio = (fila , coorInicio[1]) #Va actualizando la coorInicio, solo cambiará cuando se encuentre con una casilla negra
            
          
            if(col== tablero.getAncho()-1):
                if(tablero.getCelda(fila,col)!=LLENA):
                    varAux = Variable((coorInicio),(fila,col),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)
                        ID=ID+1#si donde esta es vacio que lo tenga en cuenta
                    
                else: #si la casilla negra es la ultima
                    varAux = Variable((coorInicio),(fila,col-1),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)
                        ID=ID+1#si donde esta es vacio que lo tenga en cuenta
                    
                                    
            elif(tablero.getCelda(fila,col)==LLENA):#ya no hay error
                #si la casilla negra es la primera
                if(coorInicio[1]==col):
                    coorInicio=(fila,col+1)
                else:#si la casilla negra esta por en medio
                    
                    varAux = Variable((coorInicio),(fila,col-1),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)#si donde esta esta en negro que no la cuente para la variable
                        ID=ID+1
                    coorInicio=(coorInicio[0],col+1) 
                    
                    
               
                              
    
    return variables, ID

def sacarVariablesVer(tablero,ID):
    #X columna
    #Y fila
    variables= []
    coorInicio=(0,0)
    for col in range(0, tablero.getAncho()):
        coorInicio=(0,col)
        for fil in range( 0, tablero.getAlto()):
            coorInicio = (coorInicio[0],col) #Va actualizando la coorInicio, solo cambiará cuando se encuentre con una casilla negra
            
            if(fil== tablero.getAlto()-1):
                if(tablero.getCelda(fil,col)!=LLENA):
                    varAux = Variable((coorInicio),(fil,col),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)
                        ID=ID+1#si donde esta es vacio que lo tenga en cuenta
                    
                else: #si la casilla negra es la ultima
                    varAux = Variable((coorInicio),(fil-1,col),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)
                        ID=ID+1#si donde esta es vacio que lo tenga en cuenta
                    
                    

            elif(tablero.getCelda(fil,col)==LLENA):#error aqui
                #si la casilla negra es la primera
                if(coorInicio[0]==fil):
                    coorInicio=(fil+1,col)
                else:#si la casilla negra esta por en medio
                    varAux = Variable((coorInicio),(fil-1,col),ID)
                    
                    if(varAux.longitud() != 1):
                        variables.append(varAux)#si donde esta esta en negro que no la cuente para la variable
                        ID=ID+1
                    coorInicio=(fil+1,coorInicio[1])  
                    
            
                             
    
    return variables, ID



#Busca en la lista que quiero la variable que tenga esa coordenada de Inicio

def buscarVar(listaVar, coorBusq): #Falta implementar que si las variables son horizantales lo haga con x

    #for que recorre la lista de las variables
    for palabra in listaVar:
        #for que recorre todas las posiciones y
        for y in range(palabra.getCoorIni()[0], palabra.getCoorFin()[0]+1): #para caso vertical
            if((y,palabra.getCoorIni()[1])==coorBusq):
                return palabra
        for x in range(palabra.getCoorIni()[1], palabra.getCoorFin()[1]+1): #para caso horizontal ( no hace falta un if pq directamente si no cambia la x no entrara)
            if((palabra.getCoorIni()[0],x)==coorBusq):
                return palabra
    return Variable((-1,-1),(-1,-1),-1)
    
def crearRestricciones(varHor, varVer):
    for var in varHor:
        for col in range(var.getCoorIni()[1], var.getCoorFin()[1] + 1):
            fila = var.getCoorIni()[0]
            for var_v in varVer:
                if var_v.getCoorIni()[0] <= fila <= var_v.getCoorFin()[0] and col == var_v.getCoorIni()[1]:
                    var.getRestriccion().append(Restriccion(var_v, (fila, col)))
                    var_v.getRestriccion().append(Restriccion(var, (fila, col)))
                    print(f"Created restriction between {var.getNombre()} and {var_v.getNombre()} at ({fila}, {col})")


#########################################################################  
# Principal
#########################################################################
def main():
    global ID
    ID = 0  # Inicializar ID aquí
    
    root= tkinter.Tk() #para eliminar la ventana de Tkinter
    root.withdraw() #se cierra
    pygame.init()
    
    reloj=pygame.time.Clock()
    
    anchoVentana=COLS*(TAM+MARGEN)+MARGEN
    altoVentana= MARGEN_INFERIOR+FILS*(TAM+MARGEN)+MARGEN
    
    dimension=[anchoVentana,altoVentana]
    screen=pygame.display.set_mode(dimension) 
    pygame.display.set_caption("Practica 1: Crucigrama")
    
    botonFC=pygame.image.load("botonFC.png").convert()
    botonFC=pygame.transform.scale(botonFC,[50, 30])
    
    botonAC3=pygame.image.load("botonAC3.png").convert()
    botonAC3=pygame.transform.scale(botonAC3,[50, 30])
    
    botonReset=pygame.image.load("botonReset.png").convert()
    botonReset=pygame.transform.scale(botonReset,[50,30])
    
    almacen=creaAlmacen()
    game_over=False
    tablero=Tablero(FILS, COLS)    
    while not game_over:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:               
                game_over=True
            if event.type==pygame.MOUSEBUTTONUP:                
                #obtener posición y calcular coordenadas matriciales                               
                pos=pygame.mouse.get_pos()                
                if pulsaBotonFC(pos, anchoVentana, altoVentana): #Aqui hace el FC
                    print("FC")
                    varHor,ID= sacarVariablesHor(tablero,ID)
                    varVer,ID=sacarVariablesVer(tablero,ID)
                    
#                     for i in varHor:
#                         print("HORIZONTAL")
#                         print(i.coorInicio)
#                         print(i.coorFin)
#                         print(i.longitud())
#                         print(i.getNombre())
#                     for i in varVer:
#                         print("VERTICAL")
#                         print(i.coorInicio)
#                         print(i.coorFin)
#                         print(i.longitud())
#                         print(i.getNombre())
                        
                    res=start(almacen,tablero,varHor,varVer)
                    
                    if res==False:
                        MessageBox.showwarning("Alerta", "No hay solución")                                  
                elif pulsaBotonAC3(pos, anchoVentana, altoVentana):                    
                     print("AC3")
                     varHor, ID = sacarVariablesHor(tablero, ID)
                     varVer, ID = sacarVariablesVer(tablero, ID)
                     crearRestricciones(varHor, varVer)  # Crear restricciones entre variables
                     variables = varHor + varVer

                    # Inicializar dominios
                     for var in variables:
                         tam = var.longitud()
                         pos = busca(almacen, tam)
                         if pos != -1:
                             var.setDominio(almacen[pos].getLista())

                     print("DOMINIOS ANTES DEL AC3:")
                     for var in variables:
                         print(f"Variable {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()}): {var.getDominio()}")

                     result = AC3(variables)

                     print("DOMINIOS DESPUÉS DEL AC3:")
                     for var in variables:
                         print(f"Variable {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()}): {var.getDominio()}")

                     if result:
                         print("AC3 successfully reduced the domains.")
                     else:
                         MessageBox.showwarning("Alerta", "No hay solución")
                elif pulsaBotonReset(pos, anchoVentana, altoVentana):                   
                    tablero.reset()
                    ID=0
                elif inTablero(pos):
                    colDestino=pos[0]//(TAM+MARGEN)
                    filDestino=pos[1]//(TAM+MARGEN)                    
                    if event.button==1: #botón izquierdo
                        if tablero.getCelda(filDestino, colDestino)==VACIA:
                            tablero.setCelda(filDestino, colDestino, LLENA)
                        else:
                            tablero.setCelda(filDestino, colDestino, VACIA)
                    elif event.button==3: #botón derecho
                        c=askstring('Entrada', 'Introduce carácter')
                        tablero.setCelda(filDestino, colDestino, c.upper())   
            
        ##código de dibujo        
        #limpiar pantalla
        screen.fill(NEGRO)
        pygame.draw.rect(screen, GREY, [0, 0, COLS*(TAM+MARGEN)+MARGEN, altoVentana],0)
        for fil in range(tablero.getAlto()):
            for col in range(tablero.getAncho()):
                if tablero.getCelda(fil, col)==VACIA: 
                    pygame.draw.rect(screen, BLANCO, [(TAM+MARGEN)*col+MARGEN, (TAM+MARGEN)*fil+MARGEN, TAM, TAM], 0)
                elif tablero.getCelda(fil, col)==LLENA: 
                    pygame.draw.rect(screen, NEGRO, [(TAM+MARGEN)*col+MARGEN, (TAM+MARGEN)*fil+MARGEN, TAM, TAM], 0)
                else: #dibujar letra                    
                    pygame.draw.rect(screen, BLANCO, [(TAM+MARGEN)*col+MARGEN, (TAM+MARGEN)*fil+MARGEN, TAM, TAM], 0)
                    fuente= pygame.font.Font(None, 70)
                    texto= fuente.render(tablero.getCelda(fil, col), True, NEGRO)            
                    screen.blit(texto, [(TAM+MARGEN)*col+MARGEN+15, (TAM+MARGEN)*fil+MARGEN+5])             
        #pintar botones        
        screen.blit(botonFC, [anchoVentana//4-25, altoVentana-45])
        screen.blit(botonAC3, [3*(anchoVentana//4)-25, altoVentana-45])
        screen.blit(botonReset, [anchoVentana//2-25, altoVentana-45])
        #actualizar pantalla
        pygame.display.flip()
        reloj.tick(40)
        if game_over==True: #retardo cuando se cierra la ventana
            pygame.time.delay(500)
    
    pygame.quit()
 
if __name__=="__main__":
    main()
 
     

