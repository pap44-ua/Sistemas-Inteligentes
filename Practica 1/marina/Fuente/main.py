import pygame
import tkinter
import time
from tkinter import *
from tkinter.simpledialog import *
from tkinter import messagebox as MessageBox
from tablero import *
from dominio import *
from variable import Variable
from restriccion import Restriccion
from pygame.locals import *

GREY=(190, 190, 190)
NEGRO=(100,100, 100)
BLANCO=(255, 255, 255)

MARGEN=5 #ancho del borde entre celdas
MARGEN_INFERIOR=60 #altura del margen inferior entre la cuadrícula y la ventana
TAM=60  #tamaño de la celda
FILS=5 # número de filas del crucigrama
COLS=6 # número de columnas del crucigrama

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
######################################################################### 
# Crea lista de variables
#########################################################################
def sacarVariables(tablero):
    variables = []
    inicio = None
    num_huecoX = 0
    añadir = True
    #horizontal
    for i in range(FILS):
        for j in range(COLS):
            añadir = True
            celda = tablero.getCelda(i,j)
            if(celda==LLENA and inicio is None):
                continue
            if(celda == VACIA):
                if(inicio is None):
                    inicio = (j, i)
            if(celda == LLENA or j == COLS-1):
                if(celda==LLENA):
                    x=j-1
                else:
                    x = j
                if(inicio[0] - x == 0):
                    if(i > 0 and tablero.getCelda(i-1,x) != LLENA):
                        añadir = False
                    if(i < FILS-1 and tablero.getCelda(i+1,x) != LLENA):
                        añadir = False
                    if(x > 0 and tablero.getCelda(i,x-1) != LLENA):
                        añadir = False
                    if(x < COLS-1 and tablero.getCelda(i,x+1) != LLENA):
                        añadir = False
                if(añadir):
                    num_huecoX+=1
                    variables.append(Variable(str(num_huecoX) +"X", inicio,(x,i),"horizontal"))
                inicio= None
    inicio= None
    #vertical
    num_huecoY = 0
    for i in range(COLS):
        for j in range(FILS):
            añadir = True
            mas_de_uno = True
            celda = tablero.getCelda(j,i)
            if(celda==LLENA and inicio is None):
                continue
            if(celda == VACIA):
                if(inicio is None):
                    inicio = (i, j)
            if(celda == LLENA or j == FILS-1):
                if(celda==LLENA):
                    x=j-1
                else:
                    x = j
                if(inicio[1] - x == 0):
                    if(x > 0 and tablero.getCelda(x-1,i) != LLENA):
                        mas_de_uno = False
                    if(x < FILS-1 and tablero.getCelda(x+1, i) != LLENA):
                        mas_de_uno = False
                    if(i > 0 and tablero.getCelda(x,i-1) != LLENA):
                        mas_de_uno = False
                    if(i < COLS-1 and tablero.getCelda(x, i+1) != LLENA):
                        mas_de_uno = False
                if(mas_de_uno):
                    num_huecoY+=1
                    for variable in variables:
                        if(inicio == variable.getInicio() and (i,x) == variable.getFinal()):
                            añadir = False
                            break
                            
                    if añadir:
                        variables.append(Variable(str(num_huecoY) +"Y",inicio,(i,x),"vertical"))
                inicio= None
                    
    return variables 
######################################################################### 
# Sacar las restricciones de cada variable
#########################################################################
def sacarRestricciones(variables): 
    for i, variable in enumerate(variables):
        variables_restantes = variables.copy()
        variables_restantes.pop(i) 
        celdaX, celdaY = variable.getInicio()
        for j in range(variable.longitud()):
            for variable_r in variables_restantes:
                actualX, actualY = variable_r.getInicio()
                finX, finY = variable_r.getFinal()
                if(variable_r.getDireccion() == "horizontal"):
                    while((actualX,actualY) != (finX+1,finY)):
                        if((actualX,actualY) == (celdaX,celdaY)):
                            variable.setRestricciones(Restriccion((celdaX,celdaY), variable_r, (actualX,actualY)))
                        actualX += 1
                else:
                    while((actualX,actualY) != (finX,finY+1)):
                        if((actualX,actualY) == (celdaX,celdaY)):
                            variable.setRestricciones(Restriccion((celdaX,celdaY), variable_r, (actualX,actualY)))
                        actualY += 1
     
            if(variable.getDireccion() == "horizontal"):
                celdaX += 1
            else:
                celdaY += 1
        variables_restantes.insert(i,variable)
######################################################################### 
# Sacar dominio de las variables
#########################################################################
def sacarDominio(variables, almacen,tablero):
    factible = True
    for hueco in variables:
        palabras = []
        columna,fila = hueco.getInicio()
        for dom in almacen:
            if dom.tam == hueco.longitud():
                for palabra in dom.getLista():
                    factible = True
                    for i in range(hueco.longitud()):
                        if(hueco.getDireccion() == "horizontal"):
                            if(tablero.getCelda(fila,i) != LLENA and tablero.getCelda(fila,i) != VACIA):
                                if(palabra[i] != tablero.getCelda(fila,i)):
                                    factible = False
                        else:
                            if(tablero.getCelda(i,columna) != LLENA and tablero.getCelda(i,columna) != VACIA):
                                if(palabra[i] != tablero.getCelda(i,columna)):
                                    factible = False
                    if(factible):
                        palabras.append(palabra)
                hueco.setDominio(palabras)
######################################################################### 
# Algoritmo AC3
#########################################################################
def AC3(variables):
    restricciones = []
    for hueco in variables: 
        for restriccion in hueco.getRestricciones():
            restricciones.append((hueco,restriccion))
    i=1
    while restricciones:
        i+=1
        variableX, restriccion = restricciones.pop(0)
        variableY = restriccion.getVariable()
        cambio = False
        dominioX = variableX.getDominio()
        lista_domX = dominioX.copy()
        for valor in lista_domX:
            posicionX = restriccion.getCeldaXAfectada(variableX)
            posicionY = restriccion.getCeldaYAfectada()
            dominioY = variableY.getDominio()
            lista_domY = dominioY.copy()
            for palabra in dominioY:
                if(valor[posicionX] != palabra[posicionY]):
                    lista_domY.remove(palabra)
            if not lista_domY:
                variableX.getDominio().remove(valor)
                cambio = True
        if not variableX.getDominio():
            return False
    
        if cambio:
            for restriccion in variableX.getRestricciones():
                restricciones.append((variableX,restriccion))
######################################################################### 
# Algoritmo FC
#########################################################################
def FC(variables,indice, N):     
    variable_actual = variables[0]
    variables_pendientes = variables[1:]
    dominio = variable_actual.getDominio()
    idx = 0
    while idx < len(dominio):
        palabra = dominio[idx]
        variable_actual.setPalabra(palabra)
        if N-1 == indice:
            return True
        else: 
            if forward(variable_actual,palabra, variables_pendientes):
                idx+=1
                if FC(variables_pendientes,indice+1,N):
                    return True
                else:
                    restaura(variable_actual)
            else:
                restaura(variable_actual)
                variable_actual.getDominio().remove(palabra)
                variable_actual.setPodados(palabra)
                
    if(indice!=0):
        variable_actual.setDominio(variable_actual.getPodados())
        variable_actual.getPodados().clear()
    return False
                   
    
def forward(variable, valor, variables):
    for i,restriccion in enumerate(variable.getRestricciones()):
        variableY = restriccion.getVariable()
        if variableY in variables:
            posicionX = restriccion.getCeldaXAfectada(variable)
            posicionY = restriccion.getCeldaYAfectada()
            dominio = variableY.getDominio()
            lista_dom = dominio.copy()
            for palabra in lista_dom:
                if(valor[posicionX] != palabra[posicionY]):
                    dominio.remove(palabra)
                    variableY.setEliminados((variable.getNombre(),palabra))
            if not dominio:
                return False      
    return True

def restaura(variable):
    dominio = []
    variable.setPalabra("")
    for restriccion in variable.getRestricciones():
        variableY = restriccion.getVariable()
        eliminados = variableY.getEliminados()
        copia = eliminados.copy()
        for eliminado in copia:
            if(eliminado[0] == variable.getNombre()):
                dominio.append(eliminado[1])
                eliminados.remove(eliminado)
        variableY.setDominio(dominio)
        dominio = [] 

#########################################################################  
# Principal
#########################################################################
def main():
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
    variables = []
    while not game_over:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:               
                game_over=True
            if event.type==pygame.MOUSEBUTTONUP:                
                #obtener posición y calcular coordenadas matriciales                               
                pos=pygame.mouse.get_pos()
                if pulsaBotonAC3(pos, anchoVentana, altoVentana):
                    variables = sacarVariables(tablero)
                    sacarDominio(variables,almacen,tablero)
                    sacarRestricciones(variables)
                    res = AC3(variables)
                    if res==False:
                        MessageBox.showwarning("Alerta", "No hay solución")
                elif pulsaBotonFC(pos, anchoVentana, altoVentana):
                    if not variables:
                        variables = sacarVariables(tablero)
                        sacarDominio(variables,almacen,tablero)
                        sacarRestricciones(variables)
                    res = FC(variables,0, len(variables))
                    if res==False:
                        MessageBox.showwarning("Alerta", "No hay solución")
                    else:
                        for variable in variables:
                            longitud = variable.longitud()
                            columna, fila = variable.getInicio()
                            for i in range(longitud):
                                if(variable.getDireccion() == "horizontal"):
                                    tablero.setCelda(fila, columna+i, variable.palabra[i].upper())
                                else:
                                    tablero.setCelda(fila+i, columna, variable.palabra[i].upper())
                elif pulsaBotonReset(pos, anchoVentana, altoVentana):
                    variables = []
                    tablero.reset()
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
 
