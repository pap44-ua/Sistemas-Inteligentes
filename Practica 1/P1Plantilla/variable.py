from dominio import *

class Variable:
    def __init__(self, coorInicio, coorFin, nombre):
        self.coorInicio=coorInicio
        self.coorFin = coorFin
        self.tam=self.longitud()
        self.dominio=[]
        self.nombre=nombre
        self.palabra=""
        self.restriccion=[]
    
    def horizontal (self):
        dif_x = abs(self.coorInicio[0] - self.coorFin[0])  # Diferencia en el eje x
        dif_y = abs(self.coorInicio[1] - self.coorFin[1])  # Diferencia en el eje y

        # Si la diferencia en el eje x es 0 y la diferencia en el eje y es mayor que 0, entonces es horizontal
        if dif_x == 0 and dif_y > 0:
            return True
        # Si la diferencia en el eje y es 0 y la diferencia en el eje x es mayor que 0, entonces es vertical
        elif dif_y == 0 and dif_x > 0:
            return False
    
    def longitud (self):
        if(self.horizontal()):
            return  (self.coorFin[1]+1)-self.coorInicio[1]
        return (self.coorFin[0]+1) - self.coorInicio[0]
    
    def getDominio(self):
        return self.dominio
    
    def setDominio(self, newDominio):
        self.dominio=newDominio
        
    def getNombre(self):
        return self.nombre
    
    def setNombre(self, newNombre):
        self.nombre=newNombre
        
    def getRestriccion(self):
        return self.restriccion
    
    def setRestriccion(self, newRestriccion):
        self.restriccion=newRestriccion      

    def setNombre(self, nombre):
        self.nombre=nombre
    def getNombre(self):
        return self.nombre
    def setPalabra(self, nombre):
        self.nombre=nombre
    def getPalabra(self):
        return self.palabra
    
    def getCoorIni(self):
        return self.coorInicio
    
    def getCoorFin(self):
        return self.coorFin
        
        
