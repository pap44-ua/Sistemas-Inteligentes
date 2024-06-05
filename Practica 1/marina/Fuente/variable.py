class Variable:
    def __init__(self, nombre, inicio, final, direccion):
        if(inicio[0] != final[0] and inicio[1] != final[1]):
            raise Exception('direccion invalida')
        self.nombre = nombre
        self.inicio = inicio
        self.final = final  
        self.direccion = direccion
        self.palabra = ""
        self.dominio = [] 
        self.restricciones = []
        self.eliminados = []
        self.podados = []

    def getNombre(self):
        return self.nombre
    
    def getInicio(self):
        return self.inicio
    
    def getFinal(self):
        return self.final
     
    def longitud(self):
        direccion = self.direccion
        if(direccion == "horizontal"):
            return self.final[0] - self.inicio[0] + 1
        elif(direccion == "vertical"):
            return self.final[1] - self.inicio[1] + 1
    
    def setPalabra(self,palabra):
        self.palabra = palabra
        
    def setDominio(self, palabras):
        self.dominio.extend(palabras)

    def setRestricciones(self, restriccion):
        self.restricciones.append(restriccion)
        
    def setEliminados(self, palabra):
        self.eliminados.append(palabra)
        
    def setPodados(self, palabra):
        self.podados.append(palabra)
        
    def getPalabra(self):
        return self.palabra
    
    def getDominio(self):
        return self.dominio
    
    def getDireccion(self):
        return self.direccion
    
    def getRestricciones(self):
        return self.restricciones
    
    def getEliminados(self):
        return self.eliminados
    
    def getPodados(self):
        return self.podados
    

    
        