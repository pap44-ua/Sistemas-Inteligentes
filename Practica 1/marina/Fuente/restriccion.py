class Restriccion:
    def __init__(self, celdaX, variable, celdaY):
        self.celdaX = celdaX
        self.variable = variable
        self.celdaY = celdaY
    
    def getCeldaX(self):
        return self.celdaX
    
    def getVariable(self):
        return self.variable
    
    def getCeldaY(self):
        return self.celdaY
    
    def getCeldaXAfectada(self,variableX):
        inicio = variableX.getInicio()
        direccion = variableX.getDireccion()
        if(direccion == "horizontal"):
            return self.celdaX[0]-inicio[0]
        else:
            return self.celdaX[1]-inicio[1]
        
    def getCeldaYAfectada(self):
        inicio = self.variable.getInicio()
        direccion = self.variable.getDireccion()
        if(direccion == "horizontal"):
            return self.celdaY[0]-inicio[0]
        else:
            return self.celdaY[1]-inicio[1]