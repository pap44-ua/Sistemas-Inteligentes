

class Restriccion:
    def __init__(self,coorX,varY,coorY):
        self.coorX=coorX
        self.varY=varY
        self.coorY=coorY

    def getPosX(self):
        return self.coorX
    
    def getY(self):
        return self.varY
    
    def getPosY(self):
        return self.coorY