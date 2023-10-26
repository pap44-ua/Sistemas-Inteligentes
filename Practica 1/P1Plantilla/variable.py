

class Variable:
    def __init__(self, coorInicio, coorFin):
        self.coorInicio=coorInicio
        self.coorFin = coorFin
    
    def horizontal (self):
        if (self.coorInicio[0]- self.coorFin[0])==0:
            return True
        return False
    
    def longitud (self):
        if(self.horizontal()):
            return  (self.coorFin[1]+1)-self.coorInicio[1]
        return (self.coorFin[0]+1) - self.coorInicio[0]
