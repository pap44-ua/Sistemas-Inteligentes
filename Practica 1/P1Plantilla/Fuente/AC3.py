from collections import deque
from restriccion import Restriccion
from variable import Variable

def AC3(variables):
    queue = deque([(vi, vj) for vi in variables for vj in vi.getRestriccion()])

    print("DOMINIOS ANTES DEL AC3")
    for var in variables:
        print(f"Variable {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()}): {var.getDominio()}")

    while queue:
        (Xi, restriccion) = queue.popleft()
        Xj = restriccion.getY()
        if revise(Xi, Xj):
            if not Xi.getDominio():
                return False
            for restr in Xi.getRestriccion():
                if restr.getY() != Xj:
                    queue.append((restr.getY(), restr))

    print("DOMINIOS DESPUÉS DEL AC3")
    for var in variables:
        print(f"Variable {var.getNombre()} ({var.getCoorIni()} -> {var.getCoorFin()}): {var.getDominio()}")

    return True

def revise(Xi, Xj):
    revised = False
    for x in Xi.getDominio()[:]:  # Use slice to create a copy of the list for safe removal
        if all(not consistent(x, y, Xi, Xj) for y in Xj.getDominio()):
            Xi.getDominio().remove(x)
            revised = True
            print(f"Removed {x} from {Xi.getNombre()} because no value in {Xj.getNombre()} is consistent with it.")
    return revised

def consistent(x, y, Xi, Xj):
    intersection = None
    for restr in Xi.getRestriccion():
        if restr.getY() == Xj:
            intersection = restr.coor
            break
    
    if intersection is None:
        return False
    
    if Xi.horizontal():
        j = intersection[1] - Xi.getCoorIni()[1]
        i = intersection[0] - Xj.getCoorIni()[0]
    else:
        i = intersection[0] - Xi.getCoorIni()[0]
        j = intersection[1] - Xj.getCoorIni()[1]
    
    consistent = x[j] == y[i]
    print(f"Checking consistency between {x} (in {Xi.getNombre()}) and {y} (in {Xj.getNombre()}): {consistent}")
    return consistent

def neighbors(Xi, Xj, variables):
    return [Xk for Xk in variables if Xk != Xi and Xj in [res.getY() for res in Xk.getRestriccion()]]
