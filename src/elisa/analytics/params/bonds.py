import numpy as np

# keept order of allowed constraints method or came with better way how to validaite and evaluate fitting constraints
ALLOWED_CONSTRAINT_METHODS = ['arcsin', 'arccos', 'arctan', 'log', 'sin', 'cos', 'tan', 'exp', 'degrees', 'radians']
ALLOWED_CONSTRAINT_CHARS = ['(', ')', '+', '-', '*', '/', '.', 'e'] + [str(i) for i in range(0, 10, 1)]
TRANSFORM_TO_METHODS = ['arcs', 'arcc', 'arct', 'log', 'sin', 'cos', 'tan', 'exp', 'deg', 'rad']

arcs = np.arcsin
arcc = np.arccos
arct = np.arctan

sin = np.sin
cos = np.cos
tan = np.tan

exp = np.exp
log = np.log
log10 = np.log10

deg = np.degrees
rad = np.radians
