from theano import function as F
from theano import tensor as T
import numpy as np

x = T.dvector('x')
y = T.dvector('y')

z1 = x+y
z2 = x*y
z3 = T.dot(x,y)

f1 = F([x,y],z1)
f2 = F([x,y],z2)
f3 = F([x,y],z3)
 
#main
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print f1(v1,v2)
print f2(v1,v2)
print f3(v1,v2)