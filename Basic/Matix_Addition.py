import theano
from theano import tensor as T
import numpy as np

x = T.dmatrix('x')
y = T.dmatrix('y')

z = x+y

f = theano.function([x,y],z)

#main
print f([[1,2],[3,4]],[[2,2],[2,2]])