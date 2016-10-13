import numpy as np
from theano import tensor as T
from theano import function

x = T.dscalar('x')	# define a (double) scalar x
y = T.dscalar('y')	# define a (double) scalar y

z = x+y				# symbolic addition

f = function([x,y],z)	# inputs and outputs

#main
print f(5,2)
