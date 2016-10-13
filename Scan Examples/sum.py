import theano
from theano import tensor as T
import numpy as np

x = T.bvector('x')

def add(prior,x):
	return prior+x

results,updates = theano.scan(fn = add,n_steps = x.shape[0],sequences=x,outputs_info=0)

f = theano.function(inputs=[x],outputs=results[-1],updates=updates)

print f(xrange(10))