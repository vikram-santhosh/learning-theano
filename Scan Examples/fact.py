import theano
from theano import tensor as T
import numpy as np

i = T.ivector('i')

def fact(prior,i):
	return prior*i

results,updates = theano.scan(fn=fact,n_steps=i.shape[0],sequences=i,outputs_info=np.int64(1))

f = theano.function(inputs = [i],outputs = results,updates=updates)

print f(range(1,10))