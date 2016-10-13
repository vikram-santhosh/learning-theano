import theano
from theano import tensor as T
import numpy as np

x = T.vector('x')
k = T.iscalar('k')

def power(prior,x):
	return prior * x

results,updates = theano.scan(
							fn = power,
							non_sequences = x,
							n_steps = k,
							outputs_info = T.ones_like(x))

ans = results[-1]
pow_op = theano.function(inputs = [x,k], outputs = ans,updates=updates)

print pow_op([1,2,3],5)