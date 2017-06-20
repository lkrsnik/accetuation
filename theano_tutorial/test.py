import numpy as np
import theano.tensor as T
from theano import function

# ALGEBRA
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)

# print(f(2, 3))
# print(numpy.allclose(f(16.3, 12.1), 28.4))
print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

# exercise
import theano
a = T.vector()                                  # declare variable
b = T.vector()                                  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b               # build symbolic expression
f = function([a, b], out)                       # compile function
print(f([1, 2], [4, 5]))

###################################################
# OTHER EXAMPLES

# logistic function
x = T.dmatrix('x')
logistic_eq = 1 / (1 + T.exp(-x))
logistic = function([x], logistic_eq)
print(logistic([[0, 1], [-1, -2]]))


# multiple things calculation
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])
print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))


# default value
c = T.matrix('c')
c = a + b
f = function([a, theano.In(b, value=[[1, 1], [1, 1]])], c)
print(f([[1, 1], [1, 1]]))


# accumulator
state = theano.shared([[0, 0], [0, 0]])
print("accumulator")
print(state.get_value())


state = theano.shared(np.matrix('0 0; 0 0', dtype=np.int32))
print(type(np.matrix('0 0; 0 0', dtype=np.int64)))
print(type(np.matrix('0 1; 2 3', dtype=np.int64)))
inc = T.imatrix('inc')
expression = state+inc
print(type(expression))
accumulator = function([inc], state, updates=[(state, state+inc)])


accumulator(np.matrix('1 2; 3 4', dtype=np.int32))
print(state.get_value())
accumulator(np.matrix('1 1; 1 1', dtype=np.int32))
print(state.get_value())

# function copy
print("function copy")
new_state = theano.shared(np.matrix('0 0; 0 0', dtype=np.int32))
new_accumulator = accumulator.copy(swap={state: new_state})
new_accumulator(np.matrix('1 2; 3 4', dtype=np.int32))
print(new_state.get_value())
print(state.get_value())

# random numbers
# POSSIBLE THAT THIS DOES NOT WORK ON GPU
print("random numbers")
srng = T.shared_randomstreams.RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)     # Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

print(f())
print(f())

print(g())
print(g())

print("sharing streams between functions")
state_after_v0 = rv_u.rng.get_value().get_state()
# nearly_zeros()       # this affects rv_u's generator
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1

print(v1)
print(v2)
print(v3)
