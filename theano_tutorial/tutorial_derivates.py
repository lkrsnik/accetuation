import numpy as np
import theano
import theano.tensor as T

# normal gradient
x = T.dscalar('x')
z = T.dscalar('z')
y = x ** 3 + z ** 2
gy = T.grad(y, [x, z])

f = theano.function([x, z], gy)

# print(theano.pp(f.maker.fgraph.outputs[0]))
# print(theano.pp(f.maker.fgraph.outputs[1]))

print(f(4, 8))

# logistic gradient
x = T.dmatrix('x')
l = T.sum(1 / (1 + T.exp(-x)))
gl = T.grad(l, x)

f_lg = theano.function([x], gl)

print(f_lg([[0, 1], [-1, -2]]))

# np.matrix([[1, 2], [3, 4]])

# jacobian matrix
print('jacobian matrix1')
x = T.dvector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
f = theano.function([x], J, updates=updates)
print(f([1, 2, 3, 4, 5]))

# already implemented jacobian matrix
# W, V = T.dmatrices('W', 'V')
J = theano.gradient.jacobian(y, x)
f2 = theano.function([x], J)
print(f2([1, 2, 3, 4, 5]))

# jacobian matrix with matrix :)
W, V = T.dmatrices('W', 'V')
x = T.dvector('x')
y = T.dot(x, W)
J = theano.gradient.jacobian(y, W)
f2 = theano.function([W, x], J)
print(f2(np.array([[1, 1], [1, 1]]), np.array([0, 1])))

JV2 = T.dot(J, V)
f2 = theano.function([W, V, x], JV2)
print(f2(np.array([[1, 1], [1, 1]]),  np.array([[2, 2], [2, 2]]), np.array([0, 1])))


print('jacobian matrix2')
x = T.dvector('x')
z = T.dvector('z')
y = x ** 2 + z ** 2
J, updates = theano.scan(lambda i, y, x, z: T.grad(y[i], [x, z]), sequences=T.arange(y.shape[0]), non_sequences=[y,x,z])
f = theano.function([x, z], J, updates=updates)
test = T.arange(y.shape[0])
t_f = theano.function([x, z], test)
print(f([4, 4], [1, 1]))
print(t_f([4, 4], [1, 1]))

# hessian matrix
x = T.dvector('x')
y = x ** 3
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy, x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = theano.function([x], H, updates=updates)
print(f([4, 4]))

# jacobian times vector

# R-operator
W = T.dmatrix('W')
V = T.dmatrix('V')
x = T.dvector('x')
y = T.dot(x, W)
JV = T.Rop(y, W, V)
f = theano.function([W, V, x], JV)
print(f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1]))

# L-operator
W = T.dmatrix('W')
v = T.dvector('v')
x = T.dvector('x')
y = T.dot(x, W)
VJ = T.Lop(y, W, v)
f = theano.function([v,x], VJ)
print(f([2, 2], [0, 1]))