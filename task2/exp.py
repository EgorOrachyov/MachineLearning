import numpy as np
import scipy.sparse as ss

a = np.zeros((5,1))
b = np.zeros((5,2))

print(a.transpose().dot(b))
print(a.transpose() + 1)

a[1][0] = 10
b[1][0] = 2
b[1][1] = 4

aa = ss.csc_matrix(a)
bb = ss.csc_matrix(b)

cc = aa.multiply(bb)

print(np.asarray([[1,1]]))
print(b + cc)
print(b + 0.5)

print(aa.todense())
print(bb.todense())
print(aa.multiply(bb).todense())

print(0.25 * np.asarray([[2]] * 4).__pow__(4))

print(np.asarray([[1] * 10] * 4))
print(np.add.reduce(np.asarray([[1] * 10] * 4), axis=1).reshape((4,1)))

r = np.add.reduce(np.asarray([1,2,3,4]))
print(r)

m = ss.csc_matrix(np.asarray([[1,2],[2,3],[3,10]]))
print(m.transpose().todense())
print(m.sum(axis=0))


# V, w, w0
# x
#
#
# w0' = 1
# w' = x
# Vif' = xi * dot(x,V*f) - xi^2 * Vif =>
# V*f' = x * dot(x,V*f) - diag(x^2) * V*f   =>   dot(x,V) - вектор (dot(x,V*1) ... dot(x,V*k))
# V' =  x * dot(x,V) - mult(x^2,V)
