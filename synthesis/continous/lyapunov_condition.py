import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

A = np.array([
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [-5.5, -2.5, 5, -1],
  [-2.5, -5.5, -1, 5]
])

B = np.array([
  [0, 0],
  [0, 0],
  [0, 1.],
  [1., 0]
])

A = cvx.Constant(A)
B = cvx.Constant(B)

n = A.shape[0]
m = B.shape[1]

W = cvx.Variable((n, n), symmetric=True)
Z = cvx.Variable((m, n))

epsilon = 1e-6
constraints = [
  W >> cvx.Constant(np.eye(n) * epsilon),
  A@W + B@Z + W@A.T + Z.T@B.T << -cvx.Constant(np.eye(n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)

prob.solve()

if prob.status == 'optimal':
  P = np.linalg.inv(W.value)
  K = Z.value@P

  print(' ')
  print("status:", prob.status)
  print(' ')
  print("K ", K)
  print('\nClosed Loop Poles ', np.linalg.eigvals(A.value + B.value@K))
  print(' ')
else:
  print('The system in unstable')
