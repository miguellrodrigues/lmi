import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

A = np.array([
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [-5.5, -2.5, -5, -1],
  [-2.5, -5.5, -1, -5]
])

A = cvx.Constant(A)
n = A.shape[0]

P = cvx.Variable((n, n), PSD=True)

epsilon = 1e-6
constraints = [
  P >> cvx.Constant(np.eye(n) * epsilon),
  A.T@P + P@A << -cvx.Constant(np.eye(n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)

prob.solve()

if prob.status == 'optimal':
  print(' ')
  print("status:", prob.status)
  print(' ')
  print("P\n", P.value)
  print(' ')
else:
  print('The system in unstable')
