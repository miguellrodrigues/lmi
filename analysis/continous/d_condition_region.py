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

P = cvx.Variable((n, n), symmetric=True)

# region parameters
# H{alpha, beta} = {x+jy | -beta < x < -alpha}

alpha = .1
beta = 4.4

epsilon = 1e-6
constraints = [
  P >> cvx.Constant(np.eye(n) * epsilon),
  A.T @ P + P @ A + 2 * alpha * P << -cvx.Constant(np.eye(n) * epsilon),
  -A.T @ P - P @ A - 2 * beta * P << -cvx.Constant(np.eye(n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)

try:
  prob.solve()
except cvx.SolverError:
  print('The poles of the system are not located in the region denoted by the parameters alpha and beta')
  print(np.linalg.eigvals(A.value))
else:
  if prob.status == 'optimal':
    print(' ')
    print("status:", prob.status)
    print(' ')
    print("P\n", P.value)
    print(' ')


