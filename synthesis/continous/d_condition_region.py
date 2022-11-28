import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

plt.style.use([
  'science',
  'notebook',
  'grid'
])

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

# #

alpha = .1
beta = 4

# #

W = cvx.Variable((n, n), symmetric=True)
Z = cvx.Variable((m, n))

epsilon = 1e-6
constraints = [
  W >> cvx.Constant(np.eye(n) * epsilon),

  A @ W + W @ A.T + B @ Z + Z.T @ B.T + 2 * alpha * W << -cvx.Constant(np.eye(n) * epsilon),
  -A @ W - W @ A.T - B @ Z - Z.T @ B.T - 2 * beta * W << -cvx.Constant(np.eye(n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)

try:
  prob.solve()
except cvx.SolverError:
  print('Is not possible to put the system poles in the desired region')
else:
  if prob.status == 'optimal':
    P = np.linalg.inv(W.value)
    K = Z.value @ P

    Acl = (A.value + B.value @ K)
    eig_vals = np.linalg.eigvals(Acl)

    print(' ')
    print("status:", prob.status)
    print(' ')
    print("K ", K)
    print('\nClosed Loop Poles ', eig_vals)
    print(' ')

    plt.figure()
    plt.axvline(-alpha, color='r', linestyle='--')
    plt.axvline(-beta, color='r', linestyle='--')

    plt.scatter(eig_vals.real, eig_vals.imag, marker='x')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Closed Loop Poles')
    plt.show()



