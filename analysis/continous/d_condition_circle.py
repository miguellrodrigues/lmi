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

q = 3
r = 2.5

LMI = cvx.bmat([
  [-r*P, q*P + A@P],
  [q*P + P@A.T, -r*P]
])

epsilon = 1e-6
constraints = [
  P >> cvx.Constant(np.eye(n) * epsilon),
  LMI << -cvx.Constant(np.eye(2*n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)


def plot():
  eig_vals = np.linalg.eigvals(A.value)

  # plot the eigenvalues
  plt.figure()
  plt.scatter(eig_vals.real, eig_vals.imag, marker='x', color='red')
  plt.xlabel('Real')
  plt.ylabel('Imaginary')
  plt.title('Eigenvalues of A')

  # plot a circle centered at q with radius r
  theta = np.linspace(0, 2 * np.pi, 100)
  x = r * np.cos(theta) - q
  y = r * np.sin(theta)

  plt.plot(x, y, color='blue')
  plt.show()


try:
  prob.solve()
except cvx.SolverError:
  print('The poles of the system are not located in the region denoted by the parameters alpha and beta')

  plot()
else:
  if prob.status == 'optimal':
    print(' ')
    print("status:", prob.status)
    print(' ')
    print("P\n", P.value)
    print(' ')

    plot()


