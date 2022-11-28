import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

plt.style.use([
  'science',
  'notebook',
  'grid'
])

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

alpha = .0
theta = np.pi/4
r = 4.2

LMI1 = cvx.bmat([
  [-r*P, A@P],
  [P@A.T, -r*P]
])

LMI2 = cvx.bmat([
  [(A@P + P@A.T)*np.sin(theta), (A@P - P@A.T)*np.cos(theta)],
  [(P@A.T - A@P)*np.cos(theta), (A@P + P@A.T)*np.sin(theta)]
])

epsilon = 1e-6
constraints = [
  P >> cvx.Constant(np.eye(n) * epsilon),

  2*alpha*P + A@P + P@A.T << -cvx.Constant(np.eye(n) * epsilon),
  LMI1 << -cvx.Constant(np.eye(2*n) * epsilon),
  LMI2 << -cvx.Constant(np.eye(2*n) * epsilon),
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

  # plot a semicircle centered at origin with radius r (dotted style)
  t = np.linspace(0, np.pi, 100)

  x = r * np.cos(t)
  y = r * np.sin(t)

  plt.plot(-y, x, linestyle='dotted', color='black')

  # #
  intersect_x = r * np.cos(np.pi-theta)
  intersect_y = r * np.sin(np.pi-theta)

  # plot a line from 0,0 to (intersect_x, intersect_y)
  plt.plot([0, intersect_x], [0, intersect_y], color='black', linestyle='dashed')
  plt.plot([0, intersect_x], [0, -intersect_y], color='black', linestyle='dashed')

  plt.axvline(x=-alpha, color='black', linestyle='dashed')

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


