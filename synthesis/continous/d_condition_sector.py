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

# #

W = cvx.Variable((n, n), symmetric=True)
Z = cvx.Variable((m, n))


alpha = .0
theta = np.radians(75)
r = 4.2

LMI1 = cvx.bmat([
  [-r*W, A@W + B@Z],
  [W@A.T + Z.T@B.T, -r*W]
])

LMI2 = cvx.bmat([
  [(A@W + W@A.T + B@Z + Z.T@B.T)*np.sin(theta), (A@W - W@A.T + B@Z - Z.T@B.T)*np.cos(theta)],
  [(W@A.T - A@W + Z.T@B.T - B@Z)*np.cos(theta), (A@W + W@A.T + B@Z + Z.T@B.T)*np.sin(theta)]
])

epsilon = 1e-6
constraints = [
  W >> cvx.Constant(np.eye(n) * epsilon),

  2*alpha*W + A@W + W@A.T + B@Z + Z.T@B.T << -cvx.Constant(np.eye(n) * epsilon),
  LMI1 << -cvx.Constant(np.eye(2*n) * epsilon),
  LMI2 << -cvx.Constant(np.eye(2*n) * epsilon),
]

objective = cvx.Minimize(0)
prob = cvx.Problem(objective, constraints)


def plot(_eig_vals):
  # plot the eigenvalues
  plt.figure()

  plt.scatter(_eig_vals.real, _eig_vals.imag, marker='x', color='red')
  plt.xlabel('Real')
  plt.ylabel('Imaginary')
  plt.title('Eigenvalues of Acl')

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

    plot(eig_vals)

