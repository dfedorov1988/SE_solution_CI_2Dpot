import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt("x.dat")
Y = np.loadtxt("y.dat")
psiI = np.loadtxt("psiI.dat")
psiJ = np.loadtxt("psiJ.dat")

fig = plt.figure(figsize=(5, 6))

ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax1.plot_surface(X, Y, psiI, cmap="winter", edgecolor='black',
                 alpha=1.0, linewidth=0.25)

ax2 = fig.add_subplot(2, 1, 2, projection='3d')
ax2.plot_surface(X, Y, psiJ, cmap="autumn_r", edgecolor='black',
                 alpha=1.0, linewidth=0.25)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Density")

ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
ax1.set_zticks([0, 0.05, 0.1, 0.15])

ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')

ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim3d([0, 0.18])

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Density")

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('w')
ax2.yaxis.pane.set_edgecolor('w')
ax2.zaxis.pane.set_edgecolor('w')

ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
ax2.set_zticks([0, 0.05, 0.1, 0.15])

ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim3d([0, 0.18])

plt.show()
