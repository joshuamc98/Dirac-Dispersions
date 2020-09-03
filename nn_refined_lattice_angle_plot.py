import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
from mpl_toolkits.mplot3d import Axes3D

t_nn = 1 #Energy Hopping Scale
a = 0.142E-9 #Using Graphene scale
point_scale = 100
decay_factor = 1
plot_number = 3
plot_scale = 3

start = time.time()


def h_ab(k_x, k_y):

    h_ab = 2 * t_nn * np.cos((a * np.cos(lattice_angle)) * k_x +  (a * np.sin(lattice_angle)) * k_y)

    return h_ab

def h_ac(k_x, k_y):

    h_ac = 2 * t_nn * np.cos(a * k_x)

    return h_ac

def h_bc(k_x, k_y):

    h_bc = 2 * t_nn * math.exp(-1 * decay_factor * (2 * np.cos((math.pi - lattice_angle) / 2) - 1)) * np.cos(2 * a * pow(np.cos((math.pi - lattice_angle) / 2), 2) * k_x - 2 * a * np.sin((math.pi - lattice_angle) / 2) * np.cos((math.pi - lattice_angle) / 2) * k_y)

    return h_bc


def diag_plot(lattice_angle):

    k_x = np.linspace(-1 * (math.pi) / (plot_scale * a) , math.pi / (plot_scale * a) , point_scale)
    k_y = np.linspace(-1 * (math.pi) / (plot_scale * a) , math.pi / (plot_scale * a) , point_scale)
    lattice_angle_deg = round(lattice_angle * 180 / math.pi, 1)

    E_1 = np.zeros((point_scale, point_scale))
    E_2 = np.zeros((point_scale, point_scale))
    E_3 = np.zeros((point_scale, point_scale))

    K_x, K_y = np.meshgrid(k_x, k_y)

    for i in range(0, point_scale, 1):
        for j in range(0, point_scale, 1):
            m = np.array([[0, h_ab(k_x[i], k_y[j]), h_ac(k_x[i], k_y[j])],
                   [h_ab(k_x[i], k_y[j]), 0, h_bc(k_x[i], k_y[j])],
                    [h_ac(k_x[i], k_y[j]), h_bc(k_x[i], k_y[j]), 0]])
            eigenvalues = np.linalg.eigvals(m)
            eigenvalues.sort()
            E_1[j][i] = eigenvalues[0]
            E_2[j][i] = eigenvalues[1]
            E_3[j][i] = eigenvalues[2]



    fig = plt.figure()
    ax = plt.gca(projection = '3d')


    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlim3d(-3, 3)
    ax.view_init(elev = 15, azim = 315)
    ax.set_xlabel('$k_xa$ (DIMENSIONLESS)', fontsize = 8)
    ax.set_ylabel('$k_ya$ (DIMENSIONLESS)', fontsize = 8)
    ax.set_zlabel('$E/t_{nn}$ (DIMENSIONLESS)', fontsize = 8)
    #ax.set_title(f'{lattice_angle_deg} degree lattice electronic bands assuming nearest neighbour tight binding', fontsize = 10)



    #Plot_1 = ax.plot_surface(K_x * a, K_y * a, E_3 / t_nn, cmap = 'Reds_r')
    Plot_2 = ax.plot_surface(K_x * a, K_y * a, E_2 / t_nn, cmap = 'Blues_r')
    Plot_3 = ax.plot_surface(K_x * a, K_y * a, E_1 / t_nn, cmap = 'Greens_r')

    plt.savefig(f'{lattice_angle_deg}d_nn.png')

for x in range(0, plot_number + 1):
    lattice_angle = math.pi/3 + (x * math.pi) / (6 * plot_number)
    diag_plot(lattice_angle)
    print(f"Plotted {x + 1} out of {plot_number + 2}")

end = time.time()

t = end - start
seconds = round(t % 60)
minutes = round(t / 60)
hours = round(t / 3600)

print(f"Time Elapsed: {hours} hours, {minutes} minutes, {seconds} seconds")
