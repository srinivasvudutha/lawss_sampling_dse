import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w_nodes = 0.5
w_time = 0.25
w_dist = 0.25

all = pd.read_csv('output.csv', delimiter = ',')
print(all)
df = all[['IU', 'T', 'nodes', 'sim_time_s', 'distance_m']].to_numpy()

grouped = all.groupby(['IU', 'T'])

means = grouped.mean().reset_index()
mins = grouped.min().reset_index()
print(mins)

# normalize (important so scales don't dominate)
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

IU = means['IU']
T = means['T']
nodes = means['nodes']
sim_time = means['sim_time_s']
distance = means['distance_m']
nodes_n = norm(nodes)          # want max
time_n = norm(sim_time)        # want min
dist_n = norm(distance)        # want min
score = (w_nodes * nodes_n - w_time * time_n - w_dist * dist_n)
best_idx = np.argmax(score)
best_IU = IU[best_idx]
best_T = T[best_idx]
print(f'Best: IU = {best_IU}, T = {best_T}')

fig_heat = plt.figure()
ax_heat = fig_heat.add_subplot()
ax_heat.tricontourf(IU, T, score, levels=15)

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax1.scatter(means['IU'], means['T'], means['nodes'])
ax1.set_xlabel('IU')
ax1.set_ylabel('T')
ax1.set_zlabel('nodes')

ax2.scatter(means['IU'], means['T'], means['sim_time_s'])
ax2.set_xlabel('IU')
ax2.set_ylabel('T')
ax2.set_zlabel('sum_time_s')

ax3.scatter(means['IU'], means['T'], means['distance_m'])
ax3.set_xlabel('IU')
ax3.set_ylabel('T')
ax3.set_zlabel('distance_m')
plt.show()