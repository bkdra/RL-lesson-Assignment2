import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -----------------------------
# Trajectory 1
# -----------------------------
# traj1 = np.array([
#     [0,  0, 10.5],
#     [2,  0, 10.5],
#     [4,  2, 10.5],
#     [4,  4, 10.5],
#     [2,  6, 10.5],
#     [0,  6, 10.5],
#     [-2, 4, 10.5],
#     [-2, 2, 10.5],
#     [0,  0, 10.5],
# ])
traj1 = np.array([
    [0,  0, 10.5],
    [3,  0, 10.5],
    [6,  2, 10.5],
    [8,  5, 11.5],
    [6,  8, 11.0],
    [2, 10, 10.5],
    [-2, 8, 10.5],
    [-4, 5, 10.5],
    [-2, 2, 10.5],
    [2,  4, 10.5],
    [5,  6, 11.0],
    [0,  0, 10.5],
])
# -----------------------------
# Trajectory 2
# -----------------------------

traj2 = np.array([
    [ 0, -1.5, 10.2],
    [ 3, -1.5, 10.2],
    [ 6, 1.5, 10.2],
    [ 6, 4.5, 10.2],
    [ 3, 7.5, 10.2],
    [ 0, 7.5, 10.2],
    [-3, 4.5, 10.2],
    [-3, 1.5, 10.2],
    [ 0, -1.5, 10.2]
])
# traj2 = np.array([
#     [-0.8, -2.0, 10.2 ],
#     [ 3.4, -2.0, 10.2 ],
#     [ 7.6, 0.8, 10.2 ],
#     [10.4, 5.0, 10.2 ],
#     [ 7.6, 9.2, 10.2 ],
#     [ 2.0, 12.0, 10.2 ],
#     [-3.6, 9.2, 10.2 ],
#     [-6.4, 5.0, 10.2 ],
#     [-3.6, 0.8, 10.2 ],
#     [ 2.0, 3.6, 11.35 ],
#     [ 6.2, 6.4, 10.925],
#     [-0.8, -2.0, 10.2 ]
# ])
# -----------------------------
# Spline function
# -----------------------------
def spline_interpolate(traj, num_points=300):
    # 用 arc-length parameterization（避免速度不均）
    diff = np.diff(traj, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    t = np.insert(np.cumsum(dist), 0, 0)
    t = t / t[-1]

    cs_x = CubicSpline(t, traj[:, 0])
    cs_y = CubicSpline(t, traj[:, 1])
    cs_z = CubicSpline(t, traj[:, 2])

    t_new = np.linspace(0, 1, num_points)

    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)

    return x_new, y_new, z_new


# -----------------------------
# Interpolate
# -----------------------------
x1, y1, z1 = spline_interpolate(traj1)
x2, y2, z2 = spline_interpolate(traj2)

# -----------------------------
# Save trajectories (6 columns per line)
# -----------------------------
traj1_out = np.column_stack((x1, y1, z1))
traj2_out = np.column_stack((x2, y2, z2))
combined_out = np.column_stack((traj1_out, traj2_out))

np.savetxt("trajectory2_noFace_new.txt", combined_out, fmt="%.6f")

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1, y1, z1, label="Trajectory 1 spline")
ax.scatter(traj1[:,0], traj1[:,1], traj1[:,2])

ax.plot(x2, y2, z2, label="Trajectory 2 spline")
ax.scatter(traj2[:,0], traj2[:,1], traj2[:,2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()