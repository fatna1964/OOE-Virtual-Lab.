import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================
# INFERIOR MIRAGE (3D) with VERY CLEAR curvature
# n(y) = n_ground + k*y   (n higher aloft, lower near hot ground)
# We CHOOSE k so that the 78° ray turns at y_turn_target (very visible).
# =====================================================

# Scaled height so curvature is visible
y_min, y_max = 0.0, 1.0
x_limit = 6.0

# Base refractive index at ground (scaled demo)
n_ground = 1.0

# Ray settings
theta_deg = 78.0  # big angle from vertical
phi_deg = 35.0    # azimuth (3D direction in x-z plane)
y0 = 1.0          # start at top
x0, z0 = 0.0, 0.0

# Choose turning height for the BIG-angle ray (make this ~0.05..0.30 for strong bend)
y_turn_target = 0.10

sin_th = np.sin(np.deg2rad(theta_deg))

# For inferior: n(y)=n_ground + k*y
# Turning occurs when n(y_turn)=C, and C = n(y0)*sin(theta) = (n_ground + k*y0)*sin_th
# Solve: n_ground + k*y_turn = (n_ground + k*y0)*sin_th
# => k*(y_turn - y0*sin_th) = n_ground*(sin_th - 1)
# => k = n_ground*(sin_th - 1) / (y_turn - y0*sin_th)
den = (y_turn_target - y0 * sin_th)
if den == 0:
    raise ValueError("Bad y_turn_target. Choose a different value.")
k = n_ground * (sin_th - 1.0) / den
if k <= 0:
    raise ValueError(
        f"Computed k <= 0 (k={k}). Choose y_turn_target < sin(theta) (~{sin_th:.3f})."
    )

def n_of_y(y):
    return n_ground + k * y

# =====================================================
# 3D ray trace: start DOWN, at turning point go UP
# Invariant: C = n(y) * sin(theta)   (theta from vertical)
# =====================================================
def trace_inferior_3d(theta0_deg, phi_deg, x0, y0, z0, ds=0.01, max_steps=400000):
    theta0 = np.deg2rad(theta0_deg)
    phi = np.deg2rad(phi_deg)

    hx, hz = np.cos(phi), np.sin(phi)

    C = n_of_y(y0) * np.sin(theta0)

    x, y, z = x0, y0, z0
    xs, ys, zs = [x], [y], [z]

    dy_sign = -1.0  # start going DOWN
    turned = False
    y_turn = None

    for _ in range(max_steps):
        n = n_of_y(y)
        sin_theta = C / n

        # Turning point
        if sin_theta >= 1.0:
            sin_theta = 1.0 - 1e-6
            if not turned:
                turned = True
                y_turn = y
            dy_sign = +1.0  # go UP

        theta = np.arcsin(sin_theta)

        dx = ds * np.sin(theta) * hx
        dz = ds * np.sin(theta) * hz
        dy = dy_sign * ds * np.cos(theta)

        x_new, y_new, z_new = x + dx, y + dy, z + dz

        # stop at ground
        if y_new <= y_min:
            y_new = y_min
            x_new = x + dx
            z_new = z + dz
            xs.append(x_new); ys.append(y_new); zs.append(z_new)
            break

        # stop at top after turning (so we clearly see the arc)
        if y_new >= y_max and dy_sign > 0:
            y_new = y_max
            x_new = x + dx
            z_new = z + dz
            xs.append(x_new); ys.append(y_new); zs.append(z_new)
            break

        x, y, z = x_new, y_new, z_new
        xs.append(x); ys.append(y); zs.append(z)

        if x >= x_limit:
            break

    return np.array(xs), np.array(ys), np.array(zs), turned, y_turn, C

# Generate ray
xs, ys, zs, turned, y_turn, C = trace_inferior_3d(theta_deg, phi_deg, x0, y0, z0, ds=0.01)

# Speed up animation a bit
stride = 2
idx = np.arange(0, len(xs), stride)
if idx[-1] != len(xs) - 1:
    idx = np.append(idx, len(xs) - 1)

xs, ys, zs = xs[idx], ys[idx], zs[idx]
N = len(xs)

# =====================================================
# Plots: 3D + 2D projection (x-y) + n(y)
# =====================================================
fig = plt.figure(figsize=(15, 5))

ax3d = fig.add_subplot(1, 3, 1, projection="3d")
ax2d = fig.add_subplot(1, 3, 2)
axn  = fig.add_subplot(1, 3, 3)

fig.suptitle(
    f"Inferior Mirage (BIG angle) — θ={theta_deg}°, target turn y≈{y_turn_target:.2f}\n"
    f"Computed k={k:.4f} (bigger k => stronger curvature)",
    fontsize=12
)

# --- 3D setup ---
ax3d.set_title("3D path")
ax3d.set_xlabel("x")
ax3d.set_ylabel("y (height)")
ax3d.set_zlabel("z")
ax3d.set_xlim(0, x_limit)
ax3d.set_ylim(y_min, y_max)
ax3d.set_zlim(zs.min() - 0.2, zs.max() + 0.2)
ax3d.view_init(elev=18, azim=-60)

ray3d_line, = ax3d.plot([], [], [], linewidth=2)
ray3d_dot = ax3d.scatter([], [], [], s=35)

# --- 2D (x-y) setup (THIS makes curvature obvious) ---
ax2d.set_title("2D projection (x–y) — curvature should be CLEAR")
ax2d.set_xlabel("x")
ax2d.set_ylabel("y")
ax2d.set_xlim(0, x_limit)
ax2d.set_ylim(y_min - 0.02, y_max + 0.02)
ax2d.grid(True)
ax2d.axhline(y_min, color="black", linewidth=1)

ray2d_line, = ax2d.plot([], [], linewidth=2)
ray2d_dot,  = ax2d.plot([], [], "o", markersize=6)

# --- n(y) plot ---
ygrid = np.linspace(y_min, y_max, 400)
ngrid = n_of_y(ygrid)
axn.set_title("n(y) profile (Inferior: higher aloft)")
axn.set_xlabel("n(y)")
axn.set_ylabel("y")
axn.plot(ngrid, ygrid, linewidth=2)
axn.grid(True)
marker, = axn.plot([], [], "o", markersize=7)
info = axn.text(0.02, 0.98, "", transform=axn.transAxes, va="top")

def init():
    ray3d_line.set_data([], [])
    ray3d_line.set_3d_properties([])
    ray3d_dot._offsets3d = ([], [], [])
    ray2d_line.set_data([], [])
    ray2d_dot.set_data([], [])
    marker.set_data([], [])
    info.set_text("")
    return ray3d_line, ray3d_dot, ray2d_line, ray2d_dot, marker, info

def update(i):
    i = min(i, N - 1)

    # 3D
    ray3d_line.set_data(xs[:i+1], ys[:i+1])
    ray3d_line.set_3d_properties(zs[:i+1])
    ray3d_dot._offsets3d = ([xs[i]], [ys[i]], [zs[i]])

    # 2D x-y
    ray2d_line.set_data(xs[:i+1], ys[:i+1])
    ray2d_dot.set_data([xs[i]], [ys[i]])

    # n(y)
    n_now = n_of_y(ys[i])
    marker.set_data([n_now], [ys[i]])

    info.set_text(
        f"Frame {i+1}/{N}\n"
        f"y={ys[i]:.3f}\n"
        f"n(y)={n_now:.4f}\n"
        f"C={C:.4f}\n"
        f"Turned={turned}"
    )
    return ray3d_line, ray3d_dot, ray2d_line, ray2d_dot, marker, info

anim = FuncAnimation(fig, update, frames=np.arange(0, N), init_func=init, interval=25, blit=False)

plt.tight_layout()
plt.show()
