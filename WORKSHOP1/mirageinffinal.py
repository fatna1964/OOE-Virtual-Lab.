
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================
# 1) Physical parameters (Inferior Mirage)
# =====================================================
y_min, y_max = 0.0, 10.0
n_ground = 1.0000
k_vis = 0.035 / y_max   # انحناء واضح بدون مبالغة

def n_of_y(y):
    return n_ground + k_vis * y

# =====================================================
# 2) Ray tracing (down -> turning -> up)
# invariant: n(y)*sin(theta)=C  (theta from vertical)
# =====================================================
def trace_full_ray(y0, x0, theta0_deg, ds=0.05, max_steps=200000):
    theta0 = np.deg2rad(theta0_deg)
    C = n_of_y(y0) * np.sin(theta0)

    x, y = x0, y0
    xs = [x]
    ys = [y]

    dy_sign = -1.0  # start downward

    for _ in range(max_steps):
        n = n_of_y(y)
        sin_theta = C / n

        if sin_theta >= 1.0:
            sin_theta = 1.0 - 1e-6
            dy_sign = +1.0

        theta = np.arcsin(sin_theta)

        dx = ds * np.sin(theta)
        dy = dy_sign * ds * np.cos(theta)

        x_new = x + dx
        y_new = y + dy

        # stop exactly at ground
        if y_new <= y_min:
            denom = (y - y_new)
            t = (y - y_min) / denom if denom != 0 else 0.0
            x = x + t * (x_new - x)
            y = y_min
            xs.append(x)
            ys.append(y)
            break

        x, y = x_new, y_new
        xs.append(x)
        ys.append(y)

        if (y >= y_max) and (dy_sign > 0):
            break

        if x > 80:
            break

    return np.array(xs), np.array(ys)

def subsample_keep_last(x, y, stride):
    idx = list(range(0, len(x), stride))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)
    idx = np.array(idx, dtype=int)
    return x[idx], y[idx]

# =====================================================
# 3) Create one window per angle
# =====================================================
def make_ray_window(theta_deg, color, ds=0.05, stride=2, interval=35):

    x0, y0 = 0.0, y_max
    x, y = trace_full_ray(y0, x0, theta_deg, ds=ds)
    x, y = subsample_keep_last(x, y, stride)

    N = len(x)

    fig, (ax_ray, ax_n) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Ray Simulation — theta = {theta_deg}°", fontsize=13)

    # ----- Ray plot -----
    ax_ray.set_title("Ray path")
    ax_ray.set_xlabel("x")
    ax_ray.set_ylabel("y")
    ax_ray.set_ylim(y_min - 0.2, y_max + 0.2)
    ax_ray.axhline(y_min, color="black", linewidth=1)
    ax_ray.grid(True)
    ax_ray.set_xlim(0, max(5, x.max() * 1.05))

    line, = ax_ray.plot([], [], color=color, linewidth=2)
    dot,  = ax_ray.plot([], [], "o", color=color, markersize=6)

    # ----- n(y) plot -----
    ys_plot = np.linspace(y_min, y_max, 400)
    ns_plot = n_of_y(ys_plot)

    ax_n.set_title("n(y)")
    ax_n.set_xlabel("n(y)")
    ax_n.set_ylabel("y")
    ax_n.set_ylim(y_min - 0.2, y_max + 0.2)
    ax_n.grid(True)
    ax_n.plot(ns_plot, ys_plot, color="gray", linewidth=2)

    marker, = ax_n.plot([], [], "o", color=color, markersize=7)

    text_info = ax_n.text(0.02, 0.98, "", transform=ax_n.transAxes, va="top")

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        marker.set_data([], [])
        text_info.set_text("")
        return line, dot, marker, text_info

    def update(i):
        i = min(i, N - 1)

        line.set_data(x[:i+1], y[:i+1])
        dot.set_data([x[i]], [y[i]])

        n_now = n_of_y(y[i])
        marker.set_data([n_now], [y[i]])

        text_info.set_text(
            f"Frame {i+1}/{N}\n"
            f"y={y[i]:.3f}\n"
            f"n(y)={n_now:.6f}\n"
            f"k={k_vis:.6f}"
        )

        return line, dot, marker, text_info

    anim = FuncAnimation(
        fig, update,
        frames=np.arange(0, N),
        init_func=init,
        interval=interval,
        blit=True
    )

    fig.tight_layout()
    return fig, anim

# =====================================================
# 4) Create two separate windows
# =====================================================

fig12, anim12 = make_ray_window(12, color="blue")
fig78, anim78 = make_ray_window(78, color="red")

plt.show()
