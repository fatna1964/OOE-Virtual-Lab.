
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Scaled domain for clear curvature
# =========================
y_min, y_max = 0.0, 1.0
x_limit = 6.0

# =========================
# Superior profile: n(y)=n_ground - k*y
# =========================
n_ground = 1.05

def subsample_keep_last(x, y, stride):
    idx = list(range(0, len(x), stride))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)
    idx = np.array(idx, dtype=int)
    return x[idx], y[idx]

def make_n_of_y(k):
    def n_of_y(y):
        return n_ground - k * y
    return n_of_y

def trace_superior_up_turn_down(theta_deg, n_of_y, x0=0.0, y0=0.10, ds=0.01, max_steps=200000):
    theta0 = np.deg2rad(theta_deg)
    n0 = n_of_y(y0)
    if n0 <= 0:
        raise ValueError("n(y0) <= 0. Adjust k/n_ground/y0.")

    C = n0 * np.sin(theta0)

    x, y = x0, y0
    xs, ys = [x], [y]

    dy_sign = +1.0
    turned = False
    y_turn = None

    for _ in range(max_steps):
        n = n_of_y(y)
        if n <= 0:
            break

        sin_theta = C / n

        if sin_theta >= 1.0:
            sin_theta = 1.0 - 1e-6
            if not turned:
                turned = True
                y_turn = y
            dy_sign = -1.0

        theta = np.arcsin(sin_theta)

        dx = ds * np.sin(theta)
        dy = dy_sign * ds * np.cos(theta)

        x_new = x + dx
        y_new = y + dy

        if y_new <= y_min:
            y_new = y_min
            x_new = x + dx
            xs.append(x_new); ys.append(y_new)
            break

        if y_new >= y_max:
            y_new = y_max
            x_new = x + dx
            xs.append(x_new); ys.append(y_new)
            if dy_sign > 0:
                break
            break

        x, y = x_new, y_new
        xs.append(x); ys.append(y)

        if x >= x_limit:
            break

    return np.array(xs), np.array(ys), turned, y_turn, C

def make_window(theta_deg, color, k, y0=0.10, ds=0.01, stride=2, interval=35):
    n_of_y = make_n_of_y(k)

    x, y, turned, y_turn, C = trace_superior_up_turn_down(theta_deg, n_of_y, y0=y0, ds=ds)
    x, y = subsample_keep_last(x, y, stride)
    N = len(x)

    fig, (ax_ray, ax_n) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Superior Mirage — theta={theta_deg}° | k={k:.4f}", fontsize=13)

    # Ray plot
    ax_ray.set_title("Ray path (UP → TURN → DOWN)")
    ax_ray.set_xlabel("x")
    ax_ray.set_ylabel("y")
    ax_ray.set_xlim(0, x_limit)
    ax_ray.set_ylim(y_min - 0.02, y_max + 0.02)
    ax_ray.grid(True)
    ax_ray.axhline(y_min, color="black", linewidth=1)

    if turned and y_turn is not None:
        ax_ray.axhline(y_turn, linestyle="--", linewidth=1, color=color)
        ax_ray.text(0.02, 0.92, f"turn y≈{y_turn:.3f}", transform=ax_ray.transAxes, color=color)

    line, = ax_ray.plot([], [], color=color, linewidth=2)
    dot,  = ax_ray.plot([], [], "o", color=color, markersize=6)

    # n(y) plot
    ys_plot = np.linspace(y_min, y_max, 400)
    ns_plot = n_of_y(ys_plot)

    ax_n.set_title("n(y) (higher at ground)")
    ax_n.set_xlabel("n(y)")
    ax_n.set_ylabel("y")
    ax_n.set_ylim(y_min - 0.02, y_max + 0.02)
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
            f"n(y)={n_now:.4f}\n"
            f"C={C:.4f}\n"
            f"Turned={turned}"
        )
        return line, dot, marker, text_info

    anim = FuncAnimation(fig, update, frames=np.arange(0, N),
                         init_func=init, interval=interval, blit=True)

    fig.tight_layout()
    return fig, anim

# =====================================================
# Choose k_red so that turning point is at y_turn_target for theta=78
# =====================================================
y0 = 0.10
theta_red = 78
y_turn_target = 0.7

# We want: n(y_turn_target) = C = n(y0)*sin(theta)
# with n(y) = n_ground - k*y
# => k = (n_ground - C) / y_turn_target
# BUT C depends on k via n(y0)=n_ground - k*y0, so solve explicitly:
# C = (n_ground - k*y0)*sin(theta)
# k = (n_ground - C)/y_turn_target
# => k = (n_ground - (n_ground - k*y0)*sinθ) / y_turn_target
# => k*(y_turn_target - y0*sinθ) = n_ground*(1 - sinθ)
# => k = n_ground*(1 - sinθ) / (y_turn_target - y0*sinθ)

sin_th = np.sin(np.deg2rad(theta_red))
den = (y_turn_target - y0 * sin_th)
if den <= 0:
    raise ValueError("Bad target. Choose y_turn_target larger or y0 smaller.")
k_red = n_ground * (1 - sin_th) / den

# blue ray: keep a moderate k (it may not turn much, that’s normal)
k_blue = k_red * 0.6

# =====================================================
# Two windows
# =====================================================
fig12, anim12 = make_window(12, "blue", k=k_blue, y0=y0, ds=0.01, stride=2, interval=35)
fig78, anim78 = make_window(78, "red",  k=k_red,  y0=y0, ds=0.01, stride=2, interval=35)

plt.show()
