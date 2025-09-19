import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# -----------------------------
#     CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_SEED = 0


# -----------------------------
#  PARAMETERS
# -----------------------------

# Inner ascent (x)
N_STARTS     = 100
ASCENT_STEPS =  30
ASCENT_STEP  = 1e-2

# Outer descent (W)
OUTER_STEPS        = 450
W_STEPS_PER_OUTER  = 10
LR_W               = 1e-4
GRAD_CLIP          = 1.0

PLOT_EVERY   = 100   # set 0 to disable plots


# -----------------------------
#    ENERGY FUNCTION
# -----------------------------

def energy(z: torch.Tensor) -> torch.Tensor:
    x, y = z[..., 0], z[..., 1]
    return (x**2 - 1.0)**2 * (x**2 - 4.0)**2 + (y**2 - 1.0)**2 + (y**2 - 4.0)**2


# -----------------------------
#   NEURAL NETWORK
# -----------------------------

class SaddlePointNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SaddlePointNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)   
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)   

    def forward(self, x_in):
        # First layer: Project to S^1 (not trainable)
        norm = torch.norm(x_in, p=2, dim=-1, keepdim=True)
        s1_input = x_in / (norm + 1e-6)  # epsilon for stability

        # Trainable layers
        x = torch.relu(self.fc1(s1_input))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)

        # Ensure the network is an odd function: N(x) - N(-x)
        output_neg_x = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(-s1_input)))))
        return output - output_neg_x

# -----------------------------
#   TRAINING LOOP INNER MAX
# -----------------------------

def inner_maximize_x_manual(net: nn.Module,
                            n_starts=N_STARTS,
                            steps=ASCENT_STEPS,
                            step_size=ASCENT_STEP,
                            device=DEVICE):
    
    # Random starts in R^2; net normalizes internally to S^1
    x = torch.randn(n_starts, 2, device=device, requires_grad=True)

    # Manual ascent loop; DO NOT track grads for W (weights are frozen outside)
    for _ in range(steps):
        z = net(x)
        e = energy(z)
        (g_x,) = torch.autograd.grad(e.sum(), x, create_graph=False)
        x = (x + step_size * g_x).detach().requires_grad_(True)  # ascent step

    with torch.no_grad():
        e_final = energy(net(x))
        idx = torch.argmax(e_final)       # pick the best start
        x_max = x[idx:idx+1]
        e_max = float(e_final[idx].item())  # E(x_max)
    return x_max, e_max


# Heat map plotting with rope overlay
def sample_circle(n, device=DEVICE):
    theta = torch.linspace(0.0, 2*math.pi, n, device=device)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

def rope_points(net: nn.Module, n=800):
    with torch.no_grad():
        u = sample_circle(n)
        z = net(u)
    return z.detach().cpu().numpy()

# -----------------------------
#   PLOTTING
# -----------------------------

def plot_state(net: nn.Module, step, show=True,
               grid_min=-3.0, grid_max=3.0, grid_res=500,
               norm_mode="log",         # "log" | "power" | "linear"
               gamma=0.5,               # only used if norm_mode=="power"
               vmin_q=0.02, vmax_q=0.98,# clip colors to these quantiles
               add_contours=True, n_levels=15):
    xs = torch.linspace(grid_min, grid_max, grid_res)
    ys = torch.linspace(grid_min, grid_max, grid_res)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    Z_t = energy(torch.stack([X, Y], dim=-1))
    Z = Z_t.cpu().numpy()

    # Choose a color normalization
    if norm_mode == "log":
        # E2 >= 0; add tiny epsilon for strict positivity
        norm = mpl.colors.LogNorm(vmin=max(1e-6, np.quantile(Z, vmin_q)),
                                  vmax=np.quantile(Z, vmax_q))
    elif norm_mode == "power":
        norm = mpl.colors.PowerNorm(gamma=gamma,
                                    vmin=np.quantile(Z, vmin_q),
                                    vmax=np.quantile(Z, vmax_q))
    else:
        norm = mpl.colors.Normalize(vmin=np.quantile(Z, vmin_q),
                                    vmax=np.quantile(Z, vmax_q))

    rope = rope_points(net, n=800)

    plt.figure(figsize=(6.8, 6.8))
    im = plt.imshow(
        Z.T,
        extent=[grid_min, grid_max, grid_min, grid_max],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        norm=norm,                # << key change
        cmap="viridis"            # pick any perceptual cmap you like
    )
    plt.colorbar(im, label="E(x, y)")

    plt.plot(rope[:, 0], rope[:, 1], linewidth=2.3, color="white", label="N(S^1)")

    # Mark known critical points for this energy
    a = math.sqrt(2.5)

    # Saddles: (±1,0), (±2,0), (0,±a), (±a,±a)
    sx = [ 1, -1,  2, -2,  0,  0,  a, -a,  a, -a]
    sy = [ 0,  0,  0,  0,  a, -a,  a,  a, -a, -a]
    plt.scatter(sx, sy, s=45, marker='x', c="white", label="saddles")

    # Local maxima: (0,0), (±a, 0)
    mx = [0,  a, -a]
    my = [0,  0,  0]
    plt.scatter(mx, my, s=30, facecolors='none', edgecolors='white', label="local maxima")

    plt.title(f"Rope N(S^1) — step {step} (heat map)")
    plt.xlim(grid_min, grid_max); plt.ylim(grid_min, grid_max)
    plt.legend(loc="upper right")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
#  TRAINING LOOP OUTER MIN
# -----------------------------

def outer_minimize_w(net: nn.Module,
                     opt: torch.optim.Optimizer,
                     x_max: torch.Tensor,
                     steps=W_STEPS_PER_OUTER,
                     clip_grad=GRAD_CLIP):
    """
    Fix x_max and update the weights to minimize E(N(x_max)).
    """
    last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = energy(net(x_max)).mean()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()
        last = float(loss.item())
    return last

# -----------------------------
# TRAINING MAIN FUNCTION
# -----------------------------

def train():
    torch.manual_seed(TORCH_SEED)

    net = SaddlePointNet(hidden_dim=64).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR_W) #lr is the learning rate

    if PLOT_EVERY:
        plot_state(net, step=0) #initial state of N

    for t in range(1, OUTER_STEPS + 1):
        # 1) Inner max over x with weights frozen
        for p in net.parameters(): 
            p.requires_grad_(False)
        x_max, e_max = inner_maximize_x_manual(net)

        # --- NEW: log location of the max point on the rope ---
        with torch.no_grad():
            z_max = net(x_max)
            zx, zy = float(z_max[0,0]), float(z_max[0,1])
            print(f"[{t:04d}] hard-max E: {e_max:.4f} at ({zx:.3f}, {zy:.3f})")

        # 2) Outer min over W at x_max
        for p in net.parameters(): 
            p.requires_grad_(True)
        e_after = outer_minimize_w(net, opt, x_max)

        # Logging (optional: keep or remove the old line)
        if t % 20 == 0:
            print(f"[{t:04d}] after min-step: {e_after:.4f}")

        # Plot
        if PLOT_EVERY and (t % PLOT_EVERY == 0):
            plot_state(net, step=t)


    print("Done.")

if __name__ == "__main__":
    train()
