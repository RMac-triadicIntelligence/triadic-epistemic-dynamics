import numpy as np
import matplotlib.pyplot as plt  # Optional for plotting; uncomment relevant sections if needed

def hill(x, gamma=10.0, k=0.5, n=6):
    """Hill-type activation function with numerical stability"""
    return gamma * x**n / (k**n + x**n + 1e-12)

def triadic_drift(y, gamma=10.0, k=0.5, n=6, beta=0.2, kappa=2.0, mu=0.8, sigma=0.5, tau=0.1):
    """
    Extended triadic dynamics with dwelling.
    x1, x2, x3: evidence facets (cyclic mutual activation)
    d: dwelling (ambiguity tolerance, rises under low coherence)
    """
    x1, x2, x3, d = y
    m = (x1 + x2 + x3) / 3
    coupling = gamma * (1 + kappa * d)
    decay = beta * (1 - mu * d)
    dx1dt = hill(x2, coupling, k, n) * (1 - x1) - decay * x1
    dx2dt = hill(x3, coupling, k, n) * (1 - x2) - decay * x2
    dx3dt = hill(x1, coupling, k, n) * (1 - x3) - decay * x3
    ddt = sigma * (1 - m) * (1 - d) - tau * d
    return [dx1dt, dx2dt, dx3dt, ddt]

def renyi_entropy(probs, alpha):
    """
    Compute Rényi entropy of order alpha for a discrete probability distribution.
    """
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    if alpha == 1:
        return -np.sum(probs * np.log(probs))
    elif np.isclose(alpha, 0):
        return np.log(len(probs))
    elif np.isinf(alpha):
        return -np.log(np.max(probs))
    else:
        return 1.0 / (1 - alpha) * np.log(np.sum(probs**alpha))

def renyi_spectrum(probs, alphas=None):
    if alphas is None:
        alphas = [0, 0.5, 1, 2, 5, np.inf]
    return [renyi_entropy(probs, a) for a in alphas]

def run_ensemble(
    n_trajectories=1000,
    t_span=np.linspace(0, 50, 500),
    noise_strength=0.05,
    intervention_time=25.0,
    nudge=0.5
):
    """
    Run stochastic ensemble with optional grace nudge.
    """
    dt = t_span[1] - t_span[0]
    trajectories = []
    for _ in range(n_trajectories):
        y = np.zeros((len(t_span), 4))
        y[0] = [0.2, 0.1, 0.15, 0.0]  # low-coherence initial state
        for j in range(1, len(t_span)):
            dy = np.array(triadic_drift(y[j - 1]))
            noise = noise_strength * np.random.randn(4)
            y[j] = y[j - 1] + dy * dt + noise * np.sqrt(dt)
            y[j] = np.clip(y[j], 0.0, 1.0)
            # Grace nudge (guarded)
            if intervention_time is not None:
                if t_span[j - 1] < intervention_time <= t_span[j]:
                    y[j, 0] = min(y[j, 0] + nudge, 1.0)
        trajectories.append(y)
    return np.array(trajectories), t_span

def bin_epistemic_states(trajectories, n_bins=20):
    """
    Histogram epistemic state space (m, d) at each time step.
    """
    hist_range = [(0, 1), (0, 1)]
    probs_over_time = []
    for t_idx in range(trajectories.shape[1]):
        ms = np.mean(trajectories[:, t_idx, :3], axis=1)
        ds = trajectories[:, t_idx, 3]
        hist, _, _ = np.histogram2d(
            ms, ds, bins=n_bins, range=hist_range, density=True
        )
        probs = hist.flatten() + 1e-12
        probs /= probs.sum()
        probs_over_time.append(probs)
    return np.array(probs_over_time)

def analyze_renyi_spectrum(with_intervention=True):
    t = np.linspace(0, 50, 500)
    intervention_time = 25.0 if with_intervention else None
    nudge = 0.5 if with_intervention else 0.0
    traj, _ = run_ensemble(
        n_trajectories=2000,
        t_span=t,
        noise_strength=0.06,
        intervention_time=intervention_time,
        nudge=nudge
    )
    probs_t = bin_epistemic_states(traj, n_bins=20)
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
    spectrum = np.array([renyi_spectrum(p, alphas) for p in probs_t])
    late_avg = spectrum[-50:].mean(axis=0)
    labels = [
        r'$\alpha=0$ (Hartley)',
        r'$\alpha=0.5$',
        r'$\alpha=1$ (Shannon)',
        r'$\alpha=2$ (Collision)',
        r'$\alpha=5$',
        r'$\alpha=\infty$ (Min)'
    ]
    print("Late-time average Rényi entropies:")
    for lab, val in zip(labels, late_avg):
        print(f"  {lab}: {val:.3f}")

    # Optional: Uncomment to plot spectrum evolution
    # plt.figure(figsize=(12, 8))
    # colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    # for i, (label, col) in enumerate(zip(labels, colors)):
    #     plt.plot(t, spectrum[:, i], label=label, color=col, lw=2)
    # if with_intervention:
    #     plt.axvline(intervention_time, color='gray', linestyle=':', lw=2, label='Grace Nudge')
    # plt.xlabel('Time')
    # plt.ylabel(r'Rényi Entropy $H_\alpha$')
    # plt.title(f'Renyi Spectrum Evolution {"with" if with_intervention else "without"} Grace Nudge')
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()

# Execution
if __name__ == "__main__":
    print("=== With Grace Nudge ===")
    analyze_renyi_spectrum(with_intervention=True)
    print("\n=== No Intervention (Collapse?) ===")
    analyze_renyi_spectrum(with_intervention=False)
