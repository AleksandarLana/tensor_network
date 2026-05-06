import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn


def fibonacci_potential(L, lam=1.0, omega=0.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    alpha = 1.0 / phi

    V = np.zeros(L)

    for i in range(L):
        x = ((i + 1) * alpha + omega) % 1.0
        if x >= 1.0 - alpha:
            V[i] = lam
        else:
            V[i] = 0.0

    return V


def build_xy_mpo(L, V):
    """
    Builds the finite XY chain

        H = - sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + sum_i V_i Z_i

    with open boundary conditions.
    """
    builder = qtn.SpinHam1D(S=1 / 2)

    for i in range(L - 1):
        builder[i, i + 1] += -1.0, "X", "X"
        builder[i, i + 1] += -1.0, "Y", "Y"

    for i, Vi in enumerate(V):
        builder[i] += float(Vi), "Z"

    return builder.build_mpo(L)


def run_dmrg(H, D=64, nsweeps=10, cutoff=1e-10):
    dmrg = qtn.DMRG2(H)

    dmrg.solve(
        max_sweeps=nsweeps,
        bond_dims=[D],
        cutoffs=[cutoff],
        verbosity=1,
    )

    return float(dmrg.energy), dmrg.state


def average_xx_correlations(psi, max_r=35, window=20):
    """
    Computes averaged absolute XX correlations

        C(r) = average_i | <X_i X_{i+r}> |

    using sites near the center of the chain.

    In this XY chain with Z-field, <X_i> is expected to vanish by symmetry,
    so this is effectively the connected XX correlation.
    """
    X = qu.spin_operator("X", S=1 / 2).real

    L = psi.L
    center = L // 2

    distances = []
    correlations = []

    for r in range(1, max_r + 1):
        vals = []

        i_min = max(0, center - window)
        i_max = min(L - r, center + window)

        for i in range(i_min, i_max):
            j = i + r

            Cij = psi.correlation(X, i, j, B=X)

            vals.append(abs(complex(Cij)))

        distances.append(r)
        correlations.append(np.mean(vals))

    return np.array(distances), np.array(correlations)

def fit_exponential(r, C):
    """
    Fit

        C(r) ~ A exp(-b r)

    equivalently

        log C(r) ~ log A - b r.
    """
    mask = C > 1e-14
    r = r[mask]
    y = np.log(C[mask])

    slope, intercept = np.polyfit(r, y, 1)

    b = -slope
    A = np.exp(intercept)
    xi = 1.0 / b if b != 0 else np.inf

    return A, b, xi


def fit_anomalous_grid(r, C, gammas=np.linspace(0.4, 2.5, 211)):
    """
    Fit

        C(r) ~ A exp(-b r^gamma)

    by scanning over gamma and doing a linear least-squares fit
    for log C(r) against r^gamma.
    """
    mask = C > 1e-14
    r = r[mask]
    y = np.log(C[mask])

    best = None

    for gamma in gammas:
        x = r ** gamma
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = intercept + slope * x
        error = np.mean((y - y_fit) ** 2)

        if best is None or error < best["error"]:
            best = {
                "gamma": gamma,
                "slope": slope,
                "intercept": intercept,
                "error": error,
            }

    gamma = best["gamma"]
    b = -best["slope"]
    A = np.exp(best["intercept"])

    return A, b, gamma, best["error"]


def plot_results(results):
    plt.figure()

    for name, data in results.items():
        r = data["r"]
        C = data["C"]

        plt.semilogy(r, C, marker="o", label=f"{name}: DMRG data")

        A, b, xi = data["exp_fit"]
        plt.semilogy(
            r,
            A * np.exp(-b * r),
            linestyle="--",
            label=f"{name}: exp fit, xi={xi:.2f}",
        )

        A2, b2, gamma, err = data["anom_fit"]
        plt.semilogy(
            r,
            A2 * np.exp(-b2 * r**gamma),
            linestyle=":",
            label=f"{name}: anomalous fit, gamma={gamma:.2f}",
        )

    plt.xlabel("Distance r")
    plt.ylabel("Average correlation |<X_i X_{i+r}>|")
    plt.title("XX correlations: uniform vs Fibonacci XY chain")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("xx_correlation_decay_uniform_vs_fibonacci.png", dpi=300)


def plot_minus_log(results):
    plt.figure()

    for name, data in results.items():
        r = data["r"]
        C = data["C"]

        mask = C > 1e-14
        plt.plot(r[mask], -np.log(C[mask]), marker="o", label=name)

    plt.xlabel("Distance r")
    plt.ylabel("- log |<X_i X_{i+r}>|")
    plt.title("Decay diagnostic: linear means ordinary exponential")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("minus_log_xx_correlation_decay.png", dpi=300)


L = 100
D = 64
max_r = 35

V_uniform = np.ones(L) * 1.0
V_fibonacci = fibonacci_potential(L, lam=1.0, omega=0.0)

models = {
    "uniform XY": V_uniform,
    "Fibonacci XY": V_fibonacci,
}

results = {}

for name, V in models.items():
    print("\n===================================")
    print("Running:", name)
    print("===================================")

    H = build_xy_mpo(L, V)

    E, psi = run_dmrg(
        H,
        D=D,
        nsweeps=10,
        cutoff=1e-10,
    )

    S_mid = float(psi.entropy(L // 2))

    r, C = average_xx_correlations(
        psi,
        max_r=max_r,
        window=20,
    )

    exp_fit = fit_exponential(r, C)
    anom_fit = fit_anomalous_grid(r, C)

    results[name] = {
        "energy": E,
        "middle_entropy": S_mid,
        "r": r,
        "C": C,
        "exp_fit": exp_fit,
        "anom_fit": anom_fit,
    }

    A, b, xi = exp_fit
    A2, b2, gamma, err = anom_fit

    print("Energy:", E)
    print("Energy per site:", E / L)
    print("Middle-cut entropy:", S_mid)
    print("Exponential fit: xi =", xi)
    print("Anomalous fit: gamma =", gamma)
    print("Anomalous fit error:", err)


plot_results(results)
plot_minus_log(results)

print("\nFinished.")
print("Saved:")
print("  xx_correlation_decay_uniform_vs_fibonacci.png")
print("  minus_log_xx_correlation_decay.png")