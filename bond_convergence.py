import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn


def run_dmrg(L, J, h, max_bond_dimension, nsweeps=8, cutoff=1e-12):

    H = qtn.MPO_ham_ising(
        L=L,
        j=-J,
        bx=h,
        S=0.5,
        cyclic=False,
    )

    dmrg = qtn.DMRG2(H)

    dmrg.solve(
        max_sweeps=nsweeps,
        bond_dims=[max_bond_dimension],
        cutoffs=[cutoff],
        verbosity=0,
    )

    return float(dmrg.energy), dmrg.state


def middle_cut_entropy(psi):

    cut = psi.L // 2

    return float(psi.entropy(cut))


L = 200
h = 1.0
J = 1.0

Ds = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

energies = []
entropies = []

for D in Ds:

    print(f"Running DMRG with D = {D}")

    E, psi = run_dmrg(
        L=L,
        J=J,
        h=h,
        max_bond_dimension=D,
        nsweeps=8,
        cutoff=1e-12,
    )

    energies.append(E / L)
    entropies.append(middle_cut_entropy(psi))

energies = np.array(energies)
entropies = np.array(entropies)

E_ref = energies[-1]
S_ref = entropies[-1]

energy_error = np.abs(energies - E_ref)
entropy_error = np.abs(entropies - S_ref)

plt.figure()
plt.loglog(Ds, energy_error, marker="o")
plt.xlabel("Maximum bond dimension D")
plt.ylabel("|e(D) - e(D_ref)|")
plt.title(f"Energy-density convergence, L={L}, J={J}, h={h}")
plt.grid(True, which="both")
plt.savefig("energy_convergence.png", dpi=300)

plt.figure()
plt.semilogx(Ds, entropies, marker="o")
plt.xlabel("Maximum bond dimension D")
plt.ylabel("Middle-cut entanglement entropy")
plt.title(f"Entropy vs bond dimension, L={L}, J={J}, h={h}")
plt.grid(True)
plt.savefig("entropy_vs_D.png", dpi=300)

plt.figure()
plt.loglog(Ds, entropy_error, marker="o")
plt.xlabel("Maximum bond dimension D")
plt.ylabel("|S(D) - S(D_ref)|")
plt.title(f"Entropy convergence, L={L}, J={J}, h={h}")
plt.grid(True, which="both")
plt.savefig("entropy_convergence.png", dpi=300)

print("\nFinished successfully.")
print("Saved:")
print("  energy_convergence.png")
print("  entropy_vs_D.png")
print("  entropy_convergence.png")