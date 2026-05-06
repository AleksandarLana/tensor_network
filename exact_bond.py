import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import quimb as qu
import quimb.tensor as qtn


def build_mpo(L, J=1.0, h=1.0):
    return qtn.MPO_ham_ising(
        L=L,
        j=-J,
        bx=h,
        S=0.5,
        cyclic=False,
    )


def exact_from_mpo(H_mpo):
    H_dense = H_mpo.to_dense()
    evals, evecs = la.eigh(H_dense)
    E_exact = float(evals[0])
    psi_exact = evecs[:, 0]
    return E_exact, psi_exact


def exact_middle_entropy_quimb(psi_exact, L):
    dims = [2] * L
    left_sites = list(range(L // 2))
    return float(qu.entropy_subsys(psi_exact, dims=dims, sysa=left_sites))


def run_dmrg(H_mpo, D, nsweeps=8, cutoff=1e-12):
    dmrg = qtn.DMRG2(H_mpo)

    dmrg.solve(
        max_sweeps=nsweeps,
        bond_dims=[D],
        cutoffs=[cutoff],
        verbosity=0,
    )

    return float(dmrg.energy), dmrg.state


L = 8
J = 1.0
h = 1.0

Ds = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

H_mpo = build_mpo(L, J=J, h=h)

print("Computing exact diagonalization...")
E_exact, psi_exact = exact_from_mpo(H_mpo)

e_exact = E_exact / L
S_exact = exact_middle_entropy_quimb(psi_exact, L)

print("Exact ground state energy:", E_exact)
print("Exact energy per site:", e_exact)
print("Exact middle-cut entropy:", S_exact)

energy_densities = []
entropies = []

for D in Ds:
    print(f"\nRunning DMRG with D = {D}")

    E_dmrg, psi_dmrg = run_dmrg(H_mpo, D=D)

    e_dmrg = E_dmrg / L
    S_dmrg = float(psi_dmrg.entropy(L // 2))

    energy_densities.append(e_dmrg)
    entropies.append(S_dmrg)

    print("DMRG energy per site:", e_dmrg)
    print("DMRG middle entropy:", S_dmrg)


energy_densities = np.array(energy_densities)
entropies = np.array(entropies)


plt.figure()
plt.semilogx(Ds, energy_densities, marker="o", label="DMRG")
plt.axhline(e_exact, color="orange", linestyle="--", linewidth=2.5, label="Exact diagonalization")
plt.xlabel("Maximum bond dimension D")
plt.ylabel("Energy per site")
plt.title(f"Energy density vs bond dimension, L={L}")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("actual_energy_density_vs_exact.png", dpi=300)


plt.figure()
plt.semilogx(Ds, entropies, marker="o", label="DMRG")
plt.axhline(S_exact, color="orange", linestyle="--", linewidth=2.5, label="Exact diagonalization")
plt.xlabel("Maximum bond dimension D")
plt.ylabel("Middle-cut entanglement entropy")
plt.title(f"Middle-cut entropy vs bond dimension, L={L}")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("actual_entropy_vs_exact.png", dpi=300)


print("\nFinished successfully.")
print("Saved:")
print("  actual_energy_density_vs_exact.png")
print("  actual_entropy_vs_exact.png")