import matplotlib
matplotlib.use("Agg")

import quimb.tensor as qtn
import matplotlib.pyplot as plt

L = 20

H = qtn.MPO_ham_ising(L, j=1.0, bx=1.0, cyclic=False)

dmrg = qtn.DMRG2(H)
dmrg.solve(max_sweeps=6, verbosity=1)

energy = dmrg.energy
psi = dmrg.state

print("Ground state energy:", energy)
print("Energy per site:", energy / L)

print("before entropy")

entropies = []
for i in range(1, L):
    print("computing entropy at cut", i)
    entropies.append(psi.entropy(i))

print("after entropy")
print("Entropies:", entropies)

plt.figure()
plt.plot(range(1, L), entropies, marker="o")
plt.xlabel("Bond (cut position)")
plt.ylabel("Entanglement entropy")
plt.title("Entanglement entropy across chain")
plt.tight_layout()
plt.savefig("entropy.png", dpi=300)

print("Saved plot to entropy.png")