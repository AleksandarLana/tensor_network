import quimb as qu
import quimb.tensor as qtn

L = 20

H = qtn.MPO_ham_ising(L, j=1.0, bx=1.0, cyclic=False)

dmrg = qtn.DMRG2(H)
dmrg.solve(max_sweeps=6, verbosity=1)

energy = dmrg.energy
psi = dmrg.state

print("Ground state energy:", energy)
print("Energy per site:", energy / L)