# MPS/DMRG Warmup: Transverse-Field Ising Chain

This project implements a first tensor-network calculation using Matrix Product States (MPS) and the Density Matrix Renormalization Group (DMRG) algorithm via the library `quimb`. The aim is to approximate the ground state of a one-dimensional quantum spin system without explicitly constructing the full exponentially large Hilbert space.

We consider a chain of length L = 20, where each site carries a spin-1/2 degree of freedom. The Hilbert space is

$\mathcal{H} = (\mathbb{C}^2)^{\otimes L}$

so its dimension is $2^L = 2^{20} = 1,048,576.$ A generic state in this space therefore requires over a million complex amplitudes, and this number grows exponentially with system size. Direct numerical methods based on the full Hilbert space quickly become infeasible.

The Hamiltonian used in the computation is the transverse-field Ising model,

$$
H = -J \sum_{i=1}^{L-1} Z_i Z_{i+1} - h \sum_{i=1}^{L} X_i
$$

where $X_i$ and $Z_i$ denote Pauli matrices acting on site $i$, and we take $J = 1$ and $h = 1$. Open boundary conditions are imposed, meaning that there is no interaction between the first and last sites.

In the implementation, this Hamiltonian is constructed as:

    H = qtn.MPO_ham_ising(L, j=1.0, bx=1.0, cyclic=False)

The crucial point is that the Hamiltonian is not represented as a full $2^L \times 2^L$ matrix. Instead, it is stored as a Matrix Product Operator (MPO), which is a tensor-network representation of operators. This allows one to work with the Hamiltonian in a compressed form that reflects its locality, avoiding the exponential blow-up of the full matrix representation.

A matrix product state is a corresponding tensor-network representation of a quantum state. Instead of assigning a complex number $\psi_{i_1 \dots i_L}$ to each configuration of spins, one writes the state as a product of tensors:

$$
\psi_{i_1 \dots i_L} = A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[L]}_{i_L}
$$

Each tensor carries both a physical index and auxiliary indices that are contracted between neighbouring sites. The size of these auxiliary indices is called the bond dimension $D$, and it determines how much entanglement the state can represent.

The DMRG algorithm is a variational method that searches for the ground state within the class of matrix product states. Instead of minimizing the energy over the full Hilbert space, one solves:

$$
\min_{\psi \in \mathrm{MPS}} \langle \psi | H | \psi \rangle
$$

In the code, we use:

    dmrg = qtn.DMRG2(H)
    dmrg.solve(max_sweeps=6, verbosity=1)

The algorithm sweeps along the chain and optimizes neighbouring tensors iteratively.

From the output, the bond dimension grows up to about $D = 16$, meaning the algorithm searches within MPS of limited entanglement.

The final result is:

Ground state energy ≈ -10.6025518  
Energy per site ≈ -0.5301276

The state returned is:

    psi = dmrg.state

This is the approximate ground state in MPS form.

The effectiveness of this approach comes from the structure of ground states of local Hamiltonians. Although the Hilbert space is exponentially large, physically relevant states occupy a much smaller subset characterized by limited entanglement. Matrix product states capture exactly this structure.

## References

White (1992)  
https://doi.org/10.1103/PhysRevLett.69.2863

Schollwöck (2011)  
https://arxiv.org/abs/1008.3477

Hastings (2007)  
https://arxiv.org/abs/0705.2024