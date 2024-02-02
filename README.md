# Resource Bounds for Quantum Circuit Mapping via Quantum Circuit Complexity

## Citation:
This repository hosts the accompanying software for the following research article and master thesis. Please consider citing:

```
@article{steinberg2024resource,
  title={Resource Bounds for Quantum Circuit Mapping via Quantum Circuit Complexity},
  author={Steinberg, Matthew and Bandic, Medina and Szkudlarek, Sacha and Almudever, Carmen G and Sarkar, Aritra and Feld, Sebastian},
  journal={arXiv preprint arXiv:2402.00478},
  year={2023}
}
```
#### Abstract:
Efficiently mapping quantum circuits onto hardware is an integral part of the quantum compilation process, wherein a quantum circuit is modified in accordance with the stringent architectural demands of a quantum processor. Many techniques exist for solving the quantum circuit mapping problem, many of which relate quantum circuit mapping to classical computer science. This work considers a novel perspective on quantum circuit mapping, in which the routing process of a simplified circuit is viewed as a composition of quantum operations acting on density matrices representing the quantum circuit and processor. Drawing on insight from recent advances in quantum information theory and information geometry, we show that a minimal SWAP gate count for executing a quantum circuit on a device emerges via the minimization of the distance between quantum states using the quantum Jensen-Shannon divergence. Additionally, we develop a novel initial placement algorithm based on a graph similarity search that selects the partition nearest to a graph isomorphism between interaction and coupling graphs. From these two ingredients, we then construct a polynomial-time algorithm for calculating the SWAP gate lower bound, which is directly compared alongside the IBM Qiskit compiler for over 600 realistic benchmark experiments, as well as against a brute-force method for smaller benchmarks. In our simulations, we unambiguously find that neither the brute-force method nor the Qiskit compiler surpass our bound, implying utility as a precise estimation of minimal overhead when realizing quantum algorithms on constrained quantum hardware. This work constitutes the first use of quantum circuit uncomplexity to practically-relevant quantum computing. We anticipate that this method may have diverse applicability outside of the scope of quantum information science, and we discuss several of these possibilities.

```
@article{szkudlarek2023determining,
  title={Determining Minimal SWAP Operations for the Qubit-Mapping Problem using Quantum Information Theory},
  author={Szkudlarek, Sacha},
  year={2023}
}
```
#### Abstract:
This thesis presents a novel formulation to study the qubit-mapping problem (QMP). The presented for- mulation redefines the problem in terms of density matrices which represent the quantum algorithm and the underlying architecture—allowing the implementation of techniques from quantum information theory to es- tablish a bounded metric space for comparing these density matrices. The main contribution of this thesis is implementing this formulation in an algorithm to determine the minimal bound on the required number of SWAP operations for a pairing of a quantum algorithm to an underlying device where the initial mapping has been provided. Benchmarks have shown a clear dependence on the β-value. Emphasising the need for future investigations of this dependence to enhance the algorithm’s effectiveness for more extensive algorithms and architectures. While it is essential to acknowledge that the approach may not currently rival the state of the art.
