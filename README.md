# Topological-Fractionnal-AI
Physics-constrained AI engine replacing backpropagation with topological state-space solvers. Integrates Atangana-Baleanu fractional calculus (Mittag-Leffler kernel) to solve non-Markovian SDEs with strict coupling matrices.

Replacing statistical induction with structural isomorphism.

A novel "Gray-Box" AI architecture that substitutes backpropagation with a physics-constrained topological solver. By integrating Atangana-Baleanu fractional calculus, the model embeds non-local memory into dynamic system modeling, achieving 84.1% AUC with only 28 parameters (vs. 10,000+ for standard GCN/LSTM).

1. The Paradigm Shift
Standard Deep Learning architectures approximate complex systems through blind statistical induction on flat tensors. This results in massive parameter counts, high computational costs, and "hallucinations" (the generation of physically impossible states).
This engine operates on a strict geometric prior. The state space is not learned; it is structurally constrained by causal coupling matrices, turning the neural network into a deterministic solver for Stochastic Differential Equations (SDEs).

2. Mathematical Core
The architecture relies on three fundamental pillars:
A. Orthogonal Spectral Decomposition
The state space H is strictly divided into orthogonal subspaces (E_1…E_n), representing different scales of     viscosity or inertia (e.g., frequency bands in signal processing).
<img width="115" height="69" alt="image" src="https://github.com/user-attachments/assets/d5c2ceaf-56da-4494-8680-991b0ed2ed5e" />
B. Causal Coupling Matrix (M_ij)
Time evolution is not governed by gradient descent, but by a fixed topological matrix defining transduction rules between planes. The drift term becomes a conservation law:
<img width="294" height="67" alt="image" src="https://github.com/user-attachments/assets/3cb70022-2955-48ad-9765-ffc5042ac805" />
(Where α_i represents specific relaxation rates, and M_ij is the causal topology).
C. Non-Local Memory (Atangana-Baleanu Derivative)
To model systems with heavy-tailed asymptotics (complex materials, biological signals), the standard Markovian derivative is replaced by the Atangana-Baleanu fractional derivative in the Caputo sense, utilizing the Mittag-Leffler kernel:
<img width="473" height="61" alt="image" src="https://github.com/user-attachments/assets/6355ebad-e3b4-4f1c-aac0-990f3f7894f5" />
Result: The system intrinsically possesses long-term memory and mathematically forbids the generation of out-of-bounds physical states.

3. Empirical Evidence (Zero-Shot Topology)
Tested on complex, multi-site biomedical signal classification (ABIDE, ADNI public datasets) without data augmentation or hyper-parameter heavy tuning.
<img width="961" height="236" alt="image" src="https://github.com/user-attachments/assets/c3a91c09-1e86-4173-814b-344a82a00698" />
Note: The 99,7% reduction in parameters proves that the geometric prior (M_ij) captures the causal essence of the signal, bypassing the need for statistical brute-forcing.

4.  Repositiry Structure
|
|-- Readme.md
|-- LICENSE    # Strict evaluation/commercial licencing 
|
|-- core/    #The mathematical engine (Hardware agnostic)     
|   |--ab_derivative.py    # Mittag-Leffler kernel & AB calculus
|   |-- topology.py    # M_ij matrix extraction & spectral decomposition
|   |-- solver.py    # Main ODE/SDE solver wrapper 
|
|-- examples/
|   |--eeg_poc.py    # Proof of concept on public physiological data    
|   |--requirements.txt    # Dependencies (numpy, scipy, mne)

5. Quick Strart (Proof of Concept)
This engine requieres no GPU. It runs purely on CPU-based linear algebra.
# Clone the repository
git clone https://github.com/PORTEMANN/Topological-Fractionnal-AI.git
cd Topological-Fractional-AI

# Install dependencies
pip install -r examples/requirements.txt

# Run the Proof of Concept (extracts causal topology from raw signals)
python examples/eeg_poc.py

6. Licensing & Technology Transfer
This repository is provided strictly for algorithmic evaluation, academic peer review, and non-commercial testing.

The underlying mathematical formalisms, the topological mapping logic, and the fractional solver architecture are proprietary intellectual property.

To discuss commercial integration (materials science, fluid dynamics, edge-AI hardware, bio-signal processing) or licensing agreements, please contact: contact@portemann.eu
