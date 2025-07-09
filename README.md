# 1D Bio-Climatic Coupling

This GitHub repository contains all the necessary files to simulate the coupling between biology and climate in the case of methanogens located in a hypothetical Martian ocean at the end of the Noachian period. The coupling is performed here in a one-dimensional case. The biological model is based on Sauterey et al. (2022), and the 1D climate model used is *Exok* (Leconte 2021) (see the [documentation](https://perso.astrophy.u-bordeaux.fr/~jleconte/exo_k-doc/)).

## 1. General Principle of the Coupling
The initial prebiotic state of Mars is based on a scenario studied by Turbet and Forget (2021), to which N₂ has been added (see the red box in the diagram). The prebiotic equilibrium is calculated using *Exok*.

Methanogenic cells are then introduced, and the system is allowed to evolve under the influence of biology. The concentrations of H₂, CH₄, CO₂, and N₂ in the ocean, the concentration of methanogens in the ocean, and the gas ratios in the atmosphere all evolve simultaneously. Once a threshold of CH₄ is reached in the atmosphere, the climate equilibrium is recalculated to stay close to reality, and this process continues. The coupling stops when the model's surface temperature, assumed to be that of the ocean, falls below the ocean’s freezing point. At this point, the ocean is considered frozen and biological activity ceases (see diagram).

<img src="./1D_coupling_diagram.png" alt="Diagram representing 1D coupling" width="30%">

## 2. Description of the Different Folders in This Repository
Four folders are included in this repository. Each of them contains a `README.md` file that provides more detailed explanations of the different codes and files in the folder.

- **Ktables**: Contains the correlated-k tables useful for an atmosphere composed of CH₄, CO₂, N₂, and H₂.
- **ciatables**: Contains the CIA tables necessary for the chemical species present in the atmosphere.
- **Coupling_on_server**: Contains all the code needed to run the coupling on a server, to parallelize it, etc.
- **Local_coupling_computation**: Contains everything required to run the coupling locally on your own computer.
