# quantum-approximate-optimization
Set of tools to implement quantum optimization for classical problems (diagonal Hamiltonians)


## Installation

### Operational system
This project has been tested only on Ubuntu.

### Clone repository

Run the following command wherever you want the repository to be stored

```
git clone https://github.com/usra-riacs/quantum-approximate-optimization
```


### Package manager
We use pixi (https://pixi.sh/) to manage dependencies in our project.
If you don't have it, please install pixi using the following command **on Linux**:

```
curl -fsSL https://pixi.sh/install.sh | sh 
```

(see https://pixi.sh/v0.49.0/installation/ for details and other installation options)


### Install packages using pixi

#### Main packages
Run the following command in the root of the repository

```
pixi install --environment quapopt
```

The above installs all the basic packages required to run the project.

See below for additional packages that can be installed to extend the functionality of the repository.

Note: if you run into problems with dependecies, try removing pixi.lock file and running "install" command again.

#### [Recommended] Additional packages 

To make the best use of the repository, we recommend installing some additional packages, using larger environment instead:

```
pixi install --environment quapopt-full
```

Without the above, some functionalities might not be available.


#### [Optional] Additional packages for GPU support
If you have Nvidia GPU, we highly recommend working with repository's version that supports GPU computation.
To do so, run

```
pixi install --environment quapopt-gpu
```
or
```
pixi install --environment quapopt-gpu-full
```
for additional packages installation.


To verify if the CUDA is properly detected by packages that we use (numba and cupy), you can run

```
pixi run check_cuda
```


NOTE: currently, as far as I know, qiskit-aer-gpu does not properly install with pip. 
To use it, a brave user is advised to build the repository locally and add separate environment and feature in pixi.toml file.

#### Build cpp parts of the project
After setting up the environment, run the following command in the root of the repository

```
pixi run build_cpp
```

to compile some cpp code used in the project.



#### [Optional] Set location of data storage
If you'd like the generated data to be stored in specific location, run 
```
pixi run set_results_directory
```
and enter the relevant directory.

#### [Optional] Support for IDEs
If you use IDE like PyCharm or Microsoft Visual Studio, please refer to https://pixi.sh/dev/integration/editor/jetbrains for instructions on how to integrate pixi with your IDE (the link leads to PyCharm instructions, but others are also supported).



### [Not recommended] Install packages using pip 

It should be also possible to install the repository using pip, but it is not recommended, as it might lead to some issues with dependencies.

For basic installation, in your virtual environment, you would run:

```
pip install -e ./
```

And for full installation, you would run:

```
pip install -e ./[full]
```


Similarly, for gpu, we have

```
pip install -e ./[gpu]
```
or

```
pip install -e ./[full_gpu]
```



## Citing this repo
The following bibtex entry can be used to cite this repository:

@misc{quapopt_repo,
url={https://github.com/usra-riacs/quantum-approximate-optimization}, 
title = {quapopt -- open source GitHub repository for quantum approximate optimization},
author={Maciejewski, F. B. and Bach, B. G., and Biamonte, J. and Hadfield, S.A. and Venturelli, D.}, 
year={2025}, }

## Funding 
Development of significant parts of the repository was supported under the NSF awards #2329097 and #1918549.

## Used repositories 
We use some (refactored) code from the following repositories:
* https://github.com/jpmorganchase/QOKit (fast simulation of low-scale QAOA) under Apache License Version 2.0
* https://github.com/nasa/pysa/ (simulated annealing solver) under Apache License Version 2.0
* https://github.com/aboev/pymqlib (various classical solvers, including Burer-Monteiro algorithm) under The MIT License (MIT)

The relevant licenses for those repos can also be found in both subfolders with forked repos (whenever relevant), and the above links.

## References
This repository is based, among others, on the following papers, which describe some of the algorithms and methods used in the code:

[1] Maciejewski, Filip B., Jacob Biamonte, Stuart Hadfield, and Davide Venturelli. "[Improving quantum approximate optimization by noise-directed adaptive remapping.](https://arxiv.org/abs/2404.01412)" arXiv preprint arXiv:2404.01412 (2024).

[2] Maciejewski, Filip B., Bao G. Bach, Maxime Dupont, P. Aaron Lott, Bhuvanesh Sundar, David E. Bernal Neira, Ilya Safro, and Davide Venturelli. "[A multilevel approach for solving large-scale qubo problems with noisy hybrid quantum approximate optimization.](https://arxiv.org/abs/2408.07793)" In 2024 IEEE High Performance Extreme Computing Conference (HPEC), pp. 1-10. IEEE, 2024.

[3] Maciejewski, Filip B., Stuart Hadfield, Benjamin Hall, Mark Hodson, Maxime Dupont, Bram Evert, James Sud et al. "[Design and execution of quantum circuits using tens of superconducting qubits and thousands of gates for dense Ising optimization problems.](https://arxiv.org/abs/2308.12423)" Physical Review Applied 22, no. 4 (2024): 044074.

[4] Bach, Bao G., Filip B. Maciejewski, and Ilya Safro. "[Solving Large-Scale QUBO with Transferred Parameters from Multilevel QAOA of low depth.](https://arxiv.org/abs/2505.11464)" arXiv preprint arXiv:2505.11464 (2025).

[5] Tam, Wai-Hong, Hiromichi Matsuyama, Ryo Sakai, and Yu Yamashiro. "[Enhancing NDAR with Delay-Gate-Induced Amplitude Damping.]"(https://arxiv.org/abs/2504.12628) arXiv preprint arXiv:2504.12628 (2025).

[6] Lykov, Danylo, Ruslan Shaydulin, Yue Sun, Yuri Alexeev, and Marco Pistoia. "[Fast simulation of high-depth qaoa circuits.](https://arxiv.org/abs/2309.04841)" In Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis, pp. 1443-1451. 2023.

[7] Dupont, Maxime, and Bhuvanesh Sundar. "[Extending relax-and-round combinatorial optimization solvers with quantum correlations.](https://arxiv.org/abs/2307.05821)" Physical Review A 109, no. 1 (2024): 012429.





