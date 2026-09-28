# quantum-approximate-optimization
Set of tools to implement quantum optimization for classical problems (diagonal Hamiltonians)


## Installation

### Operating system
This project has been tested mainly on Ubuntu, with limited tests on Windows and MacOS.


### Clone repository

Run the following command wherever you want the repository to be stored

```
git clone https://github.com/usra-riacs/quantum-approximate-optimization
```



### Pre-install steps
#### [MAC] Install XCode
The Cython extensions require a C++ compiler. On macOS, this comes from Xcode Command Line Tools:
```bash
xcode-select --install
```


### Package manager
We use pixi (https://pixi.sh/) to manage dependencies in our project.
If you don't have it, please install pixi using the following command **on Linux**:

```
curl -fsSL https://pixi.sh/install.sh | sh 
```

(see https://pixi.prefix.dev/latest/ for details and other installation options)



### Install packages using pixi

#### Main packages
Run the following command in the root of the repository

```
pixi install --environment quapopt
```

The above installs all the basic packages required to run the project.

See below for additional packages that can be installed to extend the functionality of the repository.


#### [Recommended] Additional packages 

To make the best use of the repository, we recommend installing some additional packages, using larger environment instead:

```
pixi install --environment quapopt-full
```

Without the above, some functionalities might not be available.


#### [Optional] Additional packages for GPU support (Linux and Windows)
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




#### Make the repository importable
After setting up the environment, run the following command in the root of the repository, once for each
environment you want to use (for example `quapopt-full`)

```
pixi run add_repo_to_python_path
```

to put the repository on the Python path of that environment, so that notebooks and scripts can import `quapopt`.
The path is stored as an absolute path: run it again after you reinstall the environment or move the repository.


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


#### [Tip] Clean setup

If you run into troubles with pixi, it might be useful to remove pixi.lock file and try again.
```
rm ./pixi.lock
```
together with
```
pixi clean && pixi clean cache
```


### [Not recommended] Install packages using pip 

It should be also possible to install the repository using pip, but it is not recommended, as it might lead to some issues with dependencies, especially when using "full" and "gpu" environments.

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


## Tutorials

Please refer to the [tutorials](notebooks/tutorials) folder for examples of how to use the repository.

The basics are showcased in the [quick start](notebooks/tutorials/00_quick_start):
1. [Hamiltonian generation](notebooks/tutorials/00_quick_start/A_generating_hamiltonians) How to generate and save Hamiltonian instances for further use.
2. [Running basic QAOA](notebooks/tutorials/00_quick_start/B_running_qaoa_circuits) How to run basic QAOA optimization using different backends.
3. [Running QAOA with gauge transformations](notebooks/tutorials/00_quick_start/C_gauge_symmetries_and_NDAR/) How to run QAOA with gauge transformations and Noise-Directed Adaptive Remapping (NDAR).
4. [ND-AWS as an NDAR solver](notebooks/tutorials/00_quick_start/D_NDAWS_as_NDAR_solver/) How to run Noise-Directed Adaptive Warm-Starting (ND-AWS): the NDAR loop with a warm-started QAOA circuit as its sampler.

The [tutorials README](notebooks/tutorials/README.md) describes each notebook.

We plan to add more tutorials in the future.

### Paper notebooks

Since our repository is evolving, it's possible that in the future, the main branch of the repo will become incompatible with the code used in a particular paper. To get the compatible historical version, please use the relevant branch stated below.

1. (2026-09) Noise-Directed Adaptive Warm-Starting [[7]](https://arxiv.org/abs/2607.09368). The [ND-AWS paper notebooks](notebooks/papers/ndaws) run the experiments of the Noise-Directed Adaptive Warm-Starting (ND-AWS) paper and redraw its figures and table. See the README in that folder. Compatible-branch: *2026-09-paper-ndaws*.



## Citing this repo
The following bibtex entry can be used to cite this repository:

@misc{quapopt_repo,
  author = {Maciejewski, F. B. and Bach, B. G. and Biamonte, J. and Hadfield, S.A. and Venturelli, D.},
  title = {quapopt -- open source {G}it{H}ub repository for quantum approximate optimization},
 doi={https://github.com/usra-riacs/quantum-approximate-optimization},
howpublished={\url{https://github.com/usra-riacs/quantum-approximate-optimization}},
  year = {2026}
}


## Funding 
Development of significant parts of the repository was supported under the NSF awards #2329097 and #1918549.

## Used repositories 
We use some (refactored) code from the following repository:
* https://github.com/aboev/pymqlib (various classical solvers, including Burer-Monteiro algorithm) under The MIT License (MIT)

Its license can be found at the above link.


## References
This repository is based, among others, on the following papers, which describe some of the algorithms and methods used in the code:

[1] Maciejewski, Filip B., Jacob Biamonte, Stuart Hadfield, and Davide Venturelli. "[Improving quantum approximate optimization by noise-directed adaptive remapping.](https://arxiv.org/abs/2404.01412)" arXiv preprint arXiv:2404.01412 (2024).

[2] Maciejewski, Filip B., Bao G. Bach, Maxime Dupont, P. Aaron Lott, Bhuvanesh Sundar, David E. Bernal Neira, Ilya Safro, and Davide Venturelli. "[A multilevel approach for solving large-scale qubo problems with noisy hybrid quantum approximate optimization.](https://arxiv.org/abs/2408.07793)" In 2024 IEEE High Performance Extreme Computing Conference (HPEC), pp. 1-10. IEEE, 2024.

[3] Maciejewski, Filip B., Stuart Hadfield, Benjamin Hall, Mark Hodson, Maxime Dupont, Bram Evert, James Sud et al. "[Design and execution of quantum circuits using tens of superconducting qubits and thousands of gates for dense Ising optimization problems.](https://arxiv.org/abs/2308.12423)" Physical Review Applied 22, no. 4 (2024): 044074.

[4] Bach, Bao G., Filip B. Maciejewski, and Ilya Safro. "[Solving Large-Scale QUBO with Transferred Parameters from Multilevel QAOA of low depth.](https://arxiv.org/abs/2505.11464)"  in 2025 IEEE International Conference on Quantum Computing and Engineering (QCE) (Vol. 1, pp. 2120-2126). IEEE.

[5] Tam, Wai-Hong, Hiromichi Matsuyama, Ryo Sakai, and Yu Yamashiro. "[Enhancing NDAR with Delay-Gate-Induced Amplitude Damping.](https://arxiv.org/abs/2504.12628)" arXiv preprint arXiv:2504.12628 (2025).

[6] Dupont, Maxime, and Bhuvanesh Sundar. "[Extending relax-and-round combinatorial optimization solvers with quantum correlations.](https://arxiv.org/abs/2307.05821)" Physical Review A 109, no. 1 (2024): 012429.

[7] Filip B. Maciejewski, Stuart Hadfield, Oscar Wallis, George Pennington, Sebastian Brandhofer, Stefan Woerner, Daniel J. Egger, and Davide Venturelli.
 "[Quantum Approximate Optimization via Noise-Directed Adaptive Warm-Starting](https://arxiv.org/abs/2607.09368)" arXiv preprint arXiv:2607.09368 (2026).




