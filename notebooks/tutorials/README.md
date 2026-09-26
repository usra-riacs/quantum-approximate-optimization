# Tutorials

The quick-start tutorials, in the order they are meant to be read. Each one runs on the local
simulators that ship with the package; none of them needs a quantum-hardware account.

Results go to the standardized result store, at `DEFAULT_STORAGE_DIRECTORY` when that
environment variable is set and at `output/` in the repository root when it is not. Figures go
to `notebooks/tutorials/00_quick_start/temp/`.

## 00_quick_start

### A -- Hamiltonians

- [A01 -- Generating and storing random Hamiltonian instances](00_quick_start/A_generating_hamiltonians/A01_generate_and_save_Hamiltonians.ipynb)
  Builds random classical Ising instances both ways: the fine-grained
  `build_hamiltonian_generator` path, and the `generate_random_hamiltonian` shortcut every
  later notebook uses. Solves them classically and writes the instances and their known
  solutions to the store.

### B -- QAOA circuits and angle optimization

- [B01a -- Building and running a QAOA circuit with Qiskit](00_quick_start/B_running_qaoa_circuits/B01a_run_qaoa_circuit_qiskit.ipynb)
  The p=1 circuit in three routings -- linear swap network, all-to-all, and SABRE -- run on a
  Qiskit simulator, noiselessly and then under amplitude damping. Ends with the two wrappers
  that hide the Qiskit plumbing.
- [B02a -- Optimizing QAOA angles](00_quick_start/B_running_qaoa_circuits/B02a_run_qaoa_optimization.ipynb)
  A full p=1 angle optimization, noiseless and then with measurement noise, read back from the
  result store and drawn as two trajectories over a brute-force angle grid.
- [B02b -- The same optimization on two sampler backends: python and qiskit](00_quick_start/B_running_qaoa_circuits/B02b_qaoa_optimization_compare_backends.ipynb)
  Runs B02a's optimization once on each of the two sampler backends `python` and
  `qiskit`, and prints one table: setup time, the trials each one actually ran and the time per
  trial, the best energy, the angles it converged to, and whether the backends agree within a
  conservative shot-noise tolerance.
- [B03 -- Single-layer QAOA from exact expectation values](00_quick_start/B_running_qaoa_circuits/B03_run_single_layer_qaoa_expected_values.ipynb)
  p=1 QAOA without sampling, at a hundred qubits and more: two expectation-value simulators
  compared, and a warm-started time-block ansatz given the same angle-search budget two ways.

### C -- Gauge symmetries and NDAR

- [C01 -- Gauge symmetries of a QAOA landscape](00_quick_start/C_gauge_symmetries_and_NDAR/C01_run_qaoa_optimization_with_gauges.ipynb)
  A bitflip gauge leaves the spectrum alone but not a noisy optimization. Compares three
  gauges, then sweeps thirty random ones to relate optimization quality to the energy of
  |00...0>.
- [C02 -- Noise-Directed Adaptive Remapping](00_quick_start/C_gauge_symmetries_and_NDAR/C02_run_qaoa_optimization_with_NDAR.ipynb)
  NDAR turns biased readout noise into an advantage by re-gauging after each round of
  optimization. Runs the loop with a p=1 QAOA sampler as its inner optimizer and compares the
  result with a noiseless run of the same budget.

### D -- ND-AWS: warm-started QAOA inside the NDAR loop

- [D01 -- How far may a warm start sit from the solution?](00_quick_start/D_NDAWS_as_NDAR_solver/D01_warm_start_distance_to_solution.ipynb)
  The counterpart of C01 for warm starts, and with no noise in it: sweeps the Hamming distance
  between the warm start and the ground state, and plots what the single-layer optimization
  reaches at each one. This is the curve ND-AWS climbs.
- [D02 -- ND-AWS on a Qiskit backend](00_quick_start/D_NDAWS_as_NDAR_solver/D02_run_qaoa_optimization_with_NDAWS_qiskit.ipynb)
  Noise-Directed Adaptive Warm-Starting: the NDAR loop with a WARM-STARTED p=1 QAOA circuit as
  its local sampler, angles set offline on a reduced-density-matrix simulator, circuits run
  through Qiskit on the local `aer` simulator. Compares the result with plain noiseless QAOA of
  the same budget, and reads the run back to show where the circuit samples.
