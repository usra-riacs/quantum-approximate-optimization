# ND-AWS paper notebooks

These five notebooks run the experiments reported in the ND-AWS paper (Noise-Directed
Adaptive Warm-Starting) and redraw its figures and table from the data those runs
produce. Use them to repeat our experiments, on a QPU or on a simulator.

They are intentionally narrow. Each notebook is fixed to the configuration the paper used,
and only the configuration cell near the top is meant to be edited: it selects which arm
of the study to run, not how the method works.

## If you want to build your own experiments

Start from the tutorials, not from here:

- `notebooks/tutorials/00_quick_start/A_generating_hamiltonians/` — build and store
  problem instances (Erdos-Renyi, regular, Sherrington-Kirkpatrick).
- `notebooks/tutorials/00_quick_start/B_running_qaoa_circuits/` — construct QAOA
  circuits, compile them for hardware, run them, and optimize the angles.
- `notebooks/tutorials/00_quick_start/C_gauge_symmetries_and_NDAR/` — gauge
  transformations and the noise-directed adaptive remapping loop these notebooks sit on.

The tutorials show the same components with every knob exposed and explained, on small
problems that finish quickly. They are the right starting point for a different
Hamiltonian class, a different ansatz, another sampler or another optimizer.

## The notebooks

| Notebook | What it does |
|---|---|
| `00a_ndaws_main_run.ipynb` | The 100-qubit study. One arm per run: three problem classes (10% and 20% Erdos-Renyi, 3-regular) on either the QPU or a noiseless matrix-product-state simulator, with the noise-directed or the standard warm-started ansatz. Writes one dataset per arm. |
| `00b_ndaws_main_analysis.ipynb` | Reads the datasets of the 100-qubit study and draws its convergence figures, then computes the iterations-to-convergence and approximation-ratio table beside the published values. |
| `01a_ndaws_amplitude_damping_run.ipynb` | The 20-qubit simulation study under amplitude damping. Sweeps the damping strength over both ansatz arms on a local simulator. Writes one dataset per damping value and arm. |
| `01b_ndaws_amplitude_damping_analysis.ipynb` | Reads the damping sweep and draws one panel per problem class: both ansatz arms at every damping strength, against the noiseless simulation. |
| `02a_ndaws_misc_staircase_plot.ipynb` | Draws the iteration-by-iteration staircase of a single run, with the problem graph and, optionally, the device connectivity. |

The two analysis notebooks share `helpers/ndaws_paper_analysis.py`: dataset reading, the
experiment sets excluded from the paper's analysis, and the common panel styling. Each
notebook keeps what is its own — which datasets make up a panel, and how that panel is
drawn — so run them from this folder, where the helper module is importable.

## Order of use

A run notebook comes before the analysis notebook that shares its number: `00a` before
`00b`, `01a` before `01b`. If you already hold the datasets, the analysis notebooks and
`02a` run on their own — point their configuration cells at the dataset names you have.

## What you need

The simulator arms run with no account of any kind: the 20-qubit study is entirely local,
and the matrix-product-state arms of the 100-qubit study are as well.

The QPU arms need IBM Quantum credentials in the environment or in a `.env` file. The
notebooks check for `IBM_TOKEN`, `IBM_CREDENTIALS_PATH` and `IBM_ACCOUNT_NAME`, and any one
of the three passes the check; which of them your account needs is a Qiskit matter, not a
notebook one. A notebook that needs credentials and cannot find them stops with a message
saying so.

## Where results go

The run notebooks write into the standardized result store
(`DEFAULT_STORAGE_DIRECTORY`). 
If you don't set this env variable (see README.md in the repo's root), it defaults to
`output/` at the repository root, no matter which folder you started the notebook from.

Every dataset name ends with a `RUN_ID` you set in the configuration cell. Before writing, each run notebook checks
whether that dataset name already holds experiment sets and stops if it does, so a second
run cannot silently mix new results into an earlier study. Choose a fresh `RUN_ID` for
each study; a paired comparison uses one `RUN_ID` across both arms.

Figures are written under `./temp/` inside this folder.

## A note on nomenclature

Dataset names carry the prefix `NDAWS_`.
The `no-gauges-version` in a name marks the standard warm-started arm without gauge transformation; a name
without it is the noise-directed arm.

Across the repo, we generally treat NDAWS as a special case of NDAR (Noise-Directed Adaptive
Remapping).

## References

Maciejewski, Filip B., Jacob Biamonte, Stuart Hadfield, and Davide Venturelli. "[Improving quantum approximate optimization by noise-directed adaptive remapping.](https://arxiv.org/abs/2404.01412)" arXiv preprint arXiv:2404.01412 (2024).

Filip B Maciejewski, Stuart Hadfield, Oscar Wallis, George Pennington, Sebastian Brandhofer, Stefan Woerner, Daniel J Egger, Davide Venturelli "[Quantum Approximate Optimization via Noise-Directed Adaptive Warm-Starting](https://arxiv.org/abs/2607.09368)" arXiv:2607.09368 (2026).


## Citing the paper
The following bibtex entry can be used to cite the NDAWS paper:

```bibtex
@misc{maciejewski2026ndaws,
      title={Quantum Approximate Optimization via Noise-Directed Adaptive Warm-Starting}, 
      author={Filip B. Maciejewski and Stuart Hadfield and Oscar Wallis and George Pennington and Sebastian Brandhofer and Stefan Woerner and Daniel J. Egger and Davide Venturelli},
      year={2026},
      eprint={2607.09368},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2607.09368}, 
}
```


