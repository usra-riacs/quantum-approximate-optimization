"""Shared machinery for the ND-AWS paper analysis notebooks.

Both analysis notebooks read datasets the same way and draw panels on the same axes, and
both must apply the same exclusions. The exclusion lists in particular are kept here rather
than copied into each notebook: two copies drift, and a dataset dropped from one figure but
not the other is the kind of difference nobody sees.

What stays in the notebooks is what differs between the figures: which datasets make up each
panel, and how a panel is drawn.
"""
import contextlib
import io

import numpy as np

from quapopt.ancillary_functions.presets import get_standard_subfolders_hierarchy_full
from quapopt.data_analysis.data_handling import (ResultsLogger,
                                                 STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 STANDARD_NAMES_VARIABLES as SNV)
from quapopt.data_analysis.data_handling.io_utilities.dataset_management import read_dataset_info
from quapopt.data_analysis.experiments.NDAWSQAOAResultsAnalyzer import NDAWSQAOAResultsAnalyzer
from quapopt.data_analysis.visualization.PlotlySubplotsPlotter import PlotlySubplotsPlotter

FIGURE_DIR = './temp/figures'

FOM_NAME = f'{SNV.ApproximationRatioBest.id_long}'
BEST_OUT_OF = 3
PRINT_DATA = True

# Experiment-set ids excluded at read time, as in the paper's analysis: runs that did not
# complete.
faulty_ids = [
    "083470ca33b740259754e280a263255b",
    "3a4b1814134d47228bea802a094cb944",
    "af3e3ea9ce7340e4bfa9f1eadc922db3",
    "bd3e1ff481f243efb361f2eef9a64288",
    "e111b15aa3f84e4c80a5816fa8f1f496",
    "c702759e8ea941e9b9549d2a4fb6f725",
    "0554b53d2e024dbcb5ef9d9107b2651c",
    "fad604d2b637487383d3dfdc50a6e1aa",
    "0aceb892dbc54c16af34732a73288737",
    "6f0ceeb91b0d416095ffefa2f3bd478f",
    "db22d2cef52e49f78cb747105cb19467",
    "74162ac44cf9450db9b11a42da578006",
    "d7fa391126db4e1e9eb9dc2d363da9f0",
    "0838b4d303084c558d258273b414cd67",
    "2d7a419db1af4c01aedd36ed836ce56a",
    "5179295923414646851c309f58125299",
    "f6291d8221e6402685aef1310b48e9f6",
    "ddd1b376bc584657a91e80eb7a288914",
    "61ba30ab7e4840c8afb2ed3839748e9f",
    "c9bd1af0ce0243bb9b8799761c864479",
    "0a2e672a01c84e53951e162f4619adbc",
    "0f05221196d64d718d53b4e9c12af354",
    "9d3b3d26fcb54cd6a5c20fecff188a6f",
    "9f96b9859865409883533377779aac23",
    "cfc14b14d41d45488ada71f93abbee94",
    "4e6f000eaafb446e9761860e8acc7b3f",
    "ee7c1da5e565431fbeb9896100c1079b",
    "63fe0916e5464c97bf8ffc58eba7ab15",
    "7b62f28aad314fdc855ee73860affcd6",
    "743be9f634454312ad6e91dbc5c6bdbb",
]

# One pass per damping strength is used; the set ids of the unused pass are listed below.
duplicate_pass_ids = [
    # q=0.01 — the unused pass
    "002061a8ee12412ea7df397476920151",
    "1526745509ed49cdb91f05a79ef92523",
    "16033ea39e604501acf92111742512a4",
    "18e2ee0071c74bdfb8153ae4661b3ecd",
    "1daf92f86aa342ad90be50778e15bfb6",
    "25cc9426b65a43139d5f0383086f461f",
    "2c5eedea8cbb44a8a06e29637dbb75a5",
    "32c015064f98434b805901f5096cdd18",
    "40dc7b3a77e845b98776112d1a20084e",
    "42de33d39a4d43b09d3cd766373955e5",
    "47936a87b86d4d94a24c0b645f21898b",
    "47ee7e18771546128f4b4b1d32078d91",
    "53dac3fa50ce43248847e36bc7498e0d",
    "58481b8999fb49f0afa000e1a7307a94",
    "62f628b895c74daebb7c877361346420",
    "8e5429bc026b41d79f5e43c67f44d85d",
    "8f9c553d47cd4b9eb5b038cf183d0018",
    "93c02ef166d44bd9a66e0cb6de7461c3",
    "9fed78af39674ca69390ebf3a8c52134",
    "a314c816e156425783d3ba9b04d9e837",
    "abc049629d9040de8b011ccb116f1b51",
    "ac426dd91c6b429e946b4de41075b4b3",
    "b9d602e778c94e6697f111578dde46c6",
    "cabfea271e7043beae5cccf670a46927",
    "cf2a1679a26940e98437d905e80db6a9",
    "d507a2f4fa9740509e1e405c575c1852",
    "e3fc6b6164f3423b86fd2980fceae3f4",
    "e52fab2310d1476f8954e577705354ec",
    "f7bae3a76c4f406ba54c5151f2f1ee7c",
    "fbaa740a58034f4ebf1d343e0e10d893",
    # q=0.02 — the unused pass, the one that was interrupted
    "0838b4d303084c558d258273b414cd67",
    "1aecc87d295c45949ed49444d76e123c",
    "21342312da3249d7941695c539397171",
    "3fc9f9627795418d8f80a3c02c05dcb9",
    "467a3e6e4d844c1ba1131c7f225b5ad9",
    "53d09334c9ed47eea9b85d9828d86503",
    "561542fef9e64214b70c6224ddd68e89",
    "568965d01c494764a68b318d7ad8bd86",
    "5a7edc8c5f874113b962490f68f6203c",
    "5da763fccf1c4fd2b9330286ea5b3e0a",
    "5e96994f5e3e4cfca2128586bb28148d",
    "8951821b414f47b380f69b5345ac00e3",
    "b1c755c39c184d8aac87fb10e09607b8",
    "be85430103a2462680ebf2690a0c301c",
    "c18bd047950f4034b879854ac6ac0643",
    "ccf4be6c365446f09444fd384d294822",
]

excluded_ids = set(faulty_ids) | set(duplicate_pass_ids)

HAM_CLASS_DESCRIPTIONS = {
    'RG': "HMN=RG;LOC=(2,);CFD=CT~CON_CDN~NOR_CDP~loc~0.0_scale~1.0",
    'ER': "HMN=ER;LOC=(2,);CFD=CT~CON_CDN~NOR_CDP~loc~0.0_scale~1.0;ERT=GNP",
}


def ham_class_description_of(dataset_name):
    # Match the class token the name-building rule writes -- `_RG-3_`, `_ER-0.1_` -- and not a
    # bare substring: a RUN_ID with "rg" or "er" anywhere in it would otherwise pick the class,
    # and the dataset would be read from a folder that holds nothing.
    if '_RG-' in dataset_name:
        return HAM_CLASS_DESCRIPTIONS['RG']
    elif '_ER-' in dataset_name:
        return HAM_CLASS_DESCRIPTIONS['ER']
    raise NotImplementedError(f"cannot infer the Hamiltonian class from dataset name {dataset_name!r}; "
                              f"it carries neither an '_RG-' nor an '_ER-' token")


def sets_per_instance(experiment_set_ids, experiments_folders_hierarchy):
    counts = {}
    for experiment_set_id in experiment_set_ids:
        reader = ResultsLogger(experiment_folders_hierarchy=experiments_folders_hierarchy,
                               experiment_set_id=experiment_set_id,
                               experiment_instance_id='nan')
        metadata = reader.read_metadata(data_type=SNDT.CircuitsMetadata,
                                        shared_across_experiment_set=True,
                                        return_none_if_not_found=True)
        if metadata is None:
            key = '<metadata not found>'
        else:
            key = metadata['HamiltonianMetadata']['HamiltonianInstanceDescription']
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_dataset(dataset_name, backend_name, simulation, include_only_best_run,
                 best_out_of=BEST_OUT_OF, print_data=PRINT_DATA):
    """Read one dataset and return its per-instance convergence curves.

    `include_only_best_run` has no default on purpose: the two figures aggregate the three
    seeded runs per instance differently, and a shared default silently gives one of them
    the other's convention. The 100-qubit study reports the best of the three runs for each
    instance (True). The amplitude-damping sweep averages over instances and over all three
    runs (False).
    """
    noiseless_simulation = ('noiseless' in dataset_name) if simulation else False
    ham_class_description = ham_class_description_of(dataset_name)

    experiments_folders_hierarchy = get_standard_subfolders_hierarchy_full(
        experiment_category='NDAR',
        experiment_subcategory='WS-QAOA-p1',
        backend_name=backend_name,
        simulation=simulation,
        noiseless_simulation=noiseless_simulation,
        hamiltonian_class_description=ham_class_description)

    experiment_config = {'backend_name': backend_name,
                         'hamiltonian_class_description': ham_class_description,
                         'merge_instances_metadata_data_type': None,
                         'main_folders_hierarchy': experiments_folders_hierarchy}

    all_ids = read_dataset_info(dataset_name=dataset_name)
    experiment_set_ids = [_id for _id in all_ids if _id not in excluded_ids]

    print(f"\nDataset: {dataset_name}")
    if len(experiment_set_ids) == 0:
        raise ValueError(f"no experiment sets found for dataset {dataset_name}; "
                         f"check the name, the backend {backend_name!r} and the simulation flag")

    # The per-instance counts are returned rather than printed: a panel reads a dozen
    # datasets of ten instances each, and listing them fills the output with hundreds of
    # lines nobody reads. Only an uneven count is worth interrupting for.
    counts = sets_per_instance(experiment_set_ids, experiments_folders_hierarchy)
    if len(set(counts.values())) > 1:
        print(f"  WARNING: the {len(counts)} Hamiltonian instances do not all hold the same "
              f"number of runs ({sorted(set(counts.values()))}), so they enter the average "
              f"with different weights")
    # An even count warns too, and for a different reason: `best_out_of` runs per instance is
    # what the figures aggregate, so extra runs make the reported best a best-of-more. Six on
    # every instance passes the check above, because every instance holds the same six.
    if counts and max(counts.values()) > best_out_of:
        print(f"  WARNING: some Hamiltonian instances hold up to {max(counts.values())} runs "
              f"while the aggregation takes the best of {best_out_of}; the reported values "
              f"come from a larger pool than the published ones")

    results_analyzer = NDAWSQAOAResultsAnalyzer(experiment_config=experiment_config)
    results_analyzer.initialize_results_from_config(
        data_types_of_interest=[SNDT.NDAROverview],
        experiment_set_ids_experiments=[] if simulation else experiment_set_ids,
        experiment_set_ids_simulations=experiment_set_ids if simulation else [])

    number_of_qubits = list(list(results_analyzer.cost_hamiltonians.values())[0].values())[0].number_of_qubits

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        figure, (all_xs_converged, all_ys), _baseline = results_analyzer.create_ndar_overview_plot(
            figure_of_merit_name=FOM_NAME,
            figure_title=None,
            include_only_best_run=include_only_best_run,
            simulation=simulation,
            show_legend=True,
            add_mean=True,
            print_data=print_data,
            return_data=True,
            hamiltonian_instances=None,
            best_out_of=best_out_of)

    return {'dataset_name': dataset_name,
            'all_xs_converged': all_xs_converged,
            'all_ys': all_ys,
            'number_of_qubits': number_of_qubits,
            'sets_per_instance': counts,
            'analyzer_output': captured.getvalue()}


LINE_WIDTH_BORDER = 10
LINE_WIDTH_INSTANCES = 0.5
MARKER_SIZE_INSTANCES = 10
MARKER_SIZE_CONVERGED = 2 * MARKER_SIZE_INSTANCES
FONTSIZE = int(32 * 2)
AR_FLOOR = 0.001


def pad_to_common_length(all_ys):
    max_length = max([len(ys) for ys in all_ys])
    all_ys_extended = []
    for ys in all_ys:
        ys2 = ys.copy()
        if len(ys2) < max_length:
            #append last value to the end of the list so the length is of max_length
            ys2 += [np.max(ys2)] * (max_length - len(ys2))
        all_ys_extended.append(ys2)
    return np.array(all_ys_extended)


def standard_axes_kwargs(tickwidth, ticks):
    return dict(showgrid=True,
                zeroline=False,
                showline=True,
                mirror=True,
                tickwidth=tickwidth,
                tickfont=dict(size=FONTSIZE, family='Serif'),
                ticks=ticks,
                tickson="boundaries",
                ticklen=10,
                linecolor='black')


def finish_panel(merged, panel_key, subfig_letter, yrange, xticks, xrange, height, width):
    overwrite_kwargs_layout = {'margin': dict(t=0, b=0, l=10, r=0),
                               'legend': None,
                               'paper_bgcolor': 'rgba(255, 255, 255, 1.0)',
                               'plot_bgcolor': 'rgba(255, 255, 255, 1.0)',
                               'font': dict(size=FONTSIZE, family='Serif'),
                               'xaxis': dict(gridcolor='rgba(128, 128, 128, 0.2)'),
                               'yaxis': dict(gridcolor='rgba(128, 128, 128, 0.2)')}

    merged.add_shape(type='line', x0=0, y0=AR_FLOOR, x1=50, y1=AR_FLOOR,
                     line=dict(color='black', width=2.5, dash='dot'))

    merged.update_yaxes(tickvals=None, range=yrange, type='log')
    merged.update_xaxes(tickvals=xticks, ticktext=xticks, range=xrange)

    merged = PlotlySubplotsPlotter.update_figure_with_standard_specs(
        figure=merged,
        overwrite_kwargs_layout=overwrite_kwargs_layout,
        overwrite_kwargs_axes=standard_axes_kwargs(tickwidth=2, ticks="inside"))

    merged.update_layout(title_text=None)
    merged.update_yaxes(title_standoff=45,
                        tickfont=dict(size=FONTSIZE, family='Serif'),
                        tickvals=[0.001, 0.01, 0.1] + [0.2, 0.3] + [0.01 * i for i in range(1, 10)]
                                 + [0.001 * i for i in range(1, 10)],
                        ticktext=[r"$\Huge{10^{-3}}$", r"$\Huge{10^{-2}}$", r"$\Huge{10^{-1}}$"] + [""] * 20)
    merged.update_xaxes(title_standoff=30, tickfont=dict(size=FONTSIZE, family='Serif'))

    merged.add_annotation(text=f'<b>{subfig_letter}) {panel_key}</b>',
                          x=0.01, y=0.99,
                          xref='paper', yref='paper',
                          showarrow=False,
                          font=dict(size=FONTSIZE, family='Serif'),
                          bgcolor="rgba(255,255,255, 0.85)")

    merged.update_layout(showlegend=False, height=height, width=width)
    merged.update_xaxes(showline=True, linewidth=7.5, linecolor='black', mirror=True, gridwidth=2.5)
    merged.update_yaxes(showline=True, linewidth=7.5, linecolor='black', mirror=True, gridwidth=2.5,
                        range=yrange)
    axes = standard_axes_kwargs(tickwidth=5, ticks="outside")
    merged.update_layout(xaxis=axes, yaxis=axes)
    return merged
