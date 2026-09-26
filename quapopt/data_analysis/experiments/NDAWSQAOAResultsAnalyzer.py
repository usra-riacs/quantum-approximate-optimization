# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from typing import List, Optional, Type, Callable, Dict, Any

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from tqdm.notebook import tqdm
from quapopt import ancillary_functions as anf

from quapopt.data_analysis.data_handling.schemas.naming import (STANDARD_NAMES_VARIABLES as SNV,
                                                                STANDARD_NAMES_DATA_TYPES as SNDT,
                                                                STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS as SNHD)
from quapopt.data_analysis.experiments.QAOAResultsAnalyzer import QAOAResultsAnalyzer
from quapopt.data_analysis.visualization import optimization_visualization as opt_vis
from quapopt.data_analysis.visualization.PlotlySubplotsPlotter import PlotlySubplotsPlotter
from plotly.colors import qualitative, sequential
from quapopt.meta_algorithms.NDAR.NDARRunner import NDARRunner

from quapopt.data_analysis.statistics.SamplingResultsAnalyzer import SamplingResultsAnalyzer

from quapopt.data_analysis.data_handling import (LoggingLevel,
                                                 ResultsLogger,
                                                 STANDARD_NAMES_DATA_TYPES as SNDT
                                                 )
from quapopt.hamiltonians.generators import create_hamiltonian_from_descriptions
from quapopt.ancillary_functions.presets.ndar import build_universal_logging_callable
from quapopt.meta_algorithms.NDAR import ConvergenceCriterion
from quapopt.meta_algorithms.NDAR import ConvergenceCriterionNames, ConvergenceCriterion, NDARIterationResult


class NDAWSQAOAResultsAnalyzer(QAOAResultsAnalyzer):
    def __init__(self,
                 experiment_config: Dict[str, Any],
                 df_processing_functions: List[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 subset_variables_values_dict: Optional[Dict[str, Any | List[Any]]] = None, ):

        super().__init__(experiment_config=experiment_config,
                         df_processing_functions=df_processing_functions,
                         subset_variables_values_dict=subset_variables_values_dict,
                         )

        self._ndar_overviews = None


    def set_data_type(self,
                      data_type: Type[SNDT],
                      results_df: pd.DataFrame,
                      ):

        if data_type == SNDT.NDAROverview:
            results_df = results_df.copy()

            def _update_df(row):
                experiment_set_id = row[SNV.ExperimentSetID.id_long]
                ham_class_description, ham_instance_description = self._id_to_hamiltonian_mapping[experiment_set_id]
                hamiltonian_instance = self._cost_hamiltonians[ham_class_description][ham_instance_description]
                energy_best = row[SNV.EnergyBest.id_long]
                energy_mean = row[SNV.EnergyMean.id_long]

                row[SNV.ApproximationRatioBest.id_long] = hamiltonian_instance.calculate_approximation_ratio(
                    energy_best)
                row[SNV.ApproximationRatioMean.id_long] = hamiltonian_instance.calculate_approximation_ratio(
                    energy_mean)

                row[SNV.HamiltonianClassDescription.id_long] = ham_class_description
                row[SNV.HamiltonianInstanceDescription.id_long] = ham_instance_description

                return row

            results_df = results_df.apply(_update_df, axis=1)
            nice_order = [
                f"Simulation",
                f"{SNV.HamiltonianClassDescription.id_long}",
                f"{SNV.HamiltonianInstanceDescription.id_long}",
                f"{SNV.NDARIteration.id_long}",
                f"{SNV.EnergyBest.id_long}",
                f"{SNV.ApproximationRatioBest.id_long}",
                f"{SNV.BitflipHammingWeight.id_long}",
                f"{SNV.WSBiasParameters.id_long}",
                f"{SNV.EnergyMean.id_long}",
                f"{SNV.ApproximationRatioMean.id_long}",
                f"{SNV.SamplerName.id_long}",
                f"{SNV.HamiltonianRepresentationIndex.id_long}",
                f"{SNV.TrialIndex.id_long}",
                f"{SNV.Angles.id_long}",
                # f"{SNV.AttractorModel.id_long}",
                #    f"{SNV.ConvergenceCriterion.id_long}",
                #   f"{SNV.BitstringBest.id_long}",
                f"{SNV.Bitflip.id_long}",
                f"{SNV.ExperimentSetID.id_long}"
            ]
            results_df.drop(columns=[f"{SNV.AttractorModel.id_long}",
                                     f"{SNV.ConvergenceCriterion.id_long}",
                                     f"{SNV.BitstringBest.id_long}",
                                     # f"{SNV.Bitflip.id_long}"
                                     ],
                            inplace=True,
                            errors='ignore')

            results_df = results_df.sort_values(by=[f"{SNV.HamiltonianClassDescription.id_long}",
                                                    f"{SNV.HamiltonianInstanceDescription.id_long}",
                                                    f"{SNV.ExperimentSetID.id_long}",
                                                    f"{SNV.NDARIteration.id_long}"

                                                    ])


            self._ndar_overviews = results_df[nice_order]
        else:
            return super().set_data_type(data_type=data_type, results_df=results_df)

    def get_data_type(self,
                      data_type: Type[SNDT],
                      ) -> Optional[pd.DataFrame]:

        if data_type == SNDT.NDAROverview:
            return self._ndar_overviews
        else:
            return super().get_data_type(data_type=data_type)


    def initialize_results_from_config(self,
                                       data_types_of_interest: List[Type[SNDT]],
                                       experiment_set_ids_experiments: Optional[List[str]] = None,
                                       experiment_set_ids_simulations: Optional[List[str]] = None,
                                       show_progress_bar: bool = False,
                                       noiseless_simulation: bool = True,
                                       ):
        self._add_hamiltonians_metadata(experiment_set_ids_experiments=experiment_set_ids_experiments,
                                        experiment_set_ids_simulations=experiment_set_ids_simulations)

        super()._initialize_results_from_config(
            data_types_of_interest=data_types_of_interest,
            experiment_set_ids_experiments=experiment_set_ids_experiments,
            experiment_set_ids_simulations=experiment_set_ids_simulations,
            show_progress_bar=show_progress_bar,
        )

    def _add_hamiltonians_metadata(self,
                                   data_type_for_inference: Type[SNDT] = SNDT.CircuitsMetadata,
                                   experiment_set_ids_experiments: Optional[List[str]] = None,
                                   experiment_set_ids_simulations: Optional[List[str]] = None,
                                   ):

        experiment_set_ids_experiments = [] if experiment_set_ids_experiments is None else experiment_set_ids_experiments
        experiment_set_ids_simulations = [] if experiment_set_ids_simulations is None else experiment_set_ids_simulations

        all_experiment_set_ids = experiment_set_ids_experiments + experiment_set_ids_simulations

        hamiltonians_metadata = {}
        id_to_hamiltonian_mapping = {}
        for experiment_set_id in all_experiment_set_ids:
            logger_kwargs_main = {'experiment_folders_hierarchy': self._experiment_config['main_folders_hierarchy'],
                                  'experiment_set_id': experiment_set_id,  # Used to group the experiments
                                  }
            results_reader = ResultsLogger(**logger_kwargs_main,
                                           experiment_instance_id='nan')
            metadata_experiment = results_reader.read_metadata(data_type=SNDT.CircuitsMetadata,
                                                               shared_across_experiment_set=True,
                                                               return_none_if_not_found=True)
            if metadata_experiment is None:
                continue

            instance_description = metadata_experiment['HamiltonianMetadata']['HamiltonianInstanceDescription']
            class_description = metadata_experiment['HamiltonianMetadata']['HamiltonianClassDescription']

            if class_description not in hamiltonians_metadata:
                hamiltonians_metadata[class_description] = {}


            hamiltonian_instance = create_hamiltonian_from_descriptions(class_description=class_description,
                                                                        instance_description=instance_description,
                                                                        default_backend='numpy')

            hamiltonians_metadata[class_description][instance_description] = hamiltonian_instance
            id_to_hamiltonian_mapping[experiment_set_id] = (class_description, instance_description)

        self.set_cost_hamiltonians(cost_hamiltonians=hamiltonians_metadata)
        self._id_to_hamiltonian_mapping = id_to_hamiltonian_mapping

    @staticmethod
    def save_figure(
            figure: Figure,
            figure_subfolder: Optional[str] = None,
            figure_filename: Optional[str] = None,
            extension='png',
            scale: int = 2
    ):

        PlotlySubplotsPlotter.save_figure_static(figure=figure,
                                                 figure_subfolder=figure_subfolder,
                                                 figure_filename=figure_filename,
                                                 extension=extension,
                                                 scale=scale)


    def _create_ndar_overview_plot_absolute_metrics(self,
                                  figure_of_merit_name: str,
                                  figure_title: Optional[str] = None,
                                  in_3d: bool = False,
                                  simulation: bool = False,
                                  markersize_scatter=5.0,
                                  colormap_name='Viridis',
                                  include_only_best_run: bool = True,

                                  #   add_simulation:bool = True,
                                  add_mean: bool = False,
                                  show_legend: bool = True,
                                  print_data: bool = True,
                                  hamiltonian_classes=None,
                                  hamiltonian_instances=None,
                                  return_data=False,
                                  # TODO(FBM); this should be inferred
                                  best_out_of: Optional[int] = None,
                                  ):


        if in_3d:
            raise NotImplementedError("in_3d=True not implemented yet")

        x_name = SNV.NDARIteration.id_long
        y_name = figure_of_merit_name

        df_res = self.get_data_type(data_type=SNDT.NDAROverview)

        unique_colors = sequential.solar

        df_filtered = df_res.copy()

        bounds_y, ticks_y, bounds_x, ticks_x = None, None, None, None

        if figure_of_merit_name == SNV.ApproximationRatioBest.id_long:
            bounds_y = (0.725, 1.025)
            ticks_y = np.linspace(0.5, 1.0, 11, endpoint=True)
        elif figure_of_merit_name == SNV.BitflipHammingWeight.id_long:
            bounds_y = -1, df_filtered[figure_of_merit_name].max() + 1
            ticks_y = None
        elif figure_of_merit_name == SNV.ApproximationRatioMean.id_long:
            bounds_y = (0.45, 1.025)
            ticks_y = np.linspace(0.5, 1.0, 11, endpoint=True)

        # bounds_x = -1, df_filtered[x_name].max() + 1
        bounds_x = -1, None

        plotter = PlotlySubplotsPlotter(results_dataframe=df_filtered,
                                        subplot_variable_names=None,
                                        n_cols=1,
                                        in_3d=in_3d)

        figure = plotter.create_subplots_figure(figure_title=figure_title)

        fom_name = figure_of_merit_name


        nicer_names = {
            SNV.ApproximationRatioMean.id_long: r"$\Huge{\left<AR\right>}$",
            SNV.EnergyMean.id_long: '<E>',
            SNV.ApproximationRatioBest.id_long: r"$\Huge{\mathrm{AR}}$",
            SNV.EnergyBest.id_long: 'E-best',
            x_name: r"$\Huge{\mathrm{Iteration}}$",
        }



        for specs, df_res_i in plotter.iter_subplots():
            row_index, col_index = plotter.get_plotly_row_col_from_spec(spec=specs)
            if df_res_i is None:
                continue
            if df_res_i.empty:
                continue

            df_res_i_experiment = df_res_i[df_res_i['Simulation'] == simulation]
            df_res_i_experiment_grouped = df_res_i_experiment.groupby(by=['HamiltonianClassDescription',
                                                                          'HamiltonianInstanceDescription'])

            _len_instance_ids = len(df_res_i_experiment['HamiltonianInstanceDescription'].unique())

            if print_data and _len_instance_ids!=10:
                anf.cool_print("WATCHOUT, number of unique Hamiltonian instances:", _len_instance_ids, 'red')



            # max_iteration = df_res_i_experiment_grouped[x_name].max().max()

            all_ys = []
            all_xs_converged = []


            for index_group, ((ham_class_d, ham_inst_d), grouped_df_hamiltonian) in enumerate(
                    df_res_i_experiment_grouped):
                if hamiltonian_classes is not None and ham_class_d not in hamiltonian_classes:
                    continue
                if hamiltonian_instances is not None and ham_inst_d not in hamiltonian_instances:
                    continue

                # print('hejka', len(grouped_df_hamiltonian[SNV.ExperimentSetID.id_long].unique()))
                if include_only_best_run:
                    _sets_here = len(grouped_df_hamiltonian[SNV.ExperimentSetID.id_long].unique())

                    if print_data and _sets_here!=3:
                        anf.cool_print("WATCHOUT, unique experiment set ids:", _sets_here ,'red')
                        print("________")
                        #let's print hamiltonian instance and class names
                        print(ham_inst_d)
                        print("________")

                    group_df_best = anf.contract_dataframe_with_minmax_values(df=grouped_df_hamiltonian,
                                                                              variable_name=SNV.ApproximationRatioBest.id_long,
                                                                              find_maximal_value=True)

                    group_best_id = group_df_best[SNV.ExperimentSetID.id_long].values[0]
                    group_df_plot = grouped_df_hamiltonian[
                        grouped_df_hamiltonian[SNV.ExperimentSetID.id_long] == group_best_id].copy()
                else:
                    group_df_plot = grouped_df_hamiltonian.copy()


                color_hamiltonian = unique_colors[index_group]
                marker_dict = dict(color=color_hamiltonian, size=10, symbol='circle')
                line_dict = dict(color=color_hamiltonian, dash='solid', width=5)
                group_df_plot_by_id = group_df_plot.groupby(by=[SNV.ExperimentSetID.id_long])

                for index_group_by_id, ((experiment_set_id,), group_df_plot_by_id_i) in enumerate(group_df_plot_by_id):
                    group_plot_run = group_df_plot_by_id_i.sort_values(by=SNV.NDARIteration.id_long,
                                                                       ascending=True).copy()

                    figure.add_scatter(x=group_plot_run[x_name],
                                       y=group_plot_run[y_name],
                                       mode='lines+markers',
                                       marker=marker_dict,
                                       line=line_dict,
                                       name=ham_inst_d,
                                       legendgroup=ham_inst_d,
                                       showlegend=index_group_by_id == 0 and show_legend,
                                       row=row_index,
                                       col=col_index,
                                       # hoverinfo='skip'
                                       )

                    ys_i = group_plot_run[y_name].tolist()


                    xs_i = group_plot_run[x_name].tolist()
                    xs_i_max = max(xs_i)
                    #in the case we broke the optimization through reaching ground state, we add 3 iterations for fair comparison
                    if ys_i[-1]==1.0 and ys_i[-2]!=1.0:
                        xs_i_max+=3
                        ys_i+=[ys_i[-1]]*3


                    all_ys.append(ys_i)
                    all_xs_converged.append(xs_i_max)

            max_iteration = np.max(all_xs_converged)

            all_ys_extended = []
            for ys in all_ys:
                ys2 = ys.copy()
                if len(ys2)<max_iteration+1:
                    ys2 = ys2+[np.max(ys2)]*(max_iteration+1-len(ys2))
                all_ys_extended.append(ys2)

            all_ys_extended = np.array(all_ys_extended)


            # print(all_ys_extended[0:10,0:10])


            if print_data:
                #    print(sorted(all_xs_converged), len(all_xs_converged))

                print('dataset shape:', )

                print('min/max iteration converged:', np.min(all_xs_converged), np.max(all_xs_converged))
                print('median/mean iteration converged:', np.median(all_xs_converged), np.mean(all_xs_converged))
                print('STD iteration converged:', np.std(all_xs_converged))

                print("min/max AR converged:", np.min(all_ys_extended[:, -1]), np.max(all_ys_extended[:, -1]))
                print('median/mean AR converged:', np.median(all_ys_extended[:,-1], axis=0),
                      np.mean(all_ys_extended[:,-1], axis=0))
                print('STD AR converged:', np.std(all_ys_extended[:,-1], axis=0))





                print("___")
                print("ROW:" f"[ {np.min(all_xs_converged)}, {np.max(all_xs_converged)}, {np.median(all_xs_converged)}, "
                      f"{np.mean(all_xs_converged)} (pm {np.std(all_xs_converged)}),  {np.min(all_ys_extended[:, -1])}, {np.max(all_ys_extended[:, -1])}, "
                      f"{np.median(all_ys_extended[:,-1], axis=0)}, {np.mean(all_ys_extended[:,-1], axis=0)} (pm {np.std(all_ys_extended[:,-1], axis=0)})]")

                if all_ys_extended.shape[1]>=11:
                    print('AR mean (+-SD) at iteration 10:', f"{np.mean(all_ys_extended[:,10])}, (pm {np.std(all_ys_extended[:,10])})")

                print("___")



            if add_mean:
                median_ys = np.mean(all_ys_extended, axis=0)

                # figure.add_scatter(x=np.arange(max_iteration+1),
                #                    y=median_ys,
                #                    mode='lines',
                #                   # marker=marker_dict,
                #                    line=dict(color='black', dash='dot', width=5.0),
                #                    name=f"Median",
                #                    legendgroup="Median",
                #                    showlegend=show_legend,
                #                    row=row_index,col=col_index,
                # )

                mean_ys = np.mean(all_ys_extended, axis=0)

                figure.add_scatter(x=np.arange(max_iteration + 1),
                                   y=mean_ys,
                                   mode='lines',
                                   # marker=marker_dict,
                                   line=dict(color='firebrick', dash='dash', width=7.5),
                                   name=f"Mean",
                                   legendgroup="Mean",
                                   showlegend=show_legend,
                                   row=row_index, col=col_index,
                                   )


            if y_name in [SNV.ApproximationRatioBest.id_long, SNV.ApproximationRatioMean.id_long]:
                # add horizontal dashed black line for Y = 1
                figure.add_hline(y=1.0, line_width=2.0, line_dash='dash',
                                 row=row_index, col=col_index
                                 )

            figure.update_xaxes(range=bounds_x, ticks='outside', tickvals=ticks_x, title_font=dict(size=64))
            figure.update_yaxes(range=bounds_y, ticks='outside', tickvals=ticks_y, title_font=dict(size=64))

            if show_legend:
                height = 1600
                width = 2600
            else:
                height = 1200
                width = 1600

            figure.update_layout(xaxis_title=nicer_names.get(x_name, x_name),
                                 xaxis_title_font=dict(size=64),
                                 yaxis_title=nicer_names.get(y_name, y_name),
                                 yaxis_title_font=dict(size=64),
                                 height=height,
                                 width=width,
                                 font=dict(size=64),
                                 # but for title smaller font
                                 title=dict(text=figure_title, font=dict(size=32)),
                                 # row=row_index, col=col_index,
                                 )

        if return_data:
            return figure, (all_xs_converged, all_ys), None

        return figure

    def create_ndar_overview_plot(self,
                                  figure_of_merit_name: str,
                                  figure_title: Optional[str] = None,
                                  in_3d: bool = False,
                                  simulation: bool = False,
                                  markersize_scatter=5.0,
                                  colormap_name='Viridis',
                                  include_only_best_run: bool = True,

                                  #   add_simulation:bool = True,
                                  add_mean: bool = False,
                                  show_legend: bool = True,
                                  print_data: bool = True,
                                  hamiltonian_classes=None,
                                  hamiltonian_instances=None,
                                  return_data=False,
                                  # TODO(FBM); this should be inferred
                                  best_out_of: Optional[int] = None,
                                  ):

        fom_name = figure_of_merit_name


        return self._create_ndar_overview_plot_absolute_metrics(
            figure_of_merit_name=figure_of_merit_name,
            figure_title=figure_title,
            in_3d=in_3d,
            simulation=simulation,
            markersize_scatter=markersize_scatter,
            colormap_name=colormap_name,
            include_only_best_run=include_only_best_run,
            add_mean=add_mean,
            show_legend=show_legend,
            print_data=print_data,
            hamiltonian_classes=hamiltonian_classes,
            hamiltonian_instances=hamiltonian_instances,
            return_data=return_data,
            best_out_of=best_out_of,
        )

