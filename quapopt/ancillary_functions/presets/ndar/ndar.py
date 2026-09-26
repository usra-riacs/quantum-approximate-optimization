# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from quapopt import ancillary_functions as anf
from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
import pandas as pd
from quapopt import ancillary_functions as anf

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

from quapopt.data_analysis.data_handling import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 STANDARD_NAMES_VARIABLES as SNV, ResultsLogger, LoggingLevel)
from quapopt.ancillary_functions.presets.ndar._names import LocalSamplerName
from quapopt.meta_algorithms.NDAR import NDARIterationResult, handle_bitstring_format_inefficient


from quapopt.optimization.classical_solvers.LocalSearch.greedy_swap import run_single_swap_choose_best_over_neighborhood
from quapopt.circuits import backend_utilities as bck_utils


def _offline_angle_simulator_name():
    """The simulator the offline p=1 angle optimization asks for.

    CUDA where the machine has it, the CPU implementation otherwise. The expectation-value
    runner reserves CUDA for more than 30 qubits when the caller chooses nothing, which is
    the right rule for a single large run but not here: this optimization is many small p=1
    evaluations, where CUDA pays off at any width. Asking for it unconditionally is what
    made the whole path need a GPU, so the preference is stated here and the CPU
    implementation carries the rest.
    """
    from quapopt import AVAILABLE_SIMULATORS

    return 'cuda' if 'cuda' in AVAILABLE_SIMULATORS else 'cython'


def _reformat_df_to_ndar(df:pd.DataFrame,
                          r_best_results:int=1,
                         additional_data_row_callable:Optional[Callable[[pd.DataFrame], Optional[pd.DataFrame]]]=None,)->List[NDARIterationResult]:
    # sort w.r.t. EnergyBest
    # take best r results:

    df = df.sort_values(by=[SNV.EnergyBest.id_long], ascending=True)

    if r_best_results>1:
        #we want pairs of (ham_rep_index, bitstring) to be unique
        df['(Bitstring, i)'] = df.apply(lambda row: (row[SNV.BitstringBest.id_long],
                                                     row[SNV.HamiltonianRepresentationIndex.id_long]), axis=1)
        df = df.drop_duplicates(subset=['(Bitstring, i)'], keep='first')
        df = df.drop(columns=['(Bitstring, i)'])

    df = df.head(r_best_results).copy()

    #print('hejka4', df)

    results_format_ndar = []
    for index, row in df.iterrows():
        best_energy = row[SNV.EnergyBest.id_long]
        best_bitstring = handle_bitstring_format_inefficient(row[SNV.BitstringBest.id_long])
        best_rep_index = row[SNV.HamiltonianRepresentationIndex.id_long]
        tup_0 = (best_energy, best_bitstring, best_rep_index)

        additional_data_row = pd.DataFrame([row])

        # print('hejka4',additional_data_row)

        if additional_data_row_callable is not None:
            additional_data_row = additional_data_row_callable(additional_data_row)

        results_format_ndar.append((tup_0, additional_data_row))

    return results_format_ndar





def decorate_kwargs_updater_with_logger_NDAR(sampler_name:Optional[str]=None,
                                             logger_kwargs_main:Optional[dict]=None,
                                             local_kwargs_updater:Optional=None,
                                             ):
    if local_kwargs_updater is None:
        local_kwargs_updater = lambda x,y,z: {}


    def local_kwargs_updater_with_logger_kwargs(ndar_iteration_index: int,
                                            results_format_ndar: Tuple[float, Tuple[int], int],
                                            additional_data=None):

        results_logger = None
        if logger_kwargs_main is not None:
            _prefix = f"{SNV.NDARIteration.id}={ndar_iteration_index};{SNV.SamplerName.id}={sampler_name}"
            results_logger = ResultsLogger(**logger_kwargs_main,
                                           table_name_prefix=_prefix)
            results_logger.set_logging_level(level=LoggingLevel.VERY_DETAILED)

        kwargs_here = local_kwargs_updater(ndar_iteration_index=ndar_iteration_index,
                                           results_format_ndar=results_format_ndar,
                                           additional_data=additional_data)

        kwargs_here['results_logger'] = results_logger

        return kwargs_here

    return local_kwargs_updater_with_logger_kwargs



def build_local_sampler_callable_qiskit_with_offline_angles_p1_NDAR(
                                                        number_of_samples_qiskit:int,
                                                        number_of_function_calls_simulator:int,
                                                        optimize_bias_offline:bool,
                                                        qiskit_config_dict,
                                                        simulator_config_dict,
                                                        ansatz_config_dict,
                                                        r_best_results:int=1,
                                                        angles_grid_callable: Optional[Callable[[float, float], Tuple[List[float], List[float]]]]=None,
                                                        verbosity:int=1,
no_gauges_version:bool=False,
add_hdls:bool=False
):



    from quapopt.optimization.QAOA.implementation.QAOARunnerSampler import QAOARunnerSampler
    from quapopt.optimization.QAOA.simulation.QAOARunnerExpValuesRDMs import QAOARunnerExpValuesRDMs
    from quapopt.optimization.parameter_setting.variational.QAOAOptimizationRunner import QAOAOptimizationRunner
    from quapopt.optimization.QAOA import QubitMappingType
    from quapopt.optimization.QAOA.circuits.time_block_ansatz import TimeBlockBatchingType
    from quapopt.optimization.QAOA import PhaseSeparatorType, MixerType, InitialStateType, QAOAResult

    if angles_grid_callable is None:
        angles_grid_callable = lambda gamma, beta: ([gamma], [beta])



    #ansatz config
    time_block_size = ansatz_config_dict.get('time_block_size', None)
    time_block_seed_outer = ansatz_config_dict.get('time_block_seed', -1)
    phase_separator_type = ansatz_config_dict.get('phase_separator_type', PhaseSeparatorType.QAOA)
    mixer_type = ansatz_config_dict.get('mixer_type', MixerType.ws_qaoa_identical)
    initial_state_type = ansatz_config_dict.get('initial_state_type', InitialStateType.ws_qaoa_identical)


    #qiskit config:
    qubit_mapping_type = qiskit_config_dict['qubit_mapping_type']
    session_ibm = qiskit_config_dict['session_ibm']
    qiskit_backend = qiskit_config_dict['qiskit_backend']
    backend_name = qiskit_config_dict['backend_name']


    qiskit_pass_manager = qiskit_config_dict.get('qiskit_pass_manager', None)
    pass_manager_kwargs = qiskit_config_dict.get('pass_manager_kwargs', None)
    pass_manager_seeds_list = qiskit_config_dict.get('pass_manager_seeds_list', None)


    qiskit_sampler_options: Optional[Dict[str, Any]] = qiskit_config_dict.get('qiskit_sampler_options', None)



    gate_builder = qiskit_config_dict['gate_builder']
    qubit_indices_physical = qiskit_config_dict['qubit_indices_physical']

    simulation = qiskit_config_dict['simulation']
    noiseless_simulation = qiskit_config_dict['noiseless_simulation']

    time_block_batching_type = ansatz_config_dict.get('time_block_batching_type', None)

    if time_block_batching_type is None:
        if qubit_mapping_type in [QubitMappingType.sabre, QubitMappingType.fully_connected]:
            time_block_batching_type = TimeBlockBatchingType.FRACTIONAL
        elif qubit_mapping_type == QubitMappingType.linear_swap_network:
            time_block_batching_type = TimeBlockBatchingType.SWAP_NETWORK

        else:
            raise ValueError("Unsupported qubit mapping type for time block batching type assignment.")



    #simulator config:
    classical_optimizer = simulator_config_dict['classical_optimizer']
    betas_search_space_size = simulator_config_dict.get('betas_search_space_size', None)
    efficient_finding_of_best_beta = simulator_config_dict.get('efficient_finding_of_best_beta', True)
    # None keeps the seed the optimizer was configured with, which is what makes an unpinned
    # angle search reproducible; an integer overrides it.
    optimizer_seed = simulator_config_dict.get('optimizer_seed', None)
    # How the mixer angle is found for each phase angle: None takes the closed form where it
    # applies, False always searches the grid of betas_search_space_size points.
    analytical_betas = simulator_config_dict.get('analytical_betas', None)




    def local_sampler_callable_qiskit_with_offline_angles(classical_hamiltonian_representations:List[ClassicalHamiltonian],
                                                          bias_parameters_WS_list:Optional[List[float]] = None,
                                                          results_logger: Optional[ResultsLogger] = None,
                                                          time_block_seed: Optional[int] = None,
                                                          ndar_iteration_index: Optional[int] = None,
                                                          ):


        if verbosity>0:
            anf.cool_print("\nInitializing qiskit simulator",'...','green')

        if time_block_seed is None:
            time_block_seed = time_block_seed_outer


        if no_gauges_version:
            # recover original Hamiltonian representation -- we do not gauge-transform the phase Hamiltonian
            classical_hamiltonian_representations_original = [x.recover_original_hamiltonian_representation() for x in classical_hamiltonian_representations]
        else:
            classical_hamiltonian_representations_original = classical_hamiltonian_representations

        #print('hejka2',len(classical_hamiltonian_representations))
        qaoa_sampler:QAOARunnerSampler = QAOARunnerSampler(hamiltonian_representations_cost = classical_hamiltonian_representations_original,
                                                           logger_kwargs =results_logger.config.to_dict() if results_logger is not None else None,
                                                           logging_level =  results_logger.logging_level if results_logger is not None else None
                                                           )
        qiskit_sampler_options_applied = qiskit_sampler_options.copy() if qiskit_sampler_options is not None else None


        if qiskit_sampler_options_applied is not None:
            if 'seed_simulator' in qiskit_sampler_options_applied and qiskit_sampler_options_applied['seed_simulator'] is not None:
                qiskit_sampler_options_applied['seed_simulator'] += ndar_iteration_index*10**4



        qaoa_sampler.initialize_backend_qiskit(simulation=simulation,
                                               noiseless_simulation=noiseless_simulation,
                                               session_ibm=session_ibm,
                                               qiskit_backend=qiskit_backend,
                                               qiskit_pass_manager=qiskit_pass_manager,
                                               pass_manager_kwargs=pass_manager_kwargs,
                                               pass_manager_seeds_list=pass_manager_seeds_list,
                                               program_gate_builder=gate_builder,
                                               qubit_mapping_type=qubit_mapping_type,
                                               qaoa_depth=1,
                                               time_block_size=time_block_size,
                                               time_block_seed=time_block_seed,
                                               phase_separator_type=phase_separator_type,
                                               mixer_type=mixer_type,
                                               initial_state=initial_state_type,
                                               qubit_indices_physical=qubit_indices_physical,
                                               qiskit_sampler_options=qiskit_sampler_options_applied,
                                               )


        if bias_parameters_WS_list is None:
            bias_parameters_WS_list = [0.5]

        if verbosity>0:
            anf.cool_print("Done, optimizing angles and bias offline",'...','blue')


        simulator_results = {}
        for rep_index, (ham_i, ham_i_original) in enumerate(zip(classical_hamiltonian_representations, classical_hamiltonian_representations_original)):
            dfs_list_sim_i = []

            for c_param in bias_parameters_WS_list:
                qaoa_simulator_exp_values = QAOARunnerExpValuesRDMs(hamiltonian_representations_cost=[ham_i],
                                                                    time_block_size=time_block_size,
                                                                    time_block_seed=time_block_seed,
                                                                    time_block_batching_type=time_block_batching_type,
                                                                    ws_bias_parameters=c_param,
                                                                    simulator_name=_offline_angle_simulator_name())

                qaoa_optimizer_simulator = QAOAOptimizationRunner(qaoa_runner=qaoa_simulator_exp_values)

                x, y = qaoa_optimizer_simulator.run_optimization(qaoa_depth=1,
                                                                 number_of_function_calls=number_of_function_calls_simulator,
                                                                 number_of_samples=np.inf,
                                                                 classical_optimizer=classical_optimizer,
                                                                 verbosity=1,
                                                                 show_progress_bar=False,
                                                                 store_correlators=False,
                                                                 find_best_beta_only=efficient_finding_of_best_beta,
                                                                 betas_search_space_size=betas_search_space_size,
                                                                 analytical_betas=analytical_betas,
                                                                 optimizer_seed=optimizer_seed)

                best_res_simulator = x[0]
                best_energy_simulator = best_res_simulator[0]
                best_angles_simulator = best_res_simulator[1][1][0][2]

                best_gamma_simulator = best_angles_simulator[0]
                best_beta_simulator = best_angles_simulator[1]

                df_sim_i = pd.DataFrame(data={SNV.Backend.id_long:[f'RDMsSimulator-{qaoa_simulator_exp_values.simulator_name}'],
                                              SNV.Simulated.id_long:[True],
                                              SNV.HamiltonianRepresentationIndex.id_long:[rep_index],
                                              SNV.WSBiasParameters.id_long:[c_param],
                                              SNV.Angles.id_long:[[best_gamma_simulator, best_beta_simulator]],
                                              SNV.EnergyMean.id_long:[best_energy_simulator]
                                              })
                dfs_list_sim_i.append(df_sim_i)

            df_sim_i = pd.concat(dfs_list_sim_i, axis=0, ignore_index=True)



            df_sim_i = df_sim_i.sort_values(by=SNV.EnergyMean.id_long, ascending=True)

            simulator_results[rep_index] = df_sim_i

        if verbosity>0:
            anf.cool_print("Done, running qiskit",'...','blue')


        all_dfs_qiskit = []
        for rep_index, (ham_i, ham_i_original) in enumerate(zip(classical_hamiltonian_representations, classical_hamiltonian_representations_original)):
            df_sim_i = simulator_results[rep_index]

            if optimize_bias_offline:
                # in this case, we only use the bias that was optimal in simulations
                angles_sim_i = df_sim_i[SNV.Angles.id_long].values[0]
                c_sim_i = df_sim_i[SNV.WSBiasParameters.id_long].values[0]

                gammas_list = [angles_sim_i[0]]
                betas_list = [angles_sim_i[1]]
                c_params_list = [c_sim_i]

            else:
                # otherwise, we implement all biases
                angles_sim_i = df_sim_i[SNV.Angles.id_long].values
                c_params_list = df_sim_i[SNV.WSBiasParameters.id_long].values

                gammas_list = [ang[0] for ang in angles_sim_i]
                betas_list = [ang[1] for ang in angles_sim_i]

            if no_gauges_version:
                bitflip_combined, permutation_combined = ham_i.get_concatenated_transformations()

                if bitflip_combined is None:
                    bitflip_combined = [0]*ham_i.number_of_qubits

                # The bias pattern below warm-starts toward the current frame's all-zeros
                # state, written in the original frame. That image is (0 o p) XOR b = b for
                # any accumulated permutation p, so the flip alone is the pattern.
                # NDAR itself applies bitflips only: this initialization and the back-map
                # `best_bitstring ^ bitflip_combined` further down both assume the identity
                # permutation, and under a permutation the back-map would be (s XOR b) o p^-1.


                gammas_batch = []
                betas_batch = []
                bias_batch = []
                for g,b,c in zip(gammas_list, betas_list, c_params_list):
                    # bias_parameters_WS_list.append(c)
                    gammas_add, betas_add = angles_grid_callable(g,b)
                    gammas_batch+=gammas_add
                    betas_batch+=betas_add

                    local_biases = []
                    for b_i in bitflip_combined:
                        if b_i == 0:
                            local_biases.append(c)
                        else:
                            local_biases.append(1-c)

                    bias_batch+=[local_biases]*len(gammas_add)

                gammas_batch = np.array(gammas_batch).reshape(-1,1)
                betas_batch = np.array(betas_batch).reshape(-1,1)
                bias_batch = np.array(bias_batch)

            else:


                gammas_batch = []
                betas_batch = []
                bias_batch = []
                for g,b,c in zip(gammas_list, betas_list, c_params_list):
                    # bias_parameters_WS_list.append(c)
                    gammas_add, betas_add = angles_grid_callable(g,b)

                    gammas_batch+=gammas_add
                    betas_batch+=betas_add
                    bias_batch+=[c]*len(gammas_add)


                gammas_batch = np.array(gammas_batch).reshape(-1,1)
                betas_batch = np.array(betas_batch).reshape(-1,1)
                bias_batch = np.array(bias_batch).reshape(-1,1)
            #
            # def depth_filter_function(x):
            #     _test_1 = not getattr(x.operation, '_directive', False)
            #     _test_2 = x.operation.name.lower() != 'rz'
            #     return _test_1 and _test_2
            #
            # from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import SabreMappedQAOACircuit
            #
            # circuit:SabreMappedQAOACircuit = qaoa_sampler.backends[0].ansatz.quantum_circuit
            #
            # print(type(circuit))
            #
            # filtered_depth = circuit.depth(depth_filter_function)
            #
            # gates = bck_utils.count_gates_occurences_in_circuit(circuit)
            #
            # circuit_properties = {}
            # circuit_properties['depth'] = filtered_depth
            # circuit_properties['sx'] = gates.get('sx', 0)
            # circuit_properties['rz'] = gates.get('rz', 0)
            # circuit_properties['cz']= gates.get('cz', 0)
            # circuit_properties['x']= gates.get('x', 0)
            # depth_2q = circuit.depth(lambda x: len(x.qubits) > 1)
            # circuit_properties['depth_2q']= depth_2q
            #
            # print(circuit_properties)
            #
            #
            #
            # raise KeyboardInterrupt

            results_qiskit = qaoa_sampler.run_qaoa_qiskit_batch(angles_PHASE_batch = gammas_batch,
                                                               angles_MIXER_batch = betas_batch,
                                                               bias_parameters_WS_batch = bias_batch,
                                                               number_of_samples=number_of_samples_qiskit,
                                                                hamiltonian_representation_index=rep_index)

            if r_best_results != 1:
                raise NotImplementedError('r_best_results!=1 not implemented yet.')
            if add_hdls:

                results_qiskit_hdls = []

                for res_qaoa in results_qiskit:
                    res_qaoa:QAOAResult = res_qaoa
                    bts_qaoa = res_qaoa.bitstrings_array
                    best_energy_hdls, best_bitstring_hdls = run_single_swap_choose_best_over_neighborhood(bitstrings_array=bts_qaoa,
                                                                                                        cost_hamiltonian=ham_i_original if no_gauges_version else ham_i,
                                                                                                        tolerance_1swap=0.0,
                                                                                                        tolerance_2swap=0.0,
                                                                                                        show_progress_bar=False)


                    results_qiskit_hdls.append((res_qaoa, best_energy_hdls, best_bitstring_hdls))


                results_qiskit_sorted = sorted(results_qiskit_hdls, key=lambda x: x[1])

                best_energies = [results_qiskit_sorted[0][1]]
                best_bitstrings = [results_qiskit_sorted[0][2]]
                best_rs = [0]
                #print('applied hdls')

                results_qiskit_sorted = [x[0] for x in results_qiskit_sorted]


            else:

                results_qiskit_sorted = sorted(results_qiskit, key=lambda x: x.energy_best)
                best_energies = [results_qiskit_sorted[0].energy_best]
                best_bitstrings = [results_qiskit_sorted[0].bitstring_best]
                best_rs = [0]

            #
            # if r_best_results == 1:
            #     best_energies = [results_qiskit_sorted[0].energy_best]
            #     best_bitstrings = [results_qiskit_sorted[0].bitstring_best]
            #     best_rs = [0]
            # else:
            #     raise NotImplementedError("r_best_results > 1 is not implemented for the qiskit local sampler with offline angles")
            #     #TODO(FBM): rework this!
            #     all_ens, all_bts, all_res_inds = [], [], []
            #     _indices_res = list(range(len(results_qiskit_sorted[0:r_best_results])))
            #     for res_index, res_object in zip(_indices_res,results_qiskit_sorted[0:r_best_results]):
            #         unique_bts, counts_bts = np.unique(res_object.bitstrings_array,
            #                                            return_counts=True, axis = 0)
            #
            #         if no_gauges_version:
            #             unique_energies = ham_i_original.evaluate_energy(bitstrings_array=unique_bts,
            #                                                 backend_output='numpy')
            #         else:
            #
            #             unique_energies = ham_i.evaluate_energy(bitstrings_array=unique_bts,
            #                                                     backend_output='numpy')
            #         unique_energies_argsort = np.argsort(unique_energies)
            #         unique_energies = unique_energies[unique_energies_argsort][0:r_best_results]
            #         unique_bts = unique_bts[unique_energies_argsort][0:r_best_results]
            #
            #
            #         all_bts+=[unique_bts]
            #         all_ens+=[unique_energies]
            #         all_res_inds+=[np.array([res_index]*len(unique_energies))]
            #
            #     all_ens = np.concatenate(all_ens, axis=0)
            #     all_bts = np.concatenate(all_bts, axis=0)
            #     all_res_inds = np.concatenate(all_res_inds, axis=0)
            #
            #     #now let's find the best r energies:
            #     best_indices = np.argsort(all_ens)[0:r_best_results]
            #     best_energies, best_rs, best_bitstrings= all_ens[best_indices], all_res_inds[best_indices], all_bts[best_indices]

            for r_index, best_energy, best_bitstring in zip(best_rs, best_energies, best_bitstrings):


                if no_gauges_version:
                    #recover original bitstring representation -- we do not gauge-transform the phase Hamiltonian,
                    # so we should store "bitflip" as the bitstring
                    #This is a workaround for the fact that this is inside NDAR loop
                    #that will perform gauge-transformations outside this function.
                    best_bitstring = np.array(best_bitstring)^np.array(bitflip_combined)
                    best_bitstring = best_bitstring.tolist()


                qaoa_res_row_r:QAOAResult = results_qiskit_sorted[r_index]
                #
                # print()
                #
                # print(df_sim_i)
                # print('energy mean:', qaoa_res_row_r.energy_mean)

                #raise KeyboardInterrupt

                df_qiskit_i = pd.DataFrame(data={SNV.Backend.id_long:[f'{backend_name}'],
                                              SNV.Simulated.id_long:[simulation],
                                              SNV.HamiltonianRepresentationIndex.id_long:[rep_index],
                                              SNV.WSBiasParameters.id_long:[float(qaoa_res_row_r.bias_parameters_WS[0])],
                                              SNV.Angles.id_long:[qaoa_res_row_r.angles.tolist()],
                                              SNV.EnergyMean.id_long:[qaoa_res_row_r.energy_mean],
                                              SNV.EnergyBest.id_long:[best_energy],
                                              SNV.BitstringBest.id_long:[handle_bitstring_format_inefficient(bitstring=best_bitstring)]
                                              })
                all_dfs_qiskit.append(df_qiskit_i)
        df_qiskit = pd.concat(all_dfs_qiskit, axis=0, ignore_index=True)
        #print(df_qiskit)


        df_qiskit = df_qiskit.sort_values(by=[SNV.EnergyBest.id_long], ascending=True)

        results_format_ndar = _reformat_df_to_ndar(df=df_qiskit,
                                        r_best_results=r_best_results
                                                   )




        return results_format_ndar

    return local_sampler_callable_qiskit_with_offline_angles


def build_local_sampler_kwargs_updater_callable_for_qiskit_with_offline_angles(bias_list_iteration_callable:Callable[[int], List[float]],
                                                                              time_block_seed_callable:Callable[[int], int],
                                                                               ):

    def local_sampler_kwargs_updater_qiskit_with_offline_angles(ndar_iteration_index: int,
                                                                results_format_ndar: Tuple[float, Tuple[int], int],
                                                                additional_data:Optional[Any]=None):


        return {'bias_parameters_WS_list': bias_list_iteration_callable(ndar_iteration_index),
                'time_block_seed': time_block_seed_callable(ndar_iteration_index),
                'ndar_iteration_index':ndar_iteration_index}


    return local_sampler_kwargs_updater_qiskit_with_offline_angles




def build_local_sampler_callable_and_updater_NDAR(local_sampler_name: LocalSamplerName,
                                                 r_best_results: int = 1,
                                                 **kwargs
                                                 ):
    logger_kwargs_main = kwargs.get('logger_kwargs_main', None)

    if local_sampler_name != LocalSamplerName.QiskitWithOfflineAnglesP1:
        raise ValueError("Only QiskitWithOfflineAnglesP1 is supported.")

    ansatz_config = kwargs['ansatz_config']
    simulator_config = kwargs['simulator_config']


    number_of_function_calls_simulator = kwargs['number_of_function_calls_simulator']
    angles_grid_callable = kwargs.get('angles_grid_callable', None)


    verbosity = kwargs.get('verbosity', 1)

    qiskit_config = kwargs['qiskit_config']
    no_gauges_version = qiskit_config.get('no_gauges_version', False)
    add_hdls = qiskit_config.get('add_hdls', False)

    number_of_samples_qiskit = kwargs['number_of_samples_qiskit']
    optimize_bias_offline = kwargs['optimize_bias_offline']

    local_sampler = build_local_sampler_callable_qiskit_with_offline_angles_p1_NDAR(number_of_samples_qiskit=number_of_samples_qiskit,
                                                                                    number_of_function_calls_simulator=number_of_function_calls_simulator,
                                                                                    optimize_bias_offline=optimize_bias_offline,
                                                                                    qiskit_config_dict=qiskit_config,
                                                                                    simulator_config_dict=simulator_config,
                                                                                    ansatz_config_dict=ansatz_config,
                                                                                    r_best_results=r_best_results,
                                                                                    angles_grid_callable=angles_grid_callable,
                                                                                    verbosity=verbosity,
                                                                                    no_gauges_version=no_gauges_version,
                                                                                    add_hdls=add_hdls)



    bias_list_iteration_callable = kwargs.get('bias_list_iteration_callable', None)



    if bias_list_iteration_callable is None:
        def bias_list_iteration_callable(ndar_iteration_index):
            if ndar_iteration_index == 0:
                return [0.5]
            elif ndar_iteration_index <= 3:
                return [0.1, 0.05]
            else:
                return [0.05, 0.025]

    time_block_seed_callable = kwargs.get('time_block_seed_callable', None)
    if time_block_seed_callable is None:
        def time_block_seed_callable(ndar_iteration_index):
            return -1


    # if time_block_seed == 'adaptive':




    local_kwargs_updater = build_local_sampler_kwargs_updater_callable_for_qiskit_with_offline_angles(bias_list_iteration_callable=bias_list_iteration_callable,
                                                                                                      time_block_seed_callable=time_block_seed_callable,
                                                                                                      )


    def local_kwargs_updater_with_logger_kwargs(ndar_iteration_index: int,
                                                results_format_ndar: Tuple[float, Tuple[int], int],
                                                additional_data=None):

        results_logger = None
        if logger_kwargs_main is not None:
            _prefix = f"{SNV.NDARIteration.id}={ndar_iteration_index};{SNV.SamplerName.id}={local_sampler_name.value}"
            results_logger = ResultsLogger(**logger_kwargs_main,
                                           table_name_prefix=_prefix)
            results_logger.set_logging_level(level=LoggingLevel.VERY_DETAILED)

        kwargs_here = local_kwargs_updater(ndar_iteration_index=ndar_iteration_index,
                                           results_format_ndar=results_format_ndar,
                                           additional_data=additional_data)

        kwargs_here['results_logger'] = results_logger

        return kwargs_here






    return local_sampler, local_kwargs_updater_with_logger_kwargs



def build_universal_logging_callable():
    def universal_logging_callable(ndar_iteration_result:NDARIterationResult,
                                   res_logger:Optional[ResultsLogger]):

        if res_logger is None:
            return

        df_res = ndar_iteration_result.to_dataframe_main()

        res_logger.write_results(dataframe=df_res,
                                 data_type=SNDT.NDAROverview,
                                 #we don't want to label overview via index or local sampler
                                 # because it should accumulate all local runs
                                 table_name_prefix='',
                                 table_name='',
                                 table_name_suffix=''
                                 )

        return df_res

    return universal_logging_callable




