# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
 
from quapopt.optimization.QAOA.implementation.QAOARunnerSampler import QAOARunnerSampler
import optuna
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
import numpy as np
from quapopt.optimization.QAOA.simulation import QAOARunnerExpValues
from quapopt.meta_algorithms.QRR import QRR_functions


def get_1q_rdms_diag(Zi_values):
    rdms_1q_list = []
    for i, Z in enumerate(Zi_values):
        # print(f"Z_{i} = {Z}")
        p0 = (1 + Z) / 2
        p1 = 1 - p0
        rdm = np.array([p0, p1])
        rdms_1q_list.append(((i,), rdm))
    return rdms_1q_list


def get_2q_rdms_diag(Zij_values,
                     Zi_values):
    rdms_2q_list = []
    for i in range(Zij_values.shape[0]):
        Zi = Zi_values[i]
        for j in range(i + 1, Zij_values.shape[0]):
            Zij = Zij_values[i, j]
            Zj = Zi_values[j]
            p00 = 1 / 4 * (1 + Zi + Zj + Zij)
            p01 = 1 / 4 * (1 + Zi - Zj - Zij)
            p10 = 1 / 4 * (1 - Zi + Zj - Zij)
            p11 = 1 / 4 * (1 - Zi - Zj + Zij)
            rdm = np.array([p00, p01, p10, p11])
            rdms_2q_list.append(((i, j), rdm))

    return rdms_2q_list

__default_hyperparameters_TPE__ = {"consider_prior": True,
                                   "prior_weight": 1.0,
                                   "consider_magic_clip": True,
                                   "consider_endpoints": False,
                                   "n_startup_trials": 10,
                                   "n_ei_candidates": 24,
                                   # "gamma": default_gamma,
                                   # "weights": default_weights,
                                   # "seed": Optional[int] = None,
                                   "multivariate": False,
                                   "group": False,
                                   "constant_liar": False}

class OptunaQAOAOptimizerSampler:
    def __init__(self,
                 qaoa_sampler: QAOARunnerSampler,
                 study_storage = None,
                 ):

        self._qaoa_sampler = qaoa_sampler
        self._study_storage = study_storage

        raise DeprecationWarning("This class is deprecated. Use QAOAOptimizationRunner instead.")
    @property
    def qaoa_sampler(self):
        return self._qaoa_sampler
    @property
    def study_storage(self):
        return self._study_storage


    def run_optimization(self,
                         qaoa_depth: int,
                         number_of_function_calls:int,
                         classical_optimizer:optuna.samplers.BaseSampler=None,
                         optimizer_seed=None,
                         study_name=None,
                         number_of_samples: int = None,
                         measurement_noise: ClassicalMeasurementNoiseSampler = None,
                         numpy_rng_sampling=None,
                         number_of_threads_optuna=1,
                         show_progress_bar=False,
                         #TODO FBM: refactor this
                         weird_version_temp=False,
                         verbosity=0,
                         optuna_pruner: optuna.pruners.BasePruner = None,
                         **additional_kwargs):



        if weird_version_temp:
            hamiltonian_dict = {tup: coeff for coeff, tup in self.qaoa_sampler.hamiltonian_representations[0].hamiltonian_list_representation}

        def __objective_function(trial: optuna.Trial):

            # TODO FBM: refactor this
            if not weird_version_temp:
                return self._qaoa_sampler.run_qaoa_wrapped(trial,
                                                           qaoa_depth=qaoa_depth,
                                                           number_of_samples=number_of_samples,
                                                           measurement_noise=measurement_noise,
                                                           numpy_rng_sampling=numpy_rng_sampling,
                                                           input_format='optuna',
                                                           **additional_kwargs)
            energy_mean = self._qaoa_sampler.run_qaoa_wrapped(trial,
                                                        qaoa_depth=qaoa_depth,
                                                        number_of_samples=number_of_samples,
                                                        measurement_noise=measurement_noise,
                                                        numpy_rng_sampling=numpy_rng_sampling,
                                                        input_format='optuna',
                                                        **additional_kwargs)

            # TODO FBM: clean this up
            # Zi_values = self._qaoa_sampler.correlators_history[(0,)][trial._trial_id]['Ci_noiseless']
            # Zij_values = self._qaoa_sampler.correlators_history[(0,)][trial._trial_id]['Cij_noiseless']
            # for i in range(Zij_values.shape[0]):
            #     Zi_values[i] = Zi_values[i] / hamiltonian_dict[(i,)]
            #     for j in range(i + 1, Zij_values.shape[0]):
            #         Zij_values[i, j] = Zij_values[i, j] / hamiltonian_dict[(i, j)]
            #
            # rdms_1q_list = get_1q_rdms_diag(Zi_values)
            # rdms_2q_list = get_2q_rdms_diag(Zij_values, Zi_values)
            # rdms_dict = {tup: rdm for tup, rdm in rdms_1q_list + rdms_2q_list}
            # register = [(0, 0), (0, 1), (1, 0), (1, 1)]
            # linear_chain = [(i, i + 1) for i in range(0, self.qaoa_sampler.number_of_qubits - 1, 2)]
            # candidate_state = [register[np.argmax(rdms_dict[tup])] for tup in linear_chain]
            # flattened_state = tuple([item for sublist in candidate_state for item in sublist])
            #
            # energy_state = em.calculate_energies_from_bitstrings(bitstrings_array=[flattened_state],
            #                                                      hamiltonian=self.qaoa_sampler.hamiltonian_representations[0].hamiltonian_list_representation)[0]



            return energy_mean

        if classical_optimizer is None:
            classical_optimizer = optuna.samplers.TPESampler(seed=optimizer_seed,
                                                             **__default_hyperparameters_TPE__)
        if optuna_pruner is None:
            optuna_pruner = optuna.pruners.NopPruner()

        optuna_study = optuna.create_study(sampler=classical_optimizer,
                                           pruner=optuna_pruner,
                                           direction='minimize',
                                           storage=self.study_storage,
                                           study_name=study_name,
                                           load_if_exists=False)

        # self._optuna_study.
        optuna.logging.set_verbosity(verbosity)
        optuna_study.optimize(__objective_function,
                                   n_trials=number_of_function_calls,
                                   n_jobs=number_of_threads_optuna,
                                   show_progress_bar=show_progress_bar)


        return self._qaoa_sampler.get_best_results(), optuna_study
        # return self._optuna_study



class OptunaQAOAOptimizerSamplerQRR:
    def __init__(self,
                 qaoa_sampler: QAOARunnerExpValues,
                 optimizer_seed=None,
                 study_storage = None,
                 classical_optimizer:optuna.samplers.BaseSampler = None,
                 optuna_pruner:optuna.pruners.BasePruner = None
                 ):



        #TODO(FBM): make QRR external to the QAOA sampler

        raise NotImplementedError("This class is deprecated. Will make QRR external to the QAOA sampler.")

        super().__init__(qaoa_sampler=qaoa_sampler,
                         optimizer_seed=optimizer_seed,
                         study_storage=study_storage,
                         classical_optimizer=classical_optimizer,
                         optuna_pruner=optuna_pruner)
    def run_optimization(self,
                         number_of_function_calls:int,
                         number_of_samples: int,
                         #dummy variable, only for compatibility with the other optimizer TODO(FBM): remove
                         qaoa_depth: int = None,
                         memory_intensive=True,
                         measurement_noise: ClassicalMeasurementNoiseSampler = None,
                         add_noise_to_phase = True,
                         add_noisy_constant=True,
                         #weighted_QRR=True,
                         include_1q_terms=True,
                         numpy_rng_sampling=None,
                         number_of_threads_optuna=1,
                         show_progress_bar=False,
                         weird_version_temp = False):

       # weird_version_temp = True

        if measurement_noise is not None:
            if measurement_noise.noisy_hamiltonian_arrays['cost'] is None:
                measurement_noise.add_noisy_hamiltonian_representations(
                    hamiltonian_representations_dict=self.qaoa_sampler.hamiltonian_representations_cost,
                hamiltonian_identifier='cost')
                if add_noise_to_phase:
                    measurement_noise.add_noisy_hamiltonian_representations(
                        hamiltonian_representations_dict=self.qaoa_sampler.hamiltonian_representations_phase,
                        hamiltonian_identifier='phase')

        if weird_version_temp:
            hamiltonian_dict = {tup:coeff for coeff, tup in self.qaoa_sampler.hamiltonian_representations[0]}

        def __objective_function(trial: optuna.Trial):

            return self._qaoa_sampler.run_qaoa_wrapped(trial,
                                                           # qaoa_depth=qaoa_depth,
                                                           #number_of_samples=number_of_samples,
                                                           measurement_noise=measurement_noise,
                                                           #numpy_rng_sampling=numpy_rng_sampling,
                                                           add_noise_to_phase=add_noise_to_phase,
                                                           input_format='optuna',
                                                           store_correlators=True,
                                                               memory_intensive=memory_intensive, )


        self._optuna_study.optimize(__objective_function,
                                   n_trials=number_of_function_calls,
                                   n_jobs=number_of_threads_optuna,
                                   show_progress_bar=show_progress_bar)


        all_correlators = self.qaoa_sampler.correlators_history
        best_energy, best_solution, best_rep_index = np.inf, None, None

        for hamiltonian_tup, trials_i in all_correlators.items():
            hamiltonian_index=hamiltonian_tup[0]
            hamiltonian_i = self.qaoa_sampler.hamiltonian_representations[hamiltonian_index]
            for trial_index, correlators in trials_i.items():
                if measurement_noise is None:
                    C_i, C_ij = correlators['Ci_noiseless'], correlators['Cij_noiseless']
                else:
                    C_i, C_ij = correlators['Ci_noisy'], correlators['Cij_noisy']

                if include_1q_terms:
                    for qi, exp_qi in enumerate(C_i):
                        C_ij[(qi, qi)] = exp_qi

                C_ij_weighted = QRR_functions.add_weights_to_correlations_matrix(correlations_matrix=C_ij,
                                                                            hamiltonian_list=hamiltonian_i.hamiltonian_list_representation)

                flatten_upper_triangle = []
                flatten_upper_triangle_weighted = []
                indices_subsets = []
                for i in range(len(C_i)):
                    for j in range(i, len(C_i)):
                        flatten_upper_triangle.append(C_ij[(i,j)])
                        flatten_upper_triangle_weighted.append(C_ij_weighted[(i,j)])
                        indices_subsets.append((i,j))

                noisy_constant = None
                if add_noisy_constant:
                    noisy_constant = measurement_noise.noisy_hamiltonian_representations['cost'][hamiltonian_index][1]
                    noisy_constant/=self.qaoa_sampler.number_of_qubits

                candidate_solutions = QRR_functions.find_candidate_solutions_QRR(expected_values=flatten_upper_triangle,
                                                                                 qubit_subsets=indices_subsets,
                                                                                 include_1q_terms=include_1q_terms,
                                                                                 return_correlations_matrix=False,
                                                                                 constant=noisy_constant)
                candidate_solutions_weighted = QRR_functions.find_candidate_solutions_QRR(expected_values=flatten_upper_triangle_weighted,
                                                                                    qubit_subsets=indices_subsets,
                                                                                    include_1q_terms=include_1q_terms,
                                                                                    return_correlations_matrix=False,
                                                                                          constant=noisy_constant)
                join_candidate_solutions = np.concatenate((candidate_solutions, candidate_solutions_weighted), axis=0)

                candidate_energies = hamiltonian_i.evaluate_energy(bitstrings_array=join_candidate_solutions)

                best_energy_index = np.argmin(candidate_energies)
                if candidate_energies[best_energy_index] < best_energy:
                    best_energy = candidate_energies[best_energy_index]
                    best_solution = candidate_solutions[best_energy_index]
                    best_rep_index=hamiltonian_index





        #TODO(FBM):

        return [(best_energy, (best_solution,best_rep_index))]

        #return self._qaoa_sampler.get_best_results()









