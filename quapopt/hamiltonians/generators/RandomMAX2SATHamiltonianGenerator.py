# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

 
from typing import List, Tuple, Optional, Union

import numpy as np

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase

# Define the type of the clause
#first element is the sign of the variable, second element is the variable index
ClauseFormat = Tuple[Tuple[int, int], Tuple[int, int]]
from tqdm.notebook import tqdm



from quapopt.data_analysis.data_handling import (
    HamiltonianClassSpecifierMAXkSAT,
    CoefficientsDistributionSpecifier,
    ERDOS_RENYI_TYPES)


#TODO(FBM): refactor those functions

def get_mapping_2SAT(clause: ClauseFormat,
                     qubit_index: int) -> int:
    """
    Map clause to qubit energy
    :param clause:
    :param qubit_index:
    :return:
    """
    signs, variables = clause

    if qubit_index not in variables:
        return 0
    else:
        location = int(np.where(np.array(variables) == qubit_index)[0])
        return signs[location]




def get_local_fields_from_2SAT_clauses(qubit_index: int,
                                       clauses_list: List[ClauseFormat]):
    """
    Convert 2SAT clauses to local fields
    :param qubit_index:
    :param clauses_list:
    :return:
    """
    return -sum([get_mapping_2SAT(clause=cl,
                                  qubit_index=qubit_index) for cl in clauses_list])

def get_interaction_coefficients_from_2SAT_clauses(pair_indices: Tuple[int, int],
                                                   clauses_list: List[ClauseFormat]):
    """
    Convert 2SAT clauses to ZZ interactions
    :param pair_indices:
    :param clauses_list:
    :return:
    """
    return sum(
        [get_mapping_2SAT(clause=cl,
                          qubit_index=pair_indices[0]) *
         get_mapping_2SAT(clause=cl,
                          qubit_index=pair_indices[1]) for
         cl in clauses_list])


def generate_random_2SAT_clauses(number_of_variables: int,
                                 # clause_density: float,
                                 number_of_clauses: int,
                                 numpy_rng: Optional[np.random.Generator] = None,
                                 ) -> List[ClauseFormat]:
    #TODO(FBM): make fast version of this
    if numpy_rng is None:
        numpy_rng = np.random.default_rng()

    # TODO: WARNING! might want to modify to randomly oscillate around alpha * number_of_variables instead of fixed number
    if number_of_clauses > (number_of_variables - 1) * number_of_variables / 2:
        raise ValueError("Clause density too high.")

    random_clauses = []
    all_pairs = [(i, j) for i in range(number_of_variables) for j in range(i + 1, number_of_variables)]

    # not particularly sophisticated sampling without repetitions
    #TODO(FBM): make this more efficient?
    while len(random_clauses) < number_of_clauses:
        pair_index = numpy_rng.integers(0, len(all_pairs))
        var_i, var_j = all_pairs[pair_index]

        pair_tuple = tuple(sorted([var_i, var_j]))

        sign_i = numpy_rng.choice([-1, 1])
        sign_j = numpy_rng.choice([-1, 1])

        clause_now = ((int(sign_i), int(sign_j)), pair_tuple)
        if clause_now not in random_clauses:
            random_clauses.append(clause_now)
    return random_clauses



def convert_2SAT_clauses_to_ising(clauses_list: List[ClauseFormat],
                                  number_of_qubits: int) -> List[Tuple[Union[float,int], Tuple[int, ...]]]:
    """
    Converts random MAX2SAT clauses to two-local Hamiltonian
    :param clauses_list:
    :param number_of_qubits:
    :return:
    """

    hamiltonian_list = []
    for qi in tqdm(list(range(number_of_qubits)),position=0,colour='blue',disable=True):
        h_i = get_local_fields_from_2SAT_clauses(qubit_index=qi,
                                                 clauses_list=clauses_list)
        if h_i != 0:
            hamiltonian_list.append((h_i, (qi,)))

        for qj in range(qi + 1, number_of_qubits):
            J_ij = get_interaction_coefficients_from_2SAT_clauses(pair_indices=(qi, qj), clauses_list=clauses_list)

            if J_ij != 0:
                hamiltonian_list.append((J_ij, (qi, qj)))

    return hamiltonian_list
try:
    from numba import cuda
    # #let's write kernel for get_local_fields_from_@SAT_clauses
    @cuda.jit
    def _get_local_fields_from_2SAT_clauses_kernel_cuda(
                                                        clauses_list,
                                                        local_fields):


        idx = cuda.grid(1)

        if idx>=len(local_fields):
            return

        for idx_clause in range(len(clauses_list)):
            signs, variables  = clauses_list[idx_clause]
            for ind, var in enumerate(variables):
                if var == idx:
                    local_fields[idx] += signs[ind]
                    break


    @cuda.jit
    def _get_interaction_coefficients_from_2SAT_clauses_kernel_cuda(clauses_list,
                                                                    interaction_coefficients):

        idx, idy = cuda.grid(2)

        # if idx > idy:
        #     return
        # if idy>=number_of_qubits:
        #     return

        if idy <= idx:
            return
        if idy >= interaction_coefficients.shape[1]:
            return

        for idx_clause in range(len(clauses_list)):
            signs, variables = clauses_list[idx_clause]

            found_x = False
            found_y = False
            for ind, var in enumerate(variables):
                if var == idx:
                    found_x = True
                if var == idy:
                    found_y = True
            if found_x and found_y:
                interaction_coefficients[idx, idy] += signs[0] * signs[1]
except(ImportError,ModuleNotFoundError):
    pass



def get_local_fields_from_2SAT_clauses_cuda(clauses_list: List[ClauseFormat],
                                            number_of_qubits: int):
    """
    Convert 2SAT clauses to local fields
    Args:
        qubit_index:
        clauses_list:
        number_of_qubits:

    Returns:

    """


    local_fields_device = cuda.to_device(np.zeros(number_of_qubits, dtype=np.float64))

    clauses_list_device = cuda.to_device(clauses_list)

    _get_local_fields_from_2SAT_clauses_kernel_cuda[1, number_of_qubits](
                                                                        clauses_list_device,
                                                                        local_fields_device)


    return -local_fields_device.copy_to_host()




def get_interaction_coefficients_from_2SAT_clauses_cuda(clauses_list: List[ClauseFormat],
                                                        number_of_qubits: int):
    """
    Convert 2SAT clauses to ZZ interactions


    Args:
        clauses_list:
        number_of_qubits:

    Returns:

    """


    interaction_coefficients_device = cuda.to_device(np.zeros((number_of_qubits, number_of_qubits), dtype=np.float64))
    clauses_list_device = cuda.to_device(clauses_list)

    threads_per_block = (16, 16)
    N = number_of_qubits
    blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)



    _get_interaction_coefficients_from_2SAT_clauses_kernel_cuda[blocks_per_grid, threads_per_block](clauses_list_device,
                                                                                                    interaction_coefficients_device)

    return interaction_coefficients_device.copy_to_host()

class RandomMAX2SATHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):
    def __init__(self):

        hamiltonian_class_specifier = HamiltonianClassSpecifierMAXkSAT(kSAT=2)

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier)


    def generate_instance(self,
                          number_of_qubits: int,
                          clause_density:float,
                          seed: Optional[int] = None,
                          read_from_drive_if_present=False,
                          default_backend:Optional[str]=None) -> ClassicalHamiltonian:

        hamiltonian_class_specifier = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(number_of_qubits,
                                                                                                    seed,
                                                                                                    clause_density)

        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier)
            if hamiltonian is not None:
                return hamiltonian



        numpy_rng = np.random.default_rng(seed)
        number_of_clauses = int(clause_density * number_of_qubits)
        clauses_list = generate_random_2SAT_clauses(number_of_variables=number_of_qubits,
                                                    number_of_clauses=number_of_clauses,
                                                    numpy_rng=numpy_rng)



        #print('got clauses, converting to Ising')
        from quapopt import AVAILABLE_SIMULATORS
        if number_of_qubits>150 and 'cuda' in AVAILABLE_SIMULATORS:
            local_fields = get_local_fields_from_2SAT_clauses_cuda(clauses_list=clauses_list,
                                                number_of_qubits=number_of_qubits)
            interaction_coefficients = get_interaction_coefficients_from_2SAT_clauses_cuda(clauses_list=clauses_list,
                                                                                                    number_of_qubits=number_of_qubits)
            hamiltonian_list = []
            for qi in tqdm(list(range(number_of_qubits)),position=0,colour='green',disable=True):
                h_i = local_fields[qi]
                if h_i != 0:
                    hamiltonian_list.append((h_i, (qi,)))

                for qj in range(qi + 1, number_of_qubits):
                    J_ij = interaction_coefficients[qi, qj]

                    if J_ij != 0:
                        hamiltonian_list.append((J_ij, (qi, qj)))
        else:
            hamiltonian_list = convert_2SAT_clauses_to_ising(clauses_list=clauses_list,
                                                         number_of_qubits=number_of_qubits)

        class_specific_information = dict(clauses_list=clauses_list)




        hamiltonian = ClassicalHamiltonian(hamiltonian_list_representation=hamiltonian_list,
                                           number_of_qubits=number_of_qubits,
                                           hamiltonian_class_specifier=hamiltonian_class_specifier,
                                           hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                           class_specific_data=class_specific_information)

        return self._generate_instance(number_of_qubits=number_of_qubits,
                                       random_instance=hamiltonian,
                                       default_backend=default_backend)




if __name__ == '__main__':
    # Example of usage
    number_of_variables_test = 10
    number_of_clauses_test = 40
    RM2SHG = RandomMAX2SATHamiltonianGenerator(number_of_variables=number_of_variables_test,
                                               number_of_clauses=number_of_clauses_test)
