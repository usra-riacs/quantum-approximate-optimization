# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.


from typing import Tuple, Union, Any, List
import numpy as np
import pydantic as pyd

try:
    import networkit as nk
    networkit_Graph = nk.Graph
except (ImportError, ModuleNotFoundError):
    networkit_Graph = type(None)
    nk = None


from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV, BaseName



class HamiltonianTransformation(pyd.BaseModel):
    transformation: BaseName
    value: Tuple[pyd.conint(ge=0), ...]

    class Config:
        extra = 'forbid'
        #arbitrary types are allowed
        arbitrary_types_allowed = True


_SMALL_REGISTERS = {1:{(0,):False, (1,):True},
                    2:{(0,0):False, (0,1):True, (1,0):True, (1,1):False},
                    3:{(0,0,0):False, (0,0,1):True, (0,1,0):True, (0,1,1):False,
                       (1,0,0):True, (1,0,1):False, (1,1,0):False, (1,1,1):True},
                    4:{(0,0,0,0):False, (0,0,0,1):True, (0,0,1,0):True, (0,0,1,1):False,
                       (0,1,0,0):True, (0,1,0,1):False, (0,1,1,0):False, (0,1,1,1):True,
                       (1,0,0,0):True, (1,0,0,1):False, (1,0,1,0):False, (1,0,1,1):True,
                       (1,1,0,0):False, (1,1,0,1):True, (1,1,1,0):True, (1,1,1,1):False},
                    }


def apply_bitflip_to_hamiltonian(hamiltonian:
                                    Union[networkit_Graph,
                                    List[Tuple[Union[float,int], Tuple[int, ...]]]],
                                 bitflip_tuple: Tuple[pyd.conint(ge=0,
                                                                 le=1), ...]) \
        -> Union[networkit_Graph, List[Tuple[Union[float,int], Tuple[int, ...]]]]:
    """
    Applies a bitflip transformation to the Hamiltonian graph.
    #bitflip transforms the weights as follows
    #for each node:
    # if bitflip_tuple[node] == 0, then the weights are unchanged
    # if bitflip_tuple[node] == 1, then the weights change sign
    #for each edge:
    # if both nodes have bitflip_tuple[node] == 0, then the weights are unchanged
    # if one of the nodes has bitflip_tuple[node] == 1, then the weights change sign
    # if both nodes have bitflip_tuple[node] == 1, then the weights are unchanged

    :param hamiltonian:
    :param bitflip_tuple: tuple of 0s and 1s, where 0 means no change and 1 means change sign
    :return:
    """
    #TODO(FBM): don't use bitflips

    bitflip_set = np.array(list(set(bitflip_tuple)))

    assert not np.any(bitflip_set<0) and not np.any(bitflip_set>1) ,\
        f"Invalid bitflip tuple, it should be a tuple of 0s and 1s.\nvalue:{bitflip_tuple}"

    if isinstance(hamiltonian, networkit_Graph):
        # copy the graph
        number_of_nodes = hamiltonian.numberOfNodes()

        transformed_graph = nk.Graph(number_of_nodes,
                                     weighted=True)

        for edge, original_weight in hamiltonian.iterEdgesWeights():
            node_i, node_j = edge
            if node_i == node_j:
                sign_change = _SMALL_REGISTERS[1][(bitflip_tuple[node_i],)]
            else:
                sign_change = _SMALL_REGISTERS[2][(bitflip_tuple[node_i], bitflip_tuple[node_j])]
            if sign_change:
                new_weight = -original_weight
            else:
                new_weight = original_weight

            transformed_graph.addEdge(node_i, node_j, new_weight)

        return transformed_graph

    elif isinstance(hamiltonian, list):
        transformed_hamiltonian = []
        for interaction in hamiltonian:
            coefficient, node_ids = interaction
            sign_change = _SMALL_REGISTERS[len(node_ids)][tuple(bitflip_tuple[qi] for qi in node_ids)]
            if sign_change:
                new_coefficients = -coefficient
            else:
                new_coefficients = coefficient

            transformed_hamiltonian.append((new_coefficients, node_ids))


        return transformed_hamiltonian
    else:
        raise ValueError("Hamiltonian must be a networkit graph or a list of interactions")


def invert_bitflip(bitflip_tuple: Tuple[pyd.conint(ge=0, le=1), ...]) -> Tuple[int, ...]:
    """Reverse a bitflip.

    Args:
        bitflip_tuple: The bitflip to reverse.

    Returns:
        The reversed bitflip.
    """
    return tuple(1 - bit for bit in bitflip_tuple)

def invert_permutation(permutation_tuple: Tuple[pyd.conint(ge=0), ...]) -> Tuple[int, ...]:
    """Reverse a permutation.

    Args:
        permutation_tuple: The permutation to reverse.

    Returns:
        The reversed permutation.
    """
    return tuple(permutation_tuple.index(i) for i in range(len(permutation_tuple)))

def apply_permutation_to_array(permutation: Tuple[pyd.conint(ge=0), ...],
                               array_to_permute: np.ndarray) -> np.ndarray:
    """
    Applies a permutation to a numpy array.
    :param permutation:
    :param array_to_permute:
    :return:
    """
    permutation_inv = invert_permutation(permutation_tuple=permutation)

    return array_to_permute[:, permutation_inv]

def apply_permutation_to_list(permutation: Tuple[pyd.conint(ge=0), ...],
                              list_to_permute: Union[list,tuple,np.ndarray]) -> Union[list,tuple,np.ndarray]:
    """
    Applies a permutation to a list.
    :param permutation:
    :param list_to_permute:
    :return:
    """

    if isinstance(list_to_permute, np.ndarray):
        return apply_permutation_to_array(permutation=permutation,
                                          array_to_permute=list_to_permute)

    permuted_indices = [permutation[idx] for idx in range(len(permutation))]
    permuted_list = [list_to_permute[idx_mapped] for idx_mapped in permuted_indices]

    if isinstance(list_to_permute, tuple):
        return tuple(permuted_list)
    elif isinstance(list_to_permute, list):
        return permuted_list

def multiply_permutations(permutations_to_multiply: Tuple[Tuple[pyd.conint(ge=0), ...]]) -> Tuple[
    pyd.conint(ge=0), ...]:
    number_of_nodes = len(permutations_to_multiply[0])
    final_permutation = list(range(number_of_nodes))
    for permutation in permutations_to_multiply:
        final_permutation = [permutation[i] for i in final_permutation]

    return tuple(final_permutation)


def apply_bitflip_to_bitstrings_array(bitflip_tuple: Tuple[pyd.conint(ge=0, le=1), ...],
                            bitstrings_array: np.ndarray) -> np.ndarray:
    """

    :param bitflip_tuple:
    :param array_to_flip:
    :return:
    """
    def __apply_xor(bts):
        return np.bitwise_xor(bts,
                              bitflip_tuple)

    flipped_array = np.apply_along_axis(__apply_xor, 1, bitstrings_array)
    return flipped_array.astype(int)


def apply_bitflip_to_list(bitflip_tuple: Tuple[pyd.conint(ge=0, le=1), ...],
                          list_of_bits: Union[list, tuple, np.ndarray]) -> Union[list, tuple, np.ndarray]:
    """
    Applies a bitflip transformation to a list.
    :param bitflip_tuple:
    :param list_to_flip:
    :return:
    """

    if isinstance(list_of_bits, np.ndarray):
        return apply_bitflip_to_bitstrings_array(bitflip_tuple=bitflip_tuple,
                                      array_to_flip=list_of_bits)

    flipped_list = [int(not bit) if flip else bit for bit, flip in zip(list_of_bits, bitflip_tuple)]
    if isinstance(list_of_bits, tuple):
        return tuple(flipped_list)
    elif isinstance(list_of_bits, list):
        return flipped_list



def apply_permutation_to_hamiltonian(hamiltonian: Union[networkit_Graph, List[Tuple[Union[float,int], Tuple[int, ...]]]],
                                     permutation_tuple: Tuple[pyd.conint(ge=0), ...]) \
        -> Union[networkit_Graph, List[Tuple[Union[float,int], Tuple[int, ...]]]]:
    """
    Applies a permutation transformation to the Hamiltonian graph.
    #permutation transforms the weights as folllows

    W_{permutation_tuple[i], permutation_tuple[j]} = W_{i, j}
    or
    W_{i, j} = W_{permutation_tuple^{-1}[i], permutation_tuple^{-1}[j]}

    :param hamiltonian:
    :param permutation_tuple:
    :return:
    """

    if isinstance(hamiltonian, networkit_Graph):
        assert set(permutation_tuple) == set(range(
            hamiltonian.numberOfNodes())), "Invalid permutation tuple, it should be a tuple of all integers from 0 to n-1"

        # copy the graph
        number_of_nodes = hamiltonian.numberOfNodes()

        transformed_graph = nk.Graph(number_of_nodes,
                                     weighted=True)

        for edge, original_weight in hamiltonian.iterEdgesWeights():
            node_i, node_j = edge
            node_i_permuted = permutation_tuple[node_i]
            node_j_permuted = permutation_tuple[node_j]
            transformed_graph.addEdge(node_i_permuted,
                                      node_j_permuted,
                                      original_weight)

        return transformed_graph
    elif isinstance(hamiltonian, list):
        transformed_hamiltonian = []
        for interaction in hamiltonian:
            coefficient, node_ids = interaction
            permuted_node_ids = tuple([permutation_tuple[qi] for qi in node_ids])
            transformed_hamiltonian.append((coefficient, permuted_node_ids))

        return transformed_hamiltonian
    else:
        raise ValueError("Hamiltonian must be a networkit graph or a list of interactions")

def concatenate_hamiltonian_transformations(
                              transformations_list: List[HamiltonianTransformation]):
    if isinstance(transformations_list[0], HamiltonianTransformation):
        transformations_list = [transformations_list]
    number_of_qubits = len(transformations_list[0].value)

    bitflip_combined = tuple([0] * number_of_qubits)
    permutation_combined = tuple(range(number_of_qubits))
    for transformation_full in transformations_list:
        transformation, value = transformation_full.transformation, transformation_full.value
        if isinstance(transformation, SNV.Bitflip):
            value_permuted = apply_permutation_to_list(permutation=invert_permutation(permutation_combined),
                                                       list_to_permute=bitflip_combined)
            bitflip_combined = apply_bitflip_to_list(bitflip_tuple=value,
                                                     list_of_bits=value_permuted)
        elif isinstance(transformation, SNV.Permutation):
            permutation_combined = multiply_permutations(permutations_to_multiply=(permutation_combined, value))
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
    return bitflip_combined, permutation_combined


def create_mapping(ham_prob,
                   all_pairs_ansatz):

    couplings_problem = ham_prob.couplings
    system_size_problem = ham_prob.number_of_qubits

    #qubits_problem = list(range(system_size_problem))
    mapping_to_pairs = {i:all_pairs_ansatz[i] for i in range(system_size_problem)}

    hamiltonian_ansatz = []
    maps_ordering = {}
    for i in range(system_size_problem):
        pair_i = mapping_to_pairs[i]
        for j in range(i+1,system_size_problem):
            pair_j = mapping_to_pairs[j]
            coeff_problem = couplings_problem[i][j]

            #we check if they overlap first
            overlap = list(set(pair_i).intersection(set(pair_j)))
            original_order = list(pair_i) + list(pair_j)
            if len(overlap) != 0:
                if len(overlap) != 1:
                    raise ValueError("Overlap is not 1, it is: {}".format(overlap))
                original_order = [x for x in original_order if x not in overlap]

            measured_order = tuple(sorted(original_order))
            original_order = tuple(original_order)
            if measured_order not in maps_ordering:
                maps_ordering[measured_order] = []

            maps_ordering[measured_order].append((float(coeff_problem), original_order))

            hamiltonian_ansatz.append((float(coeff_problem), original_order))
    return hamiltonian_ansatz, maps_ordering
