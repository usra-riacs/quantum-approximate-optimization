# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt import AVAILABLE_SIMULATORS
from quapopt import ancillary_functions as anf
from typing import List, Optional, Dict, Tuple
from tqdm.notebook import tqdm
import numpy as np
from quapopt.optimization.QAOA.circuits.time_block_ansatz import divide_hamiltonian_into_batches, TimeBlockBatchingType
from quapopt.optimization.QAOA.simulation.direct.cython_implementation.cython_qaoa_statevector_simulator import apply_full_qaoa_circuit_cython, apply_full_qaoa_circuit_cython_WS

import torch



def get_exp_X_operator_1q(angle:torch.Tensor,
                          device: torch.device):
    """

    :param angle:
    :param backend:
    :return:
    """
    cos_beta = torch.cos(angle)
    sin_beta = -1j*torch.sin(angle)

    return torch.tensor([[cos_beta, sin_beta],
                       [sin_beta, cos_beta]],
                        dtype=torch.complex64,
                        device=device
                      )

def get_WS_mixer_operator_1q(angle:torch.Tensor,
                             term_Z:torch.Tensor,
                             term_X:torch.Tensor,device: torch.device):
    """

    :param angle:
    :param backend:
    :return:
    """

    cos_beta = torch.cos(angle)
    sin_beta = -1j*torch.sin(angle)

    # term_Z = (1 - 2 * bias_parameter)
    # term_X = torch.sqrt((1 - bias_parameter) * bias_parameter)

    term_off = 2*sin_beta*term_X
    term_diag = cos_beta+term_Z*sin_beta


    return torch.tensor([[term_diag, term_off],
                       [term_off, torch.conj(term_diag)]],
                        dtype=torch.complex64,
                        device=device
                      )



def get_mixer_operator_WS(angle_mixer:torch.Tensor,
                          XZ_terms: torch.Tensor,
                          number_of_qubits:int,
                          device:torch.device):
    """

    :param angle_mixer:
    :param number_of_qubits:
    :param backend:
    :return:
    """
    _identical_bias = False

    if XZ_terms.shape[0]==1:
        _identical_bias = True


    _1q_mixer = get_WS_mixer_operator_1q(angle=angle_mixer,
                                         term_X=XZ_terms[0][0],
                                         term_Z=XZ_terms[0][1],
                                         device=device)

    big_mixer = _1q_mixer.clone()
    for qubit_index in range(1,number_of_qubits):
        if _identical_bias:
            big_mixer = torch.kron(big_mixer,
                                    _1q_mixer)
        else:
            big_mixer = torch.kron(big_mixer,
                                  get_WS_mixer_operator_1q(angle=angle_mixer,
                                                           term_X=XZ_terms[qubit_index][0],
                                                           term_Z=XZ_terms[qubit_index][1],
                                                           device=device))

    return big_mixer




def get_mixer_operator(angle_mixer:torch.Tensor,
                        number_of_qubits,
                       device:torch.device,
                       ):
    """

    :param angle_mixer:
    :param number_of_qubits:
    :param backend:
    :return:
    """


    _1q_mixer = get_exp_X_operator_1q(angle_mixer, device=device)

    big_mixer = _1q_mixer.clone()
    for _ in range(1, number_of_qubits):
        big_mixer = torch.kron(big_mixer, _1q_mixer)

    return big_mixer


def multiply_by_mixer_operator(angle_mixer:torch.Tensor,
                               number_of_qubits,
                               input_state:torch.Tensor):
    """
    Apply mixer operator to statevector without storing full matrix.

    Exploits tensor product structure: U_mixer = exp(-i*angle*X)^⊗n
    Uses torch.tensordot for efficient GPU-accelerated computation.

    :param angle_mixer: Mixer angle β
    :param number_of_qubits: Number of qubits
    :param input_state: Input statevector of shape (2^n,)
    :return: Statevector after mixer application
    """

    cos_beta, sin_beta = torch.cos(angle_mixer), torch.sin(angle_mixer)

    # Build single-qubit mixer matrix: exp(-i*β*X) = [[cos(β), -i*sin(β)], [-i*sin(β), cos(β)]]
    mixer_matrix = torch.tensor([[cos_beta, -1j * sin_beta],
                                 [-1j * sin_beta, cos_beta]],
                                dtype=torch.complex64,
                                device=input_state.device)

    # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    state = input_state.reshape([2] * number_of_qubits)

    # Apply single-qubit mixer to each qubit using tensordot
    for qubit_idx in range(number_of_qubits):
        # Contract mixer_matrix with state along qubit_idx dimension
        # dims=([1], [qubit_idx]): contract 2nd index of mixer (column) with qubit_idx of state
        state = torch.tensordot(mixer_matrix, state, dims=([1], [qubit_idx]))

        # Move the resulting dimension (now at position 0) back to qubit_idx
        state = torch.moveaxis(state, 0, qubit_idx)

    # Flatten back to vector form
    return state.reshape(2**number_of_qubits)

    # ==================== OLD IMPLEMENTATION (COMMENTED OUT) ====================
    # cos_beta, sin_beta = torch.cos(angle_mixer), torch.sin(angle_mixer)
    #
    # # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    # state = input_state.reshape([2] * number_of_qubits)
    #
    # # Apply single-qubit mixer to each qubit index
    # for qubit_idx in range(number_of_qubits):
    #     # Move qubit_idx axis to position 0
    #     state = torch.moveaxis(state, qubit_idx, 0)
    #
    #     # Apply exp(-i*β*X) = [[cos(β), -i*sin(β)], [-i*sin(β), cos(β)]]
    #     state[0], state[1] = (cos_beta * state[0] - 1j * sin_beta * state[1],
    #                           -1j * sin_beta * state[0] + cos_beta * state[1])
    #
    #     # Move axis back
    #     state = torch.moveaxis(state, 0, qubit_idx)
    #
    # # Flatten back to vector form
    # return state.reshape(2**number_of_qubits)
    # ============================================================================



def multiply_by_mixer_operator_WS(angle_mixer:torch.Tensor,
                                  number_of_qubits,
                                  input_state:torch.Tensor,
                                  XZ_terms:torch.Tensor):
    """
    Apply warm-started mixer operator to statevector without storing full matrix.

    Exploits tensor product structure: U_mixer = exp(-i*β*(X_term*X + Z_term*Z))^⊗n
    Uses torch.tensordot for efficient GPU-accelerated computation.

    :param angle_mixer: Mixer angle β
    :param number_of_qubits: Number of qubits
    :param input_state: Input statevector of shape (2^n,)
    :param XZ_terms: Tensor of shape (1, 2) for identical bias or (n_qubits, 2) for per-qubit bias
                     Each row is [term_X, term_Z] where term_X = 2*sqrt(c*(1-c)), term_Z = 1-2*c
    :return: Statevector after mixer application
    """

    cos_beta, sin_beta = torch.cos(angle_mixer), torch.sin(angle_mixer)

    # Check if bias is identical for all qubits
    identical_bias = XZ_terms.shape[0] == 1

    # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    state = input_state.reshape([2] * number_of_qubits)

    # Precompute mixer matrix for identical bias case
    if identical_bias:
        term_X, term_Z = XZ_terms[0]
        # WS mixer matrix: [[cos(β) - i*sin(β)*Z, -i*sin(β)*term_X],
        #                   [-i*sin(β)*term_X,     cos(β) + i*sin(β)*Z]]
        mixer_matrix = torch.tensor([[cos_beta - 1j * sin_beta * term_Z, -1j * sin_beta * term_X],
                                     [-1j * sin_beta * term_X, cos_beta + 1j * sin_beta * term_Z]],
                                    dtype=torch.complex64,
                                    device=input_state.device)

    # Apply single-qubit WS mixer to each qubit using tensordot
    for qubit_idx in range(number_of_qubits):
        # Build mixer matrix for this qubit (if per-qubit bias)
        if not identical_bias:
            term_X, term_Z = XZ_terms[qubit_idx]
            mixer_matrix = torch.tensor([[cos_beta - 1j * sin_beta * term_Z, -1j * sin_beta * term_X],
                                         [-1j * sin_beta * term_X, cos_beta + 1j * sin_beta * term_Z]],
                                        dtype=torch.complex64,
                                        device=input_state.device)

        # Contract mixer_matrix with state along qubit_idx dimension
        state = torch.tensordot(mixer_matrix, state, dims=([1], [qubit_idx]))

        # Move the resulting dimension (now at position 0) back to qubit_idx
        state = torch.moveaxis(state, 0, qubit_idx)

    # Flatten back to vector form
    return state.reshape(2**number_of_qubits)

    # ==================== OLD IMPLEMENTATION (COMMENTED OUT) ====================
    # cos_beta, sin_beta = torch.cos(angle_mixer), torch.sin(angle_mixer)
    #
    # # Check if bias is identical for all qubits (single tuple) or per-qubit (list of tuples)
    # identical_bias = XZ_terms.shape[0]==1
    #
    # # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    # state = input_state.reshape([2] * number_of_qubits)
    #
    # # Apply single-qubit mixer to each qubit index
    # for qubit_idx in range(number_of_qubits):
    #     # Move qubit_idx axis to position 0
    #     state = torch.moveaxis(state, qubit_idx, 0)
    #
    #     # Get XZ terms for this qubit
    #     if identical_bias:
    #         term_X, term_Z = XZ_terms[0]  # Same for all qubits
    #     else:
    #         term_X, term_Z = XZ_terms[qubit_idx]  # Per-qubit
    #
    #     a, b = state[0], state[1]
    #
    #     # Apply WS mixer: exp(-i*β*(term_X*X + term_Z*Z))
    #     # Matrix: [[cos(β) - i*sin(β)*Z,   -i*sin(β)*2*X],
    #     #          [-i*sin(β)*2*X,          cos(β) + i*sin(β)*Z]]
    #     state[0], state[1] = (a*cos_beta - 1j*sin_beta*(term_Z*a + b*term_X),
    #                           b*cos_beta - 1j*sin_beta*(-term_Z*b + a*term_X))
    #
    #     # Move axis back
    #     state = torch.moveaxis(state, 0, qubit_idx)
    #
    # # Flatten back to vector form
    # return state.reshape(2**number_of_qubits)
    # ============================================================================





class QAOASimulatorPytorch:
    """
    Basic QAOA simulator. It is not optimized, the main purpose is to provide a reference implementation.

    :param hamiltonian_phase: Phase Hamiltonian to be implemented
    :param time_block_size: Number of linear chains per QAOA layer
    :param time_block_seed: Seed for shuffling the Hamiltonian terms
    :param time_block_partition: Dictionary specifying the partition of the Hamiltonian terms into time blocks.


    """
    def __init__(self,
                 hamiltonian_phase:ClassicalHamiltonian,
                 time_block_size:Optional[float]=None,
                 time_block_seed:Optional[int]=-1,
                 time_block_batching_type:TimeBlockBatchingType=TimeBlockBatchingType.FRACTIONAL,
                 time_block_partition:Optional[Dict[int,ClassicalHamiltonian]]=None,
                 device: Optional[torch.device|str] = None,
                 input_state_always_the_same = True
                 ):

        #we want to precompute spectrum
        self._number_of_qubits = hamiltonian_phase.number_of_qubits
        self._dimension = torch.tensor(2 ** self._number_of_qubits)



        if device in [None, 'auto']:
            if self._number_of_qubits<=16:
                device = torch.device("cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device,str):
            device = torch.device(device)

        self._bck:torch.device = device
        self._hamiltonian_phase = hamiltonian_phase

        if hamiltonian_phase.spectrum is None:
            hamiltonian_phase.solve_hamiltonian()

        self._hamiltonian_spectrum = torch.tensor(self._hamiltonian_phase.spectrum,dtype=torch.float64,device=self._bck)

        if time_block_partition is None:
            time_block_partition = divide_hamiltonian_into_batches(hamiltonian=hamiltonian_phase,
                                                                   time_block_size=time_block_size,
                                                                   batching_type=time_block_batching_type,
                                                                   time_block_seed=time_block_seed)

        self._time_block_partition = time_block_partition




        self._batches_spectra:List[torch.Tensor] = [None]*len(time_block_partition)


        self.input_state_always_the_same = input_state_always_the_same

        self.fixed_input_state = None
        # The bias the cached initial state was constructed from (see _get_cached_input_state).
        self._fixed_input_state_bias_key = None


    @property
    def hamiltonian_phase(self):
        return self._hamiltonian_phase


    @property
    def batches_spectra(self):
        return self._batches_spectra

    def update_batches_spectra(self,
                               spectrum: torch.Tensor,
                               index:int):

        self._batches_spectra[index] = spectrum


    def solve_hamiltonian(self,
                          hamiltonian:ClassicalHamiltonian,
                          solving_backend:str=None):
        if solving_backend is None:
            if 'cuda' in AVAILABLE_SIMULATORS:
                solving_backend = 'cuda'
            else:
                solving_backend = 'python'

        #anf.cool_print("SOLVING HAMILTONIAN", '...','green')

        if solving_backend == 'cuda':
            spectrum = anf.cuda_solve_hamiltonian(hamiltonian)
        else:
            spectrum = anf.solve_hamiltonian_python(hamiltonian)

        return spectrum

    def _update_spectra(self,
                        depth:int,
                        solving_backend:str=None
                        ):

        number_of_batches = len(self._time_block_partition)

        how_many_spectra = min([number_of_batches, depth])
        for batch_index in range(how_many_spectra):
            if self._batches_spectra[batch_index] is None:
                spectrum = self.solve_hamiltonian(hamiltonian=self._time_block_partition[batch_index],
                                                  solving_backend=solving_backend)

                spectrum = torch.tensor(spectrum,dtype=torch.complex64,device=self._bck)

                assert spectrum.shape[0] == 2**self._number_of_qubits

                self.update_batches_spectra(spectrum=spectrum,
                                            index=batch_index)

    def get_standard_input_state(self,
                                 bias_parameteres_WS:Optional[torch.Tensor]=None):


        if bias_parameteres_WS is None:
            input_state = torch.ones(self._dimension, dtype=torch.complex64,device=self._bck)/torch.sqrt(self._dimension)
        else:
            _identical_bias = False
            if bias_parameteres_WS.shape[0]==1:
                _identical_bias = True

            if _identical_bias:
                sqrt_c = torch.sqrt(bias_parameteres_WS[0])
                sqrt_1c = torch.sqrt(1-bias_parameteres_WS[0])

                _1q_state = torch.tensor([sqrt_1c,sqrt_c],dtype=torch.complex64,device=self._bck)

                input_state = _1q_state.clone()
                for _ in range(self._number_of_qubits-1):
                    input_state = torch.kron(input_state,_1q_state)

            else:
                sqrt_c = torch.sqrt(bias_parameteres_WS)
                sqrt_1c = torch.sqrt(1 - bias_parameteres_WS)

                # Qubit i gets its own bias; the product runs from qubit 0 (leftmost) to qubit n-1.
                input_state = torch.tensor([sqrt_1c[0], sqrt_c[0]], dtype=torch.complex64, device=self._bck)

                for i in range(1, self._number_of_qubits):
                    input_state = torch.kron(input_state,
                                             torch.tensor([sqrt_1c[i], sqrt_c[i]],
                                                          dtype=torch.complex64, device=self._bck))

        return input_state

    @staticmethod
    def _bias_cache_key(bias_parameteres_WS: Optional[torch.Tensor]):
        """Hashable identity of a warm-start bias: None or a tuple of floats."""
        if bias_parameteres_WS is None:
            return None
        return tuple(float(x) for x in bias_parameteres_WS.detach().cpu().reshape(-1).tolist())

    def _get_cached_input_state(self, bias_parameteres_WS: Optional[torch.Tensor] = None):
        """The standard initial state for this bias, cached across calls.

        The cache is valid only for the bias it was constructed from: a call with a
        different bias replaces it. Callers clone the returned tensor before modifying it.
        """
        bias_key = self._bias_cache_key(bias_parameteres_WS)
        if self.fixed_input_state is None or bias_key != self._fixed_input_state_bias_key:
            self.fixed_input_state = self.get_standard_input_state(bias_parameteres_WS=bias_parameteres_WS)
            self._fixed_input_state_bias_key = bias_key
        return self.fixed_input_state


    def _get_qaoa_statevector_vanilla(self,
                             angles_PS:torch.Tensor,
                             angles_mixer:torch.Tensor,
                             input_state:Optional[torch.Tensor]=None,
                             show_progress_bar:bool=False):


        if input_state is None:
            if self.input_state_always_the_same:
                input_state = self._get_cached_input_state().clone()
            else:
                input_state = self.get_standard_input_state()
        else:
            input_state = input_state.clone()

        number_of_batches = len(self._time_block_partition)

        # Layer-by-layer execution for other backends
        for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                         disable = not show_progress_bar):
            #batches reset after number_of_batches, so
            batch_index = layer_index % number_of_batches

            #Phase separation
            input_state = input_state * torch.exp(-1j * angle_PS * self._batches_spectra[batch_index])




            #Mixer
            input_state = multiply_by_mixer_operator(angle_mixer=angle_mixer,
                                                    number_of_qubits=self._number_of_qubits,
                                                    input_state=input_state)


        return input_state

    def _get_qaoa_statevector_WS(self,
                             angles_PS: torch.Tensor,
                             angles_mixer: torch.Tensor,
                             bias_parameteres_WS: torch.Tensor,
                             input_state: Optional[torch.Tensor] = None,
                             show_progress_bar: bool = False):


        _identical_bias = False
        if bias_parameteres_WS.shape[0]==1:
            _identical_bias = True


        bias_parameteres_WS = torch.as_tensor(bias_parameteres_WS, dtype=torch.float64, device=self._bck)
        if input_state is None:
            if self.input_state_always_the_same:
                input_state = self._get_cached_input_state(bias_parameteres_WS=bias_parameteres_WS).clone()
            else:
                input_state = self.get_standard_input_state(bias_parameteres_WS=bias_parameteres_WS)
        else:
            input_state = input_state.clone()


        number_of_batches = len(self._time_block_partition)

        if _identical_bias:
            c = bias_parameteres_WS[0]
            XZ_terms = torch.tensor([[2*torch.sqrt(c*(1-c)), 1-2*c]], dtype=torch.float64, device=self._bck)
        else:
            Z_terms = 1-2*bias_parameteres_WS
            X_terms = 2*torch.sqrt(bias_parameteres_WS*(1-bias_parameteres_WS))
            XZ_terms = torch.tensor([[x,z] for x,z in zip(X_terms,Z_terms)], dtype=torch.float64, device=self._bck)

        # Layer-by-layer execution for other backends
        for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                         disable=not show_progress_bar):
            # batches reset after number_of_batches, so
            batch_index = layer_index % number_of_batches

            # Phase separation
            input_state = input_state * torch.exp(-1j * angle_PS * self._batches_spectra[batch_index])


            # Mixer
            input_state = multiply_by_mixer_operator_WS(angle_mixer=angle_mixer,
                                                         number_of_qubits=self._number_of_qubits,
                                                         input_state=input_state,
                                                         XZ_terms=XZ_terms)


        return input_state


    def get_exp_value(self,
                      quantum_state:torch.Tensor,
                      ):

        prob_distro = quantum_state.abs()**2
        exp_value = torch.sum(prob_distro*self._hamiltonian_spectrum)

        return exp_value

    def get_exp_value_estimator(self,
                                quantum_state:torch.Tensor,
                                 number_of_samples:int=10000,
                                 ):

        prob_distro = quantum_state.abs()**2

        samples = torch.multinomial(prob_distro,number_of_samples,replacement=True)

        exp_value = torch.sum(self._hamiltonian_spectrum[samples])/number_of_samples

        return exp_value


    def get_qaoa_statevector(self,
                             angles_PS: torch.Tensor,
                             angles_mixer: torch.Tensor,
                             bias_parameters_WS: Optional[float | List[float] | torch.Tensor] = None,
                             input_state: Optional[torch.Tensor] = None,
                             show_progress_bar: bool = False)->torch.Tensor:


        if isinstance(bias_parameters_WS, float):
            bias_parameters_WS = torch.tensor([bias_parameters_WS], dtype=torch.float64, device=self._bck)
        elif isinstance(bias_parameters_WS, list):
            bias_parameters_WS = torch.tensor(bias_parameters_WS, dtype=torch.float64, device=self._bck)



        self._update_spectra(depth=len(angles_PS))

        if input_state is None and self.input_state_always_the_same:
            input_state = self._get_cached_input_state(bias_parameteres_WS=bias_parameters_WS)


        if bias_parameters_WS is None:
            return self._get_qaoa_statevector_vanilla(angles_PS=angles_PS,
                                                      angles_mixer=angles_mixer,
                                                      input_state=input_state,
                                                      show_progress_bar=show_progress_bar)
        else:
            return self._get_qaoa_statevector_WS(angles_PS=angles_PS,
                                                 angles_mixer=angles_mixer,
                                                 bias_parameteres_WS=bias_parameters_WS,
                                                 input_state=input_state,
                                                 show_progress_bar=show_progress_bar)


    def optimize_with_adam(self,
                           depth:int,
                           number_of_samples:Optional[int]=None,
                           initial_angles_PS: Optional[np.ndarray | List[float]]=None,
                           initial_angles_mixer: Optional[np.ndarray | List[float]]=None,
                           seed_angles:Optional[int]=0,
                           bias_parameters_WS: Optional[float | List[float]] = None,
                           learning_rate: float = 0.01,
                           max_iter: int = 10000,
                           tolerance: float = 1e-6,
                           show_progress_bar: bool = True,
                           verbosity=0) -> Tuple[Tuple,Dict]:
        """
        Optimize QAOA angles using Adam optimizer with automatic differentiation.

        :param initial_angles_PS: Initial phase separation angles (depth,)
        :param initial_angles_mixer: Initial mixer angles (depth,)
        :param bias_parameters_WS: Optional warm-start bias parameters
        :param learning_rate: Adam learning rate (default: 0.01)
        :param max_iter: Maximum number of optimization iterations (default: 100)
        :param tolerance: Convergence tolerance for relative change in loss (default: 1e-6)
        :param show_progress_bar: Whether to show progress bar (default: True)
        :return: Dictionary containing:
            - 'angles_PS': Optimized phase separation angles (numpy array)
            - 'angles_mixer': Optimized mixer angles (numpy array)
            - 'final_loss': Final expectation value
            - 'loss_history': List of losses at each iteration
            - 'num_iterations': Number of iterations performed
        """

        numpy_rng = np.random.default_rng(seed_angles)
        if initial_angles_PS is None:
            initial_angles_PS = numpy_rng.uniform(low=-np.pi, high=np.pi, size=(depth,))

        if initial_angles_mixer is None:
            initial_angles_mixer = numpy_rng.uniform(low=-np.pi, high=np.pi, size=(depth,))


        _infinite_samples = number_of_samples in [np.inf,None]



        # Convert initial angles to torch tensors with gradients enabled
        angles_PS = torch.tensor(initial_angles_PS, dtype=torch.float32,
                                device=self._bck, requires_grad=_infinite_samples)
        angles_mixer = torch.tensor(initial_angles_mixer, dtype=torch.float32,
                                   device=self._bck, requires_grad=_infinite_samples)

        # Setup Adam optimizer
        optimizer = torch.optim.Adam([angles_PS, angles_mixer], lr=learning_rate)

        # Optimization history
        loss_history = []

        # Progress bar
        iterator = tqdm(range(max_iter), disable=not show_progress_bar, desc="Adam Optimization")

        best_loss, best_angles = float('inf'), None
        for iteration in iterator:


            # Compute statevector (forward pass)
            statevector = self.get_qaoa_statevector(angles_PS=angles_PS,
                                                    angles_mixer=angles_mixer,
                                                    bias_parameters_WS=bias_parameters_WS,
                                                    input_state=None,
                                                    show_progress_bar=False)

            # Compute loss (expectation value - we want to minimize it)
            if _infinite_samples:
                loss = self.get_exp_value(statevector)
            else:
                loss = self.get_exp_value_estimator(statevector, number_of_samples=number_of_samples)


            if loss<best_loss:
                best_loss = loss
                best_angles = (angles_PS.detach().cpu().numpy(),angles_mixer.detach().cpu().numpy())
                if verbosity >= 2:
                    print(f"\nIteration {iteration + 1} loss: {loss:.6f}")


            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Track history
            loss_value = loss.item()
            loss_history.append(loss_value)

            # Zero gradients
            optimizer.zero_grad()

            # Update progress bar
            if show_progress_bar:
                iterator.set_postfix({'loss': f'{loss_value:.6f}'})

            # Check convergence
            if iteration > 0:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / (abs(loss_history[-2]) + 1e-10)
                if rel_change < tolerance:
                    if verbosity>=1:
                        print(f"\nConverged after {iteration + 1} iterations (rel_change={rel_change:.2e})")
                    break

        # Return results
        return (best_loss.detach().cpu().numpy(), best_angles), {
            'angles_PS': angles_PS.detach().cpu().numpy(),
            'angles_mixer': angles_mixer.detach().cpu().numpy(),
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'num_iterations': len(loss_history)
        }