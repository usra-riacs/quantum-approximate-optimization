# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
from qiskit.converters import circuit_to_dag, dag_to_circuit
from typing import List, Optional, Dict
from enum import Enum
import numpy as np
from qiskit import QuantumCircuit
from quapopt.data_analysis.data_handling import (COEFFICIENTS_TYPE,
                                                 COEFFICIENTS_DISTRIBUTION,
                                                 CoefficientsDistributionSpecifier,
                                                 HAMILTONIAN_MODELS)
from quapopt.circuits import backend_utilities as bck_utils
class DelayScheduleType(Enum):
    NONE = "NONE"
    LINEAR = "LINEAR"
    RANDOM = "RANDOM"

class DelaySchedulerBase:
    """
    Base class for delay schedulers.
    """
    def __init__(self,
                 delay_type:DelayScheduleType=DelayScheduleType.NONE,
                 delay_unit:str='ns',
                 delay_at_the_end:Optional[float]=None,
                 add_barriers_to_layers:bool=False):

        self.delay_type = delay_type
        self.delay_unit = delay_unit

        if delay_at_the_end is None:
            delay_at_the_end = 0.0

        self.delay_at_the_end = delay_at_the_end
        self.add_barriers_to_layers = add_barriers_to_layers

    def get_string_description_base(self):
        return f"DelayType={self.delay_type.value};DelayUnit={self.delay_unit};DelayATE={self.delay_at_the_end:.1f};Barriers={self.add_barriers_to_layers}"

    def generate_layer_delays(self,instruction_indices:List[int]):
        raise NotImplementedError("This method should be implemented in subclasses.")


    def get_string_description(self):
        raise NotImplementedError("This method should be implemented in subclasses.")



    def __repr__(self):
        return self.get_string_description()


class DelaySchedulerNone(DelaySchedulerBase):
    """
    Delay scheduler that does not add any delays.
    """


    def __init__(self):
        super().__init__(delay_type=DelayScheduleType.NONE,
                         delay_unit='ns',
                         delay_at_the_end=None,
                         add_barriers_to_layers=False)

    def generate_delays(self, number_of_layers:int):
        return [0.0] * number_of_layers


    def get_string_description(self):
        return self.get_string_description_base()





class DelaySchedulerLinear(DelaySchedulerBase):
    """
    Linear delay scheduler that adds delays that increase linearly with the index of instruction in the layer
    """


    def __init__(self,
                 delay_base:float=0.01,
                 indices_ordering:Optional[Dict[int,int]]=None,
                 delay_at_the_end:float=0.0,
                 add_barriers_to_layers:bool=False):
        """

        :param delay_base: Base delay value in nanoseconds. The delay for the i-th instruction will be `delay_base * i`.
        :param indices_ordering: optional dictionary that specifies the ordering of indices for the delays;
         it replaces "i" with `indices_ordering[i]` in the delay calculation.
        """


        super().__init__(delay_type=DelayScheduleType.LINEAR,
                         delay_unit='ns',
                         delay_at_the_end=delay_at_the_end,
                         add_barriers_to_layers=add_barriers_to_layers)
        self._delay_base = delay_base
        self._indices_ordering = indices_ordering

    @property
    def delay_base(self):
        return self._delay_base
    @delay_base.setter
    def delay_base(self, value:float):
        if value < 0:
            raise ValueError("Delay base must be non-negative.")
        self._delay_base = value


    def generate_delays(self, number_of_layers:int):
        if self.delay_base == 0:
            return None

        if self._indices_ordering is None:
            return [self.delay_base * i for i in range(number_of_layers)]
        else:
            return [self.delay_base * self._indices_ordering[i] for i in range(number_of_layers)]


    def get_string_description(self):
        return f"{self.get_string_description_base()};DelayBase={self._delay_base:.4f}"

class DelaySchedulerRandom(DelaySchedulerBase):
    """
    Random delay scheduler that adds delays sampled from a specified distribution.
    """
    def __init__(self,
                 coefficients_distribution:COEFFICIENTS_DISTRIBUTION = COEFFICIENTS_DISTRIBUTION.Uniform,
                 coefficients_distribution_properties:Optional[dict] = None,
                 rng_seed:int=None,
                 delay_at_the_end:float=0.0,
                 add_barriers_to_layers:bool=False):
        """

        :param coefficients_distribution:
        Distribution from which to sample the delays.
        :param coefficients_distribution_properties:
        Properties of the distribution, such as mean and standard deviation for Normal distribution,
        see generate_delays method for what parameters are expected for each distribution.
        :param rng_seed:
        """
        super().__init__(delay_type=DelayScheduleType.RANDOM,
                         delay_unit='ns',
                         delay_at_the_end=delay_at_the_end,
                         add_barriers_to_layers=add_barriers_to_layers)

        if coefficients_distribution_properties is None:
            if coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Uniform:
                coefficients_distribution_properties = {'low':0, 'high':0.01}
            elif coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Normal:
                coefficients_distribution_properties = {'mean':0.01, 'std':0.01}
            elif coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Constant:
                coefficients_distribution_properties = {'value':0.01}
            else:
                raise ValueError(f"Unknown CoefficientsDistributionName: {coefficients_distribution}")

        self._rng_seed = rng_seed
        self._numpy_rng = np.random.default_rng(seed=rng_seed)
        self._coefficients_distribution = coefficients_distribution
        self._coefficients_distribution_properties = coefficients_distribution_properties
        self._distribution_description = CoefficientsDistributionSpecifier(
            CoefficientsType=COEFFICIENTS_TYPE.CONTINUOUS,
            CoefficientsDistributionName=self._coefficients_distribution,
            CoefficientsDistributionProperties=self._coefficients_distribution_properties).get_description_string()


    def generate_delays(self, number_of_layers:int):
        if self._coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Uniform:
            delays = self._numpy_rng.uniform(low=self._coefficients_distribution_properties['low'],
                                             high=self._coefficients_distribution_properties['high'],
                                             size=number_of_layers)
        elif self._coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Normal:
            delays = self._numpy_rng.normal(loc=self._coefficients_distribution_properties['mean'],
                                            scale=self._coefficients_distribution_properties['std'],
                                            size=number_of_layers)
        elif self._coefficients_distribution == COEFFICIENTS_DISTRIBUTION.Constant:
            delays = np.full(shape=number_of_layers,
                             fill_value=self._coefficients_distribution_properties['value'])
        else:
            raise ValueError(f"Unknown CoefficientsDistributionName: {self._coefficients_distribution}")
        # Ensure non-negative delays
        delays = np.maximum(delays, 0.0)
        return np.maximum(delays, 0.0)

    def get_string_description(self):
        return (f"{self.get_string_description_base()};"
                f"RNGSeed={self._rng_seed};"
                f"Dist={self._distribution_description}")



def add_delays_to_circuit_layers(quantum_circuit:QuantumCircuit,
                                 number_of_qubits:int,
                                 delay_scheduler:Optional[DelaySchedulerBase]=None,
                                 for_visualization:bool=False,
                                 ignore_add_barriers_flag:bool=False,
                                 ignore_delay_at_the_end:bool=False
                                 ):
    """
    Add delays to the layers of a quantum circuit.
    :param quantum_circuit:
    :param number_of_qubits: Number of nontrivial qubits in the circuit.
    :param delay_scheduler: Delay scheduler to use for generating delays.
    if None, a default DelaySchedulerLinear with a base delay of 0.01 us is used.
    :param for_visualization: whether circuit is generated only for visualization purposes.
    :param add_barriers: whether to add barriers after each layer.
    :return:
    """


    if delay_scheduler is None:
        if for_visualization:
            _delay_base = 1.0
        else:
            _delay_base = 0.01
        delay_scheduler = DelaySchedulerLinear(delay_base=_delay_base)

    #handle trivial cases
    if delay_scheduler.delay_type == DelayScheduleType.NONE:
        return quantum_circuit.copy()
    elif delay_scheduler.delay_type == DelayScheduleType.LINEAR:
        if delay_scheduler.delay_base == 0.0:
            return quantum_circuit.copy()


    dag_circuit = circuit_to_dag(quantum_circuit,copy_operations=True)
    quantum_circuit_delayed = QuantumCircuit(quantum_circuit.num_qubits,
                                             number_of_qubits)


    for circuit_layer in dag_circuit.layers():
        circuit_layer = dag_to_circuit(circuit_layer['graph'])
        circuit_layer_delayed = QuantumCircuit(quantum_circuit.num_qubits,
                                               number_of_qubits)
        delays_here = delay_scheduler.generate_delays(number_of_layers=len(circuit_layer.data))
        if delays_here is None:
            delays_here = [0.0] * len(circuit_layer.data)

        for idx_instr, (delay_length, instr) in enumerate(zip(delays_here, circuit_layer.data)):
            qubits_instr = instr.qubits
            circ_instr = QuantumCircuit.from_instructions(instructions=[instr],
                                                          qubits=qubits_instr)

            if delay_length != 0.0:
                if for_visualization and delay_scheduler.delay_type == DelayScheduleType.LINEAR:

                    #Linear schedule can be visualized nicely.
                    for qubit in qubits_instr:
                        for _ in range(idx_instr):
                            circuit_layer_delayed.delay(duration=delay_length/idx_instr,
                                                        qarg=qubit,
                                                        unit=delay_scheduler.delay_unit)
                else:
                    for qubit in qubits_instr:
                        circuit_layer_delayed.delay(duration=delay_length,
                                                    qarg=qubit,
                                                    unit=delay_scheduler.delay_unit)

            circuit_layer_delayed = circuit_layer_delayed.compose(circ_instr,
                                                                  inplace=False,
                                                                  qubits=qubits_instr)

        if not ignore_add_barriers_flag:
            if delay_scheduler.add_barriers_to_layers or for_visualization:
                circuit_layer_delayed.barrier(*circuit_layer_delayed.qubits)

        quantum_circuit_delayed = quantum_circuit_delayed.compose(circuit_layer_delayed,
                                                                  inplace=False,
                                                                  qubits=circuit_layer_delayed.qubits)


    if not ignore_delay_at_the_end:
        if delay_scheduler.delay_at_the_end > 0.0:
            nontrivial_qubits = bck_utils.get_nontrivial_physical_indices_from_circuit(quantum_circuit=quantum_circuit_delayed)
            for qubit in nontrivial_qubits:
                quantum_circuit_delayed.delay(duration=delay_scheduler.delay_at_the_end,
                                              qarg=qubit,
                                              unit=delay_scheduler.delay_unit)

    return quantum_circuit_delayed

