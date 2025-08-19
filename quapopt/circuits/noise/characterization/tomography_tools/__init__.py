# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 
 
import numpy as np
from enum import Enum


class TomographyType(Enum):
    QUANTUM_DETECTOR = 'DETECTOR'
    DIAGONAL_DETECTOR = 'DIAGONAL_DETECTOR'

    STATE = 'STATE'
    PROCESS = 'PROCESS'


class TomographyGatesType(Enum):
    PAULI = 'PAULI'


_PAULI_TOMOGRAPHY_ROTATIONS_NUMBERS = {TomographyType.QUANTUM_DETECTOR: {'RZ_rotations': 3,
                                                                  'RX_rotations': 2},
                                TomographyType.DIAGONAL_DETECTOR: {'RZ_rotations': 3,
                                                                   'RX_rotations': 2},
                                TomographyType.STATE: {'RZ_rotations': 2,
                                                       'RX_rotations': 2},
                                # appends quantum state tomography_tools and prepends process tomography_tools
                                TomographyType.PROCESS: {'RZ_rotations': (2, 3),
                                                         'RX_rotations': (2, 2)}

                                }
_PAULI_TOMOGRAPHY_COUNTS_STANDARD = {TomographyType.QUANTUM_DETECTOR:3,
                                    TomographyType.DIAGONAL_DETECTOR:2,
                                    TomographyType.STATE:3,
                                    TomographyType.PROCESS:(3,3)}


#Those are the RZ and RX angles to initialize the state to Pauli eigenstate
#Assuming input state is |0> and we apply the gates IMMEDIATELY AFTER STATE INITIALIZATION
_ANGLES_PAULI_EIGENSTATES = {
    ## Z eigenstates ##
    #|0> is 0
    0:{'RZ':(0.,0.,0.),
       'RX':(0.,0.)},
    #|1> is 1
    1:{'RZ':(-0.9199372448290238, np.pi, 2.2216554087607694),
       'RX':(np.pi / 2, -np.pi / 2)},

    ## X eigenstates ##
    #|+> is 2
    2:{'RZ':(np.pi / 2, np.pi / 2, 0),
       'RX':(np.pi / 2, 0)},
    #|-> is 3
    3:{'RZ':(np.pi / 2, -np.pi / 2, 0),
         'RX':(np.pi / 2, 0)},

    ## Y eigenstates ##
    #|i+> is 4
    4:{'RZ':(np.pi / 2, np.pi, 0),
       'RX':(np.pi / 2, 0)},

    #|i-> is 5
    5:{'RZ':(np.pi / 2, 0, 0),
       'RX':(np.pi / 2, 0)},
}

#Those are the RZ and RX angles to change mesaurement basis to Pauli basis
# Assuming default basis is Z and that we apply the gates TO THE STATE BEFORE MEASUREMENT
_ANGLES_PAULI_BASES = {

    #0 is Z basis (default measurement basis)
    0:{'RZ':(0.,0.,0.),
       'RX':(0.,0.)},

    #1 is X basis (measurement basis)
    1:{'RZ':(np.pi, np.pi / 2),
       'RX':(np.pi / 2, -np.pi / 2)},

    #2 is Y basis (measurement basis)
    2:{'RZ':(np.pi / 2, np.pi / 2),
       'RX':(np.pi / 2, -np.pi / 2)},


}





