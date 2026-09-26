
AVAILABLE_SIMULATORS_QISKIT = []


try:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    _available_devices_qiskit = [x.lower() for x in AerSimulator().available_devices()]
    if 'cpu' in _available_devices_qiskit:
        AVAILABLE_SIMULATORS_QISKIT += ['aer-cpu']
    if 'gpu' in _available_devices_qiskit:
        AVAILABLE_SIMULATORS_QISKIT += ['aer-gpu']

except(ImportError, ModuleNotFoundError):
    pass



except(ImportError, ModuleNotFoundError):
    pass
