
AVAILABLE_SIMULATORS_QISKIT = []

try:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as SamplerAer
    from qiskit import QuantumCircuit

    _backend_aer_gpu = AerSimulator(device='GPU',
                                    method='statevector')

    _backend_aer_cpu = AerSimulator(device='CPU',
                                    method='statevector')
    _sampler_aer_gpu = SamplerAer.from_backend(backend=_backend_aer_gpu)
    _sampler_aer_cpu = SamplerAer.from_backend(backend=_backend_aer_cpu)
    _circ = QuantumCircuit(1, 1)
    _circ.x(0)
    _circ.measure(0, 0)
    try:
        _job_gpu = _sampler_aer_gpu.run([_circ],
                                        shots=10)
        _res_gpu = _job_gpu.result()
        AVAILABLE_SIMULATORS_QISKIT += ['aer-gpu']

    except(Exception) as e:
        pass


    try:
        _job_cpu = _sampler_aer_cpu.run([_circ],
                                        shots=10)
        _res_cpu = _job_cpu.result()

        AVAILABLE_SIMULATORS_QISKIT += ['aer-cpu']
    except(Exception) as e:
        pass


except(ImportError, ModuleNotFoundError):
    pass
