import numpy as np
import pandas as pd
from functools import reduce
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.sparse
import operator

from qcircuit.gates import gates



class QuantumRegister(object):
    def __init__(self, n_qubits):
        """ 
        Create a quantum register by specifying the number of qubits.
        Note that the state is initialize the product state of |0> states
        
        e.g. if n_qubits = 5, it creates the states |00000>
        """
        
        try:
            size = int(n_qubits)
        except Exception:
            raise Exception("Register size must be castable to an int (%s '%s' was provided)"
                               % (type(size).__name__, size))
        if size <= 0:
            raise Exception("Register size must be positive (%s '%s' was provided)"
                               % (type(size).__name__, size))
            
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.amplitudes = np.zeros((self.n_states), dtype=np.complex128)
        self.amplitudes[0] = 1.0
        
        self.value = False
        
       
    """
    The general idea is that we will apply gates by,
        1. Initialize the quantum register as qr = QuantumRegister(2)
        2. Applying the gates to the qr.apply_some_gates
        3. The aplitudes are updated inplace, and they can be accessed as qr.amplitudes
    
    """
    def applySingleGate(self, gate, qubit_index, debug=False):
        """
        A function which takes as input the index of control and target qubits, and updates the amplituded
        of the state
        
        Args:
            gate: a string identifying the name of the gate to implement
            qubit_index: qubit's index
            debug: a boolean that prints the genrated matrix: useful for debugging gates.generateSingleGatesMatrix
        """
        # Generate the gate matrix
        gateMatrix = gates.generateSingleGatesMatrix(
            gate, self.n_qubits, qubit_index, sparse=False)
        if debug:
            print('Here is the gate \n ', gateMatrix)
        # Calculate the new state vector by multiplying by the gate
        self.amplitudes = np.array(np.matmul(self.amplitudes, gateMatrix))
        
        
        
    def applyClassicalControl(self, gate, control, target, classical_register, allGates):
        """
        A function which which applies a control from the classical register to the quatum one.
        
        Args:
            gate: a string identifying the name of the gate to implement
            control: integer representing the index of the control classical bit (note the index runs from 0 to n_qubits-1) 
            target: integer representing the index of the target qubit (note the index runs from 0 to n_qubits-1)
            classical_register: a copy of the classical register
            allGates: a boolean, when false it implements the classical control sequentially, when true it adds it to the allGates at once routine
            debug: a boolean that prints the genrated matrix: useful for debugging gates.generateSingleGatesMatrix
        """
        n_qubits = self.n_qubits
        n_bits = classical_register.n_bits

        # The +1 logic is needed because the registers index starts at zero, while the count of qubit is, unpythonically, standard
        if target + 1 > n_qubits:
            raise Exception("Target cannot be larger than total number of qubits (%s,  '%s' was provided) \
                            Remeber that quantum register is indexed starting from zero"
                               % (n_qubits, target))
            
        if control + 1 > n_bits:
            raise Exception("Target cannot be larger than total number of qubits (%s,  '%s' was provided) \
                             Remeber that the classical register is indexed starting from zero"
                               % (n_bits, target))
        
        
        control_value = classical_register.values[control]
              
        target += 1
        if control_value == 0:
            if allGates:
                return gates.generateSingleGatesMatrix('Id',
                                                       n_qubits,
                                                       target,
                                                       sparse=True)
            else:
                self.quantumRegister.applySingleGate('Id', target)
        elif control_value == 1:
            if allGates:
                return gates.generateSingleGatesMatrix(gate,
                                                       n_qubits,
                                                       target,
                                                       sparse=True)
            else:
                self.quantumRegister.applySingleGate(gate, target)
                

    def applyCnotGate(self, control_qubit, target_qubit):
        """
        A function which takes as input the index of control and target qubits, and updates the amplitudes
        of the quantum registers
        
        Args:
            control_qubit: integer representing the index of the control classical bit (note the index runs from 0 to n_qubits-1) 
            target: integer representing the index of the target qubit (note the index runs from 0 to n_qubits-1)
            target_qubit: a copy of the classical register        
        """
        gateMatrix = gates.generateCnotGate(self.n_qubits, control_qubit, target_qubit, sparse=False)
        self.amplitudes = np.array(np.matmul(self.amplitudes, gateMatrix))
        
        
    def applyMultipleGatesAtOnce(self, classical_register, gate_instructions):
        """
        Applyes the full circuit to the input state, after having generated the circuit's matrix.
        
        Note, aplitudes are updated in place.
        
        Args:
            classical_register: classical register form QuantumCircuit.classicalRegister
            gate_instructions: a list of tuples representing the instructions and the systems upon which it acts. 
                   - single quantum gates: ('quantum_gate_name', qubit_index)
                   - cnot quantum gate: ('cnot', (control_index, target_index))
                   - classical controlled gate: ('classicalControl', (control_index, target_index), 'quantum_gate_name'))
    
        """
        
        gateMatrix = gates.generateAllGatesAtOnce(self, classical_register, gate_instructions)
        self.amplitudes = np.array(np.matmul(self.amplitudes, gateMatrix.todense()))    
        
        
    def measure(self):
        """
        Turns amplitudes into probabilities, and generate the measurement outcomes
        """
        # Get this list of probabilities, by squaring the absolute value of the amplitudes
        self.probabilities = []
        for amp in np.nditer(self.amplitudes):
            probability = np.absolute(amp)**2
            self.probabilities.append(probability)

        # Now, we need to make a weighted random choice of all of the possible
        # output states (done with the range function)

        results = list(range(len(self.probabilities)))
        self.value = [int(outcome) for outcome in np.binary_repr(
                                np.random.choice(results, p=self.probabilities),
                                self.n_qubits)
                     ]
        return self.value
    
    
    def formatAmplitudes(self):
        """
        An (almost pretty) priting utility
        """
        return [i.round(4) for i in self.amplitudes]