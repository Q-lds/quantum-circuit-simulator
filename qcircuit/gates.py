import operator
from functools import reduce
import numpy as np
import scipy.sparse

class gates: 
    i = np.complex(0, 1)

    singleQubitGates = {
    # Pauli-X / Not Gate
    'X': np.matrix([
        [0, 1],
        [1, 0]
    ]),
    # Pauli-Y Gate
    'Y': np.matrix([
        [0, -i],
        [i, 0]
    ]),
    # Pauli-Z Gate
    'Z': np.matrix([
        [1, 0],
        [0, -1]
    ]),
    # Hadamard Gate
    'H': np.multiply(1. / np.sqrt(2), np.matrix([
        [1, 1],
        [1, -1]
    ])),
     # Identity Gate
    'Id': np.eye(2),
    # S & S Dagger Gate
    'S': np.matrix([
        [1, 0],
        [0, i]
    ]),
    'SDagger': np.matrix([
        [1, 0],
        [0, i]
    ]).conjugate().transpose(),
    # T & T Dagger / Pi over 8 Gate
    'T': np.matrix([
        [1, 0],
        [0, np.e**(i * np.pi / 4.)]
    ]),
    'TDagger': np.matrix([
        [1, 0],
        [0, np.e**(i * np.pi / 4.)]
    ]).conjugate().transpose()
    }
    
    
    @staticmethod
    def generateSingleGatesMatrix(gate, n_qubits, qubit1_index, sparse=False):
        """
        Turns a single system gate into the corresponding tensor structure;
        For example, if we have QuantumRegister(3), and we want to apply a X gate on qubit 2, this method 
        generates `I \otimes X \otimes I`
        
        Args:
            gate: a string containing the desired gate name
            n_qubit: total number of qubits
            qubit1_index: index representing the qubit location within the quantum register
            sparse: a boolean needed when we apply all gates at once, and we want the individual gates to be represented as parse matrices
        
        """
        identity = gates.singleQubitGates['Id']
        mainGate = gates.singleQubitGates[gate]
        gateOrder = (mainGate if i == qubit1_index else identity
                     for i in range(1, n_qubits + 1))
        
        if sparse:
            returnGate = scipy.sparse.coo_matrix(reduce(np.kron, gateOrder))
        else:
            returnGate = reduce(np.kron, gateOrder)  
        return returnGate           

       
    @staticmethod
    def generateCnotGate(n_qubits, control_qubit, target_qubit, sparse):
        """
        Implementation take from :
        https://physics.stackexchange.com/questions/277069/equation-for-a-cnot-gate-matrix-that-works-on-multiple-qubits
        
        Implements a Cnot between two arbitary qubits of a quantum register
        
        Args:
            n_qubit:  total number of qubits
            control_qubit: integer index representing the location of the contor qubit
            target_qubit: integer index representing the location of the contor qubit
            sparse: a boolean needed when we apply all gates at once, and we want the individual gates to be represented as parse matrices
        """
        control = control_qubit
        target = target_qubit

        identity = gates.singleQubitGates['Id']
        X = gates.singleQubitGates['X']
        # np.nan turns out to be a valid placeholder for numpy. See the link above for further details
        C = np.matrix([
            [np.nan, 0],
            [0, 1]])

        # Set the gate order
        gateOrder = []
        for i in range(1, n_qubits + 1):
            if (i == control):
                gateOrder.append(C)
            elif (i == target):
                gateOrder.append(X)
            else:
                gateOrder.append(identity)
                
        newGate = reduce(np.kron, gateOrder)

        # Now we need to fill in the nans left by the reduce tensor producs
        # The rule is simple:
        # Just convert
        # np.nans across the diagonal ---> 1
        # np.nans in the off-diagonal entries --->0
        index_nan = np.isnan(newGate.A1)
        modulo_slice = np.array([True if i % (newGate.shape[1]+1)==1 else False for i in range(1,len(newGate.A1)+1)])
        diagonal_and_nans = index_nan & modulo_slice
        newGate.A1[diagonal_and_nans] = 1

        index_nan = np.isnan(newGate.A1)
        newGate.A1[index_nan] = 0

        if sparse:
            returnGate = scipy.sparse.coo_matrix(np.matrix(newGate))
        else:
            returnGate = np.matrix(newGate)                 

        return returnGate
    
    @staticmethod
    def classicalControl(self, gate, control, target, allGates):
        """
        This is a wrapper around applyClassicalControl, but living in the gates class.
        
        Get the gate's name along with the control and target indexes and perform a single gate operation
        """      
        return self.applyClassicalControl(self,control, target, allGates )
        
        
    @staticmethod
    def generateAllGatesAtOnce(quantum_register, classical_register, gate_instructions):
        """
        It is more efficient to combine all the gates before applying them to the initial state. Fundamentally, we perpare each each gate by expanding calculating the correctly ordered tensor product, and then turn the resulting matrix into a sparse matric.
        
        Args:
           quantum_register: a copy of the quantum register
           classical_register: a copy of the classical_register
           gate_instructions: a list of tuples representing the instructions and the systems upon which it acts. 
               - single quantum gates: ('quantum_gate_name', qubit_index)
               - cnot quantum gate: ('cnot', (control_index, target_index))
               - classical controlled gate: ('classicalControl', (control_index, target_index), 'quantum_gate_name'))
        """
        gateList = [i[0] for i in gate_instructions]
        newGatesList = []
        for i in range(len(gateList)):
            if gateList[i].lower() == 'cnot':
                newGatesList.append(gates.generateCnotGate(quantum_register.n_qubits, 
                                                           control_qubit=gate_instructions[i][1][0],
                                                           target_qubit=gate_instructions[i][1][1],
                                                           sparse=True))
            elif gateList[i].lower() == 'classicalcontrol': 
                newGatesList.append(quantum_register.applyClassicalControl(gate_instructions[i][2], 
                                                                           control=gate_instructions[i][1][0],
                                                                           target=gate_instructions[i][1][1],
                                                                           classical_register=classical_register,
                                                                           allGates=True
                                                                          )
                                    )
            elif gateList[i].lower() in [i.lower() for i in  gates.singleQubitGates.keys()]:
                newGatesList.append(gates.generateSingleGatesMatrix(gate_instructions[i][0],
                                                                    quantum_register.n_qubits,
                                                                    gate_instructions[i][1],
                                                                    sparse=True))
            else:
                raise Exception("Gate is not recognized. ('%s' was provided) \
                             Allowed gates are 'cnot', 'classicalcontrol', 'x', 'y', 'z', 'h', 'id', 's', 'sdagger', 't', 'tdagger'. Gates names are case insensitive."
                               % (gateList[i]))
                
#         print([i.todense() for i in newGatesList])
                           
        return reduce(operator.mul, newGatesList)