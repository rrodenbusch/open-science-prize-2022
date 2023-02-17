#
# @author: rrodenbusch
#     This code is an extension to qiskit.
#
#     (C) Copyright Richard Rodenbusch 2022.
#
#  This code is licensed under the Apache License, Version 2.0. You may
#  obtain a copy of this license in the LICENSE.txt file in the root directory
#  of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
#  Any modifications or derivative works of this code must retain this
#  copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#


"""Quantum state basis object."""
from sys import version_info
import re
from builtins                   import property
from typing                     import Optional
from numpy                      import log2

from qiskit.exceptions          import MissingOptionalLibraryError, QiskitError
from qiskit.quantum_info        import Statevector
from qiskit.visualization       import array_to_latex
from qiskit.visualization.array import _num_to_latex
import numpy as np

VERSIONINFO="Statebasis(major=1, minor=0, micro=3)"

def get_version():
    """Print version info
    Args:
        None
    Returns:
        Version information string
    Raises:

    Additional Information:
    """
    return  "Python Version\n" + version_info + "\nStatebasis Version" +  VERSIONINFO

def __is_power_of_two(
        num
    ):
    return (num != 0) and (num & (num-1) == 0)

def is_valid_vector(
        vector
    ):
    """Determine if the input vector is valid format and type
    Args:
        vector: Statevector, matrix or list to be validated as a basis vector
    Returns:
        True | False
    Raises:

    Additional Information:
    """
    if not isinstance(vector, Statevector):
        try:
            vector = Statevector(vector)
        except (ValueError,TypeError):
            return False
    sv_matrix = np.matrix(vector)
    np.ndim(sv_matrix)
    if np.ndim(sv_matrix) != 2:
        return False
    rows = np.shape(sv_matrix)[0]
    if rows == 1:
        sv_matrix = sv_matrix.transpose()
    (rows,cols) = np.shape(sv_matrix)
    if cols != 1:
        return False
    if not __is_power_of_two(rows):
        return False
    return True

def validate_vector(
        vector
    ):
    """Determine if the input vector is valid format and type
        and convert to Statevector
    Args:
        vector
    Returns:
        Statevector
    Raises:
        TypeError: if vector is not mutable to Statevector
        ValueError: if vector is not valid for state basis vector

    Additional Information:
    """
    if not isinstance(vector,Statevector):
        try:
            vector = Statevector(vector)
        except (ValueError,TypeError) as err:
            raise Exception("Unable to convert input to Statevector ", type(vector)) from err
    sv_matrix = np.matrix(vector)
    np.ndim(sv_matrix)
    if np.ndim(sv_matrix) != 2:
        raise ValueError("Statevector not compatible with n x 1 matrix")
    rows = np.shape(sv_matrix)[0]
    if rows == 1:
        sv_matrix = sv_matrix.transpose()

    (rows,cols) = np.shape(sv_matrix)
    if cols != 1:
        raise ValueError("Statevector must be list or n x 1 matrix")
    if not __is_power_of_two(rows):
        raise ValueError("Statevector len != 2^n")
    return vector


from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_bloch_multivector


class Statebasis():
    """Create a new basis spanning a n-qubit statevector space.

    Args:
        num_qubits: The number of qubits in the span of the basis
        basis (list(:class:`Statevector`) or list(``np.array``): This list of
            basis vectors which span the desired space
        labels  (list(string)): Text strings identify each basis vector.  Used for
            ket display
        name (str): the name of the basis set. If not set, an
            automatically generated string will be assigned.

    Raises:
        TypeError: if the vectors list is not compatible with Statevetor
        ValueError: if the inputs lists are not valid size(s)

    Examples:

        Construct 2 qubit basis set using |0>, |1> as the basis set

        .. jupyter-execute::
            from statebasis import Statebasis
            sb = Statebasis(2,[[1,0],[0,1],['0','1'])
            sb.draw()

    """
    __name     = ''
    __data     = []
    __labels   = []
    __nqubits  = 0

    def __build_basis(self, num_qubits,basis_vectors):
        new_basis = basis_vectors
        for nbit in range(1,num_qubits):
            nbit += 1
            base = new_basis
            new_basis = []
            for j in [0,1]:
                for new_vector in base:
                    new_basis.append( Statevector(np.kron(basis_vectors[j],new_vector)) )
        self.__data = new_basis
        return new_basis

    def __build_basis_latex(self,num_qubits,basis_labels):
        new_basis_latex = basis_labels
        for _ in range(1,num_qubits):
            base = new_basis_latex
            new_basis_latex = []
            for j in [0,1]:
                for new_base in base:
                    new_basis_latex.append(basis_labels[j]+new_base)
        self.__labels = new_basis_latex
        return new_basis_latex

    def __basis_to_latex(self,**kwargs):
        basis = self.__data
        basis_latex = self.__labels

        outstr = ''
        source = False
        for key, value in kwargs.items():
            if key == 'title':
                outstr = value
            if key == 'source':
                source = value
        if source:
            outstr += "\n\n"
            newline = "\n"
        else:
            outstr += "\\\\\\\\"
            newline = "\\\\"

        idx = 0
        for newvec in basis:
            astr = array_to_latex(newvec,prefix="\\lvert "+basis_latex[idx]+"\\rangle \\text{ = }",
                                  source=True)
            outstr = outstr + newline + astr
            idx += 1

        if source is False:
            try:
                # pylint: disable=import-outside-toplevel
                from IPython.display import Latex
            except ImportError as err:
                raise MissingOptionalLibraryError(
                    libname="IPython",
                    name="array_to_latex",
                    pip_install="pip install ipython",
                ) from err
            return Latex(f"$${outstr}$$")

        return outstr

    def project(
            self,
            vector
        ):
        """ Project the input vector onto the basis set
        Args:
            vector
        Returns:
            weights: list of complex projection coeficients
        Raises:
            TypeError: if vector is not mutable to Statevector
            ValueError: if vector is not valid for state basis vector

        Additional Information:
        """
        vector = validate_vector(vector)
        basis_set = self.__data
        weights = []
        for basis_vec in basis_set:
            coef = basis_vec.inner(vector)
            weights.append(coef)
        return weights

    def __draw_projection(
            self,
            vector,
            prefix='',
            precision=5,
            source=False,
            show_all=False
        ): # pylint: disable=too-many-arguments
        outstr = prefix
        prec = precision
        # Put together the weights and vectors
        weights = self.project(vector)
        idx = 0
        first=True
        for coef in weights:
            c_latex = _num_to_latex(coef,precision=prec)
            if show_all or ( not c_latex == "0" ):
                if not first:
                    outstr += "\\text{  }\\mathbb{+}\\text{  }"
                outstr += "\\text{  (}"+c_latex+"\\text{)  }\\lvert "+self.__labels[idx]+"\\rangle"
                first = False
            idx += 1
        if source is False:
            try:
                # pylint: disable=import-outside-toplevel
                from IPython.display import Latex
            except ImportError as err:
                raise MissingOptionalLibraryError(
                        libname="IPython",
                        name="array_to_latex",
                        pip_install="pip install ipython",
                    ) from err
            return Latex(f"$${outstr}$$")

        return outstr

    def __init__(self, num_qubits, basis, labels, name=''):
        self.__name = name
        self.__nqubits = num_qubits
        self.__data    = basis
        self.__labels  = labels
        self.__build_basis(num_qubits, basis)
        self.__build_basis_latex(num_qubits, labels)

    def draw(self,
             vector: Optional[Statevector] = None,
             **kwargs
             ):
        """ Draw the basis set or a vector projected onto the basis set
        Args:
            vector (Statevector): Vector be projected onto the basis
                    If missing, the basis set of vectors is drawn
            precision (int): For numbers not close to integers or common terms, the number of
                         decimal places to round to.
            prefix (str): Latex string to be prepended to the latex, intended for labels.
            source (bool): If ``False``, will return IPython.display.Latex object. If display is
                           ``True``, will instead return the LaTeX source string.
        Returns:
            str or IPython.display.Latex:
                If ``source`` is ``True``, a ``str`` of the LaTeX
                                      representation of the basis (or projection),
                else an ``IPython.display.Latex`` representation of the basis (or projection)

        Raises:

        Additional Information:
        """
        if vector is None:
            return self.__basis_to_latex(**kwargs)

        return self.__draw_projection(vector,**kwargs)

    @property
    def name(self):
        """ Return the name of the basis set
        Args:
        Returns:
            name: string
        Raises:

        Additional Information:
        """
        return self.__name

    @property
    def data(self):
        """ Return the list of statevectors in the basis set
        Args:
        Returns:
            list: Statevector
        Raises:

        Additional Information:
        """
        return self.__data

    @property
    def labels(self):
        """ Return the list of latex labels for the basis set
        Args:
        Returns:
            list: strings
        Raises:

        Additional Information:
        """
        return self.__labels

    @property
    def num_qubits(self) -> int:
        """Return number of qubits."""
        return len(self.__nqubits)

    @property
    def version(self):
        """ Return the version string for statebasis libary
        Args:
        Returns:
            string: Version information
        Raises:

        Additional Information:
        """

        return get_version()

    @classmethod
    def from_label(cls, num_qubits, label, name: Optional[str] = ''):
        """Return a basis set as product of Pauli X,Y,Z eigenstates.
        Args:
            num_qubits: the number of qubits to be spanned by the basis set
            label (string): a eigenstate string ket label (see table for
                            allowed values).

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label {One Of}
             - Basis[0]
             - Basis[1]
           * - ``"X | H"``
             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`
             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`
           * - ``"Y | R"``
             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`
             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`
           * - ``"Z | 0"``
             - :math:`[1, 0] - math:`[0, 1]`
             - :math:`[1, 0] - math:`[0, 1]`

        Args:
            label (string): a eigenstate string label (see table for
                            allowed values).

        Returns:
            Statebasis: The N-qubit basis state

        Raises:
            QiskitError: if the label contains invalid characters.
        """
        # Check label is valid
        label = label.upper()
        if len(label) != 1 or re.match(r'^[XHYRZ0R\+]+$', label) is None:
            raise QiskitError('Label contains invalid characters.')
        match label:
            case "X":
                data = [ Statevector.from_label('+'), Statevector.from_label('-') ]
                labels = ['+','-']
            case "H":
                data = [ Statevector.from_label('+'), Statevector.from_label('-') ]
                labels = ['+','-']
            case "+":
                data = [ Statevector.from_label('+'), Statevector.from_label('-') ]
                labels = ['+','-']
            case "Y":
                data = [ Statevector.from_label('r'), Statevector.from_label('l') ]
                labels = ['R','L']
            case "R":
                data = [ Statevector.from_label('r'), Statevector.from_label('l') ]
                labels = ['R','L']
            case "Z":
                data = [ Statevector.from_label('0'), Statevector.from_label('1') ]
                labels = ['0','1']
            case "0":
                data = [ Statevector.from_label('0'), Statevector.from_label('1') ]
                labels = ['0','1']
        basis = Statebasis(num_qubits,data,labels,name=name)
        return basis

def draw(*args,basis_label='Z',**kwargs):
    """Overload for a Statevector.draw() that will accept basis_label\
        Args:
            vector: Statevector to be drawn prejected onto basis set
            basis: Options[Statebasis] The basis to be projected onto
                        if not provided basis will be generated from basis_label
            basis_label (string): a eigenstate string ket label (see table for
                                  allowed values).
                                  Default = 'Z' to use the (0,1) basis set

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label {One Of}
             - Basis[0]
             - Basis[1]
           * - ``"X | H"``
             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`
             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`
           * - ``"Y | R"``
             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`
             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`
           * - ``"Z | 0"``
             - :math:`[1, 0] - math:`[0, 1]`
             - :math:`[1, 0] - math:`[0, 1]`

        Returns:
        Returns:
            str or IPython.display.Latex:
                If ``source`` is ``True``, a ``str`` of the LaTeX
                                      representation of the basis (or projection),
                else an ``IPython.display.Latex`` representation of the basis (or projection)
        Raises:
            QiskitError: if the label contains invalid characters
            TypeError: if arg[0] is incompatible with the Statevector class
            ValueError: if arg[0] is incorrect size for vector to be spanned
        """
    vector = None
    basis  = None
    if len(args) == 0:
        raise AttributeError("Nothing to print ")
    if len(args) > 1:
        raise AttributeError("Maximum attribute count(2) exceeded")

    if isinstance(args[0],Statebasis):
        basis = args[0]
    else:
        vector = Statevector(args[0])

    if len(args) > 1:
        if vector is None:
            vector = Statevector(args[1])
        else:
            if not isinstance(args[1],Statebasis):
                raise TypeError("Unable to create state basis from input",type(args[1]))
            basis = args[1]

    if vector is None:
        raise AttributeError("No Statevector found in attributes")
    if basis is None:
        num_qubits = int(log2(len(vector)))  # Statevector is 2^n data elements
        basis = Statebasis.from_label(num_qubits,label=basis_label)
    return basis.draw(*args,**kwargs)
