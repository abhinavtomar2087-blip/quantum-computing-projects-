import numpy as np

class Ket:
    def __init__(self, vec):
        self.v = np.asarray(vec, dtype=np.complex128).reshape(-1, 1)

    def dagger(self):
        return Bra(self.v.conj().T)

    def __add__(self, other):
        return Ket(self.v + other.v)

    def __sub__(self, other):
        return Ket(self.v - other.v)

    def __mul__(self, scalar):
        return Ket(scalar * self.v)

    def __rmul__(self, scalar):
        return Ket(scalar * self.v)

    def __matmul__(self, other):
        if isinstance(other, Operator):
            return Ket(other.mat @ self.v)
        elif isinstance(other, Bra):
            return (self.v.T @ other.v.T).item()
        else:
            raise TypeError("Unsupported operand type for @")

    def __repr__(self):
        return f"Ket({self.v.flatten()})"


class Bra:
    def __init__(self, vec):
        self.v = np.asarray(vec, dtype=np.complex128).reshape(1, -1)

    def dagger(self):
        return Ket(self.v.conj().T)

    def __add__(self, other):
        return Bra(self.v + other.v)

    def __sub__(self, other):
        return Bra(self.v - other.v)

    def __mul__(self, scalar):
        return Bra(scalar * self.v)

    def __rmul__(self, scalar):
        return Bra(scalar * self.v)

    def __matmul__(self, other):
        if isinstance(other, Ket):
            return (self.v @ other.v).item()
        elif isinstance(other, Operator):
            return Bra(self.v @ other.mat)
        else:
            raise TypeError("Unsupported operand type for @")

    def __repr__(self):
        return f"Bra({self.v.flatten()})"


class Operator:
    def __init__(self, mat):
        self.mat = np.asarray(mat, dtype=np.complex128)

    def dagger(self):
        return Operator(self.mat.conj().T)

    def __add__(self, other):
        return Operator(self.mat + other.mat)

    def __sub__(self, other):
        return Operator(self.mat - other.mat)

    def __mul__(self, scalar):
        return Operator(scalar * self.mat)

    def __rmul__(self, scalar):
        return Operator(scalar * self.mat)

    def __matmul__(self, other):
        if isinstance(other, Operator):
            return Operator(self.mat @ other.mat)
        elif isinstance(other, Ket):
            return Ket(self.mat @ other.v)
        elif isinstance(other, Bra):
            return Bra(other.v @ self.mat)
        else:
            raise TypeError("Unsupported operand type for @")

    def __repr__(self):
        return f"Operator(\n{self.mat}\n)"
