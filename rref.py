import numpy as np

def modinv(a, mod):
    '''
    Finds the multiplicative inverse of a modulo mod if one exists.
    '''
    for x in range(1, mod):
        if (a * x) % mod == 1:
            return x
    raise ValueError(f'{a} has no multiplicative inverse modulo {mod}.')

def firstnonzero(row, mod):
    '''
    Finds the index of the first non-zero element of the row modulo mod.
    '''
    for i, x in enumerate(row):
        if x % mod != 0:
            return i
    raise ValueError('All elements are zero modulo the given mod.')

def swaprows(M, i, j):
    '''
    Swaps rows i and j of the matrix M.
    '''
    M[[i, j]] = M[[j, i]]
    print(f"Swapped rows {i} and {j}:")
    print_matrix(M)

def subrow(M, i, mod):
    '''
    Subtracts row i from each other row in the matrix M modulo mod.
    Assumes that the first non-zero element of row i is 1.
    '''
    f = firstnonzero(M[i], mod)
    for j in range(M.shape[0]):
        if i != j and M[j, f] % mod != 0:
            factor = M[j, f]
            print(f"R{j} = R{j} - ({factor}) * R{i}")
            M[j] -= factor * M[i]
            M[j] %= mod
            print_matrix(M)

def normrow(M, i, mod):
    '''
    Normalizes row i of the matrix M such that the first non-zero element is 1 modulo mod.
    '''
    f = firstnonzero(M[i], mod)
    inv = modinv(M[i, f], mod)
    print(f"R{i} = ({inv}) * R{i}")
    M[i] *= inv
    M[i] %= mod
    print_matrix(M)

def print_matrix(M):
    '''
    Prints the matrix M.
    '''
    print(np.array2string(M, separator=', '))

def modrref(M, mod):
    '''
    Computes the row-reduced echelon form of the matrix M modulo mod.
    '''
    print("Starting matrix:")
    print_matrix(M)
    r = 0
    while r < M.shape[0]:
        try:
            f = firstnonzero(M[r], mod)
        except ValueError:
            r += 1
            continue

        swap = False
        if r > 0:
            try:
                g = firstnonzero(M[r - 1], mod)
                if f < g:
                    swap = True
            except ValueError:
                swap = True
        if swap:
            swaprows(M, r, r - 1)
            continue

        normrow(M, r, mod)
        subrow(M, r, mod)
        r += 1

    print("Final row-reduced echelon form:")
    print_matrix(M)
    return M

# Example of usage:
# A = np.array([[1, 4, 2, 0, 2, 0, 2],
#               [3, 2, 2, 3, 4, 1, 3],
#               [0, 0, 3, 1, 2, 0, 4],
#               [2, 3, 1, 3, 3, 4, 1],
#               [3, 2, 1, 1, 0, 2, 4]], dtype=int)

A = np.array([[1, 2, 0, 1, 2, 0],
              [1, 1, 1, 1, 1, 1],
              [1, 0, 2, 0, 1, 0]], dtype=int)

modrref(A, 3)
