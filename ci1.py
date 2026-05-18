import numpy as np

# Union
def fuzzy_union(A, B):
    return np.maximum(A, B)

# Intersection
def fuzzy_intersection(A, B):
    return np.minimum(A, B)

# Complement
def fuzzy_complement(A):
    return 1 - A

# Difference
def fuzzy_difference(A, B):
    return np.maximum(0, A - B)

# Cartesian Product
def cartesian_product(A, B):
    return np.minimum.outer(A, B)

# Max-Min Composition
def max_min_composition(R, S):
    result = np.zeros((R.shape[0], S.shape[1]))

    for i in range(R.shape[0]):
        for j in range(S.shape[1]):
            result[i][j] = np.max(np.minimum(R[i, :], S[:, j]))

    return result


# Example fuzzy sets
A = np.array([0.2, 0.4, 0.6, 0.8])
B = np.array([0.3, 0.5, 0.7, 0.9])

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

# Operations
print("\nUnion:")
print(fuzzy_union(A, B))

print("\nIntersection:")
print(fuzzy_intersection(A, B))

print("\nComplement of A:")
print(fuzzy_complement(A))

print("\nDifference A-B:")
print(fuzzy_difference(A, B))

# Cartesian Product
print("\nCartesian Product:")
R = cartesian_product(A, B)
print(R)

# Another fuzzy relation
S = np.array([
    [0.6, 0.3],
    [0.8, 0.5],
    [0.4, 0.9],
    [0.7, 0.2]
])

# Max-Min Composition
print("\nMax-Min Composition:")
print(max_min_composition(R, S))