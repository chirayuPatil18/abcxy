A = {'a':0.6, 'b':0.3, 'c':0.1} 
B = {'a':0.2, 'b':1.0, 'c':0.4}
def fuzzy_union(A, B):
    return {x: max(A[x], B[x]) for x in A}

def fuzzy_intersection(A, B):
    return {x: min(A[x], B[x]) for x in A}

def fuzzy_complement(A):
    return {x: 1 - A[x] for x in A}

def cartesian_product(A, B):
    relation = {}
    for x in A:
        for y in B:
            relation[(x, y)] = min(A[x], B[y])
    return relation

def min_max_composition(R, S, X, Y, Z):
    composition = {}
    for x in X:
        for z in Z:
            values = []
            for y in Y:
                if (x, y) in R and (y, z) in S:
                    values.append(min(R[(x, y)], S[(y, z)]))
            composition[(x, z)] = max(values) if values else 0
    return composition

print("Fuzzy Union:", fuzzy_union(A, B))
print("Fuzzy Intersection:", fuzzy_intersection(A, B))
print("Fuzzy Complement of A:", fuzzy_complement(A))
print("Cartesian Product of A and B:", cartesian_product(A, B))
print("Cartesian Product of B and A:", cartesian_product(B, A))
print("Min-Max Composition of R and S:", min_max_composition(cartesian_product(A, B), cartesian_product(B, A), A.keys(), B.keys(), A.keys()))