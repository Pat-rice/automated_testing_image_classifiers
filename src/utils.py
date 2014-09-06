def compress(matrix):
    previous = matrix[0]
    c = []
    sequence = 1
    for i in matrix:
        if i != previous:
            c.append([sequence - 1, previous])
            sequence = 1
        sequence += 1
        previous = i
    c.append([sequence - 1, previous])
    return c

def decompress(matrix):
    d = []
    for s in matrix:
        for t in range(s[0]):
            d.append(s[1])
    return d

# flat = np.array([[0,1],[1,0],[0,0],[1,1],[0,1],[1,1],[1,0],[0,1],[0,1],[1,1]]).flatten()
# cp = compress(flat)
# print(np.array(decompress(cp)).flatten())