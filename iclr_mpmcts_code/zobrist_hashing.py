# Set up a 82x64 array of random ints.
from random import randint
S = 82 ## length of smiles
P = 64 ## number of atom elements
zobristnum =[[0]*P for i in range(S)]
def myinit():
    global zobristnum
    for i in range(S):
        for j in range(P):
            zobristnum[i][j]=randint(0, 2**64)


def zhash(board):
    global zobristnum
    hashing_value = 0;
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
                'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
                '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
                '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
                '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
                '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

    for i in range(S):
        piece = None
        if i<=len(board)-1:
            if board[i] in val:
                piece = val.index(board[i])
        if(piece != None):
            hashing_value ^= zobristnum[i][piece]
    return hashing_value


# Testing:
myinit()
b3=['&', 'O', '=', 'S', '(', '=', 'S', ')']
hv=zhash(b3)
x=bin(hv)
y=str(x)
k=y[-10:]
m=int(k, 2)
print (x)
print (hv)
print (zobristnum)
print (m)
