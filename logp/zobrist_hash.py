from random import randint
import math
class Item:
    key = ""
    value = 0
    def __init__(self, key, value):
        self.key = key
        self.value = value

class HashTable:
    'Common base class for a hash table'
    tableSize = 0
    entriesCount = 0
    alphabetSize = 2 * 26
    hashTable = []

    def __init__(self, nprocs):
        self.hashTable = dict()  # [[] for i in range(size)]
        self.S = 82
        self.P = 64
        self.nprocs = nprocs
        self.zobristnum = [[0] * self.P for i in range(self.S)]
        for i in range(self.S):
            for j in range(self.P):
                self.zobristnum[i][j] = randint(0, 2**64)

    def hashing(self, board):
        val = ['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
               'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
               '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
               '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
               '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
               '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
        hashing_value = 0
        for i in range(self.S):
            piece = None
            if i <= len(board) - 1:
                if board[i] in val:
                    piece = val.index(board[i])
            if(piece is not None):
                hashing_value ^= self.zobristnum[i][piece]

        tail=int(math.log2(self.nprocs))
        print (tail)
        head=int(64-math.log2(self.nprocs))
        print (head)
        hash_key = format(hashing_value, '064b')[0:head]
        hash_key = int(hash_key, 2)
        core_dest = format(hashing_value, '064b')[-tail:]
        core_dest = int(core_dest, 2)
        return hash_key, core_dest

    def insert(self, item):
        hash, _ = self.hashing(item.key)
        if self.hashTable.get(hash) is None:
            self.hashTable.setdefault(hash, [])
            self.hashTable[hash].append(item)
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == item.key:
                    del self.hashTable[hash][i]
            self.hashTable[hash].append(item)

    def search_table(self, key):
        hash, _ = self.hashing(key)
        if self.hashTable.get(hash) is None:
            return None
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == key:
                    return it.value
        return None
