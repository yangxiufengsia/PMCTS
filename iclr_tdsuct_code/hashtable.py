#!/usr/bin/env python

import sys

class Item:
    key   = ""
    value = 0

    def __init__(self,key,value):
        self.key = key
        self.value = value

    def print(self):
        print("  '" + self.key + "' / " + str(self.value) )


class HashTable:
    'Common base class for a hash table'
    tableSize    = 0
    entriesCount = 0
    alphabetSize = 2*26
    hashTable    = []


    def __init__(self, size):
        self.tableSize = size
        self.hashTable = [[] for i in range(size)]

    def char2int(self,char):
        if char >= 'A' and char <= 'Z':
            return ord(char)-65
        elif char >= 'a' and char <= 'z':
            return ord(char)-65-7
        else:
            raise NameError('Invalid character in key! Alphabet is [a-z][A-Z]')

    def char2int1(self,char):
        return ord(char)

    def hashing(self,key):
        hash = 0
        for i,c in enumerate ( key ):
            print (c)
            hash += pow(self.alphabetSize, len(key)-i-1) * self.char2int1(c)

        #print (hash % self.tableSize)
        return hash % self.tableSize

    def insert(self,item):
        hash = self.hashing(item.key)
        # print(hash)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == item.key:
                del self.hashTable[hash][i]
                self.entriesCount -= 1
        self.hashTable[hash].append(item)
        self.entriesCount += 1

    def get(self,key):
        print ("Getting item(s) with key '" + key + "'")
        hash = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                return self.hashTable[hash]
        print (" NOT IN TABLE!")
        return None

    def delete(self,key):
        print ("Deleting item with key '" + key + "'")
        hash = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                del self.hashTable[hash][i]
                self.entriesCount -= 1
                return
        print (" NOT IN TABLE!")

    def print(self):
        print ( ">>>>> CURRENT TABLE BEGIN >>>>" )
        print ( str(self.getNumEntries()) + " entries in table" )
        for i in range(self.tableSize):
            print ( " [" + str(i) + "]: " )
            for j in range(len(self.hashTable[i])):
                self.hashTable[i][j].print()
        print ( "<<<<< CURRENT TABLE END <<<<<" )

    def getNumEntries(self):
        return self.entriesCount


if __name__ == "__main__":
    hs = HashTable(10)
    item1 = Item("&",1)
    item2=Item("&c\n",2)
    hs.insert(item1)
    hs.print()
    hs.insert(item2)
    hs.print()
