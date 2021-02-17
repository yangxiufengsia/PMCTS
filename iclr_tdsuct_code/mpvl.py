from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
import random as pr
from copy import deepcopy
import itertools
import time
import math
import tensorflow as tf
import argparse
import subprocess
from load_model import loaded_model
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sys
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from threading import Thread, Lock, RLock
import threading
from Queue import Queue
from mpi4py import MPI
from RDKitText import tansfersdf
from SDF2GauInput import GauTDDFT_ForDFT
from GaussianRunPack import GaussianDFTRun
import sascorer
import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
from collections import deque

"""-----------------------------------------------------------------------------
"""
"""
-----------------------transportation table part--------------------------------
define the hash table for distributing jobs and store tree node information
"""
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
        return hash

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



"""
-----------------------tree search part-----------------------------------------
"""



"""Define some functions used for RNN"""
def chem_kn_simulation(model,state,val,added_nodes):
    all_posible=[]

    end="\n"

    position=[]
    position.extend(state)
    position.append(added_nodes)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=20, dtype='int32',
        padding='post', truncating='pre', value=0.)
    while not get_int[-1] == val.index(end):
        predictions=model.predict(x_pad)
        #print "shape of RNN",predictions.shape
        preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int=np.argmax(next_probas)
        a=predictions[0][len(get_int)-1]
        next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
        get_int.append(next_int)
        x=np.reshape(get_int,(1,len(get_int)))
        x_pad = sequence.pad_sequences(x, maxlen=20, dtype='int32',
            padding='post', truncating='pre', value=0.)
        if len(get_int)>20:
            break
    total_generated.append(get_int)
    all_posible.extend(total_generated)


    return all_posible




def predict_smile(all_posible,val):
    new_compound=[]
    beta_first=["[Li]","O","/", "C", "("]
    beta_last=[')','=','C','\\','C','(',"C","(",'F',')','(','F',')','F',')','=','O']
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        beta_first.extend(generate_smile)
        beta_first.extend(beta_last)
        new_compound.append(beta_first)


    return new_compound


def make_input_smile(generate_smile):
    new_compound=[]
    for i in range(len(generate_smile)):
        middle=[]
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com=''.join(middle)
        new_compound.append(com)
    return new_compound

"""
define the tree
"""
class Node:

    def __init__(self):
        self.state = ['&']
        self.childNodes = []
        self.parentNode=None
        self.wins = 0
        self.visits = 0
        self.vl=0
        self.threads=0
        self.reward=None


def selection(node):
    ucb=[]
    for i in range(len(node.childNodes)):
        ucb.append((node.childNodes[i].wins+node.childNodes[i].virtual_loss)/
        (node.childNodes[i].visits+node.childNodes[i].num_thread_visited)+
        1.0*sqrt(2*log(node.visits+node.num_thread_visited)
        /(node.childNodes[i].visits+node.childNodes[i].num_thread_visited)))
    m = np.amax(ucb)
    indices = np.nonzero(ucb == m)[0]
    ind=pr.choice(indices)
    s=node.childNodes[ind]

    return s

def expansion(node, model):
    state=node.state
    val=['\n', '&', 'C', 'O', '(', 'F', ')', '1', '2', '=', '#',
     '[C@H]', '[C@@H]', '3', '[O-]', '[C@@]', '[C]',
     '[CH]', '/', '[C@]', '[CH2]', '4',
      '[O+]', '[O]', '5']
    all_nodes=[]
    end="\n"
    position=[]
    position.extend(state)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))
    get_int=get_int_old
    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=20, dtype='int32',
        padding='post', truncating='pre', value=0.)
    for i in range(100):
        global graph
        with graph.as_default():
            predictions=model.predict(x_pad)
            #print "shape of RNN",predictions.shape
            preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int=np.argmax(next_probas)
            #get_int.append(next_int)
            all_nodes.append(next_int)
    all_nodes=list(set(all_nodes))
    added_nodes=[]
    for i in range(len(all_nodes)):
        state=node.state
        #added_nodes.append(val[all_nodes[i]])
        state.append(val[all_nodes[i]])
        n=Node(state=state,parentNode=node)
        node.childNodes.append(n)
    ##node.visit+=1
    #node.threads+=1

    return node



def simulation(chem_model,state,node):
    val=['\n', '&', 'C', 'O', '(', 'F', ')', '1', '2', '=', '#',
     '[C@H]', '[C@@H]', '3', '[O-]', '[C@@]', '[C]',
     '[CH]', '/', '[C@]', '[CH2]', '4',
      '[O+]', '[O]', '5']
    all_posible=chem_kn_simulation(chem_model,state)
    generate_smile=predict_smile(all_posible,val)
    new_compound=make_input_smile(generate_smile)
    #score=[]
    kao=[]
    try:
        m = Chem.MolFromSmiles(str(new_compound[0]))
    except:
        m=None
    #if m!=None and len(task[i])<=81:
    if m!=None:
        try:
            logp=Descriptors.MolLogP(m)
        except:
            logp=-1000
        SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[0]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[0]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score = -cycle_length
        SA_score_norm=(SA_score-SA_mean)/SA_std
        logp_norm=(logp-logP_mean)/logP_std
        cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
        score_one = SA_score_norm + logp_norm + cycle_score_norm
        #score.append(score_one)
        score=score_one/(1+abs(score_one))
    else:
        #score.append(-1000)
        score=-1000/(1+1000)
    #score.append(new_compound[0])
    #score.append(rank)
    node.reward=score
    return node

def update(node,r):
    node.wins+=r
    node.visits+=1
    node.reward=r
    return node

def update_vis_th(node):
    node.visits+=1
    node.threads+=1

    return node


def TDS_UCT(chem_model,num_job):
    hs=HashTable(num_job)
    q=Queue()
    while true:
        ret=comm.iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
        if ret==True:
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            tag=status.Get_tag()
            if tag==DISTRIBUTE:
                if hs.get(message.state)==None:
                    q.put(message)
                    node=expansion(message,chem_model)
                    node=update_vis_th(node)
                    hs.insert(Item(node.state,node))
                    childnode=selection(node)
                    comm.bsend(childnode,dest=hs.hashing(childnode.state),tag=distribute)
                else:
                    node=hs.get(message.state)
                    node.update_vis_th(node)
                    hs.insert(Item(node.state,node))
                    childnode=selection(node)
                    comm.bsend(childnode,dest=hs.hashing(childnode.state),tag=distribute)

            elif tag==SEARCH:
                q.put(message)
                q_node=q.get()
                node=hs.get(q_node.state)
                if node!=None:
                    if node.childNodes!=[]:
                        childnode=selection(node)
                        comm.bsend(childnode,dest=hs.hashing(childnode.state),tag=search)
                        node=simulation(chem_model,node.state,node)
                        hs.insert(Item(node.state,node))
                        comm.bsend([node.parentNode,node.reward],dest=hs.hashing(node.parentNode.state), tag='report')
                    else:
                        node=expansion(node)
                        childnode=selection(node)
                        comm.bsend(childnode,dest=hs.hashing(childnode.state),tag='search')
                        node=simulation(chem_model,node.state,node)
                        hs.insert(Item(node.state,node))
                        comm.bsend([node.parentNode,node.reward],dest=hs.hashing(node.parentNode.state), tag='report')

                else:
                    print ("bug exist! please check")

            elif tag==REPORT:
                q.put(message)
                q_node=q.get()
                node=hs.get(q_node)
                node=update(node,node.reward)
                hs.insert(Item(node.state,node))
                comm.bsend([node.parentNode,node.reward],dest=hs.hashing(node.parentNode.state), tag='report')


if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    size=comm.size
    rank=comm.rank
    status=MPI.Status()
    SEARCH, REPORT, START, DISTRIBUTE= 0, 1, 2, 3
    val=['\n', '&', 'C', 'O', '(', 'F', ')', '1', '2', '=', '#',
     '[C@H]', '[C@@H]', '3', '[O-]', '[C@@]', '[C]',
     '[CH]', '/', '[C@]', '[CH2]', '4',
      '[O+]', '[O]', '5']
    chem_model=loaded_model()
    graph = tf.get_default_graph()
    #chemical_state = chemical()
    num_simulations=12
    gau_file_index=0
    """initialization of the chemical trees and grammar trees"""
    root=['&']
    rootnode = Node(position = root)
    maxnum=0
    ind_mol=0
    reward_dis=[]
    all_compounds=[]
    all_com_beta=[]
    all_com_gap=[]
    all_com_lumo=[]
    wave_compounds=[]
    com_gap=[]
    com_lumo=[]
    depth=[]
    result=[]

    """
    start distributing jobs to all ranks
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hsm=HashTable(100)
    num_job=300
    if rank==hsm.hashing(root):
        for j in range(num_job):
            comm.send(rootnode, dest=hsm.hashing(root), tag=DISTRIBUTE)



    if rank!=None:
        TDS_UCT(chem_model,num_job)
