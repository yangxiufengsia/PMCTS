from subprocess import Popen, PIPE
from math import *
import time
import random
import numpy as np
import random as pr
from copy import deepcopy
import itertools
from time import sleep
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
from queue import *
from mpi4py import MPI
from RDKitText import tansfersdf
from SDF2GauInput import GauTDDFT_ForDFT
from GaussianRunPack import GaussianDFTRun
import sascorer
#import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
from collections import deque
from random import randint

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

class HashTable:
    'Common base class for a hash table'
    tableSize    = 0
    entriesCount = 0
    alphabetSize = 2*26
    hashTable    = []
    def __init__(self):
        #self.tableSize = size
        self.hashTable =dict() #[[] for i in range(size)]
        self.S=82
        self.P=64
        self.zobristnum =[[0]*self.P for i in range(self.S)]
        for i in range(self.S):
            for j in range(self.P):
                self.zobristnum[i][j]=randint(0, 2**64)

    def hashing(self, board):
        val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F',
            '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]',
            's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]',
            '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6',
            '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]',
            '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]',
            '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
        hashing_value = 0;
        for i in range(self.S):
            piece = None
            if i<=len(board)-1:
                if board[i] in val:
                    piece = val.index(board[i])
            if(piece != None):
                hashing_value ^=self.zobristnum[i][piece]

        hash_key=format(hashing_value, '064b')[0:63]
        hash_key=int(hash_key, 2)
        core_dest=format(hashing_value, '064b')[-1:]
        core_dest=int(core_dest, 2)
        return hash_key,core_dest

    def insert(self,item):
        hash,_ = self.hashing(item.key)
        if self.hashTable.get(hash)==None:
            self.hashTable.setdefault(hash,[])
            self.hashTable[hash].append(item)
            #self.table_stored_size.append(1)
        else:
            #self.collisions+=1
            for i,it in enumerate(self.hashTable[hash]):
                if it.key==item.key:
                    del self.hashTable[hash][i]
            self.hashTable[hash].append(item)
            #self.collisions+=len()

    def search_table(self,key):
        hash,_ = self.hashing(key)
        if self.hashTable.get(hash)==None:
            return None
        else:
            for i,it in enumerate(self.hashTable[hash]):
                if it.key==key:
                    return it.value
        return None
"""
----------------------tree search part-----------------------------------------
"""
"""Define some functions used for RNN"""
def chem_kn_simulation(model,state,val):
    all_posible=[]
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
    x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
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
        x_pad = sequence.pad_sequences(x, maxlen=82, dtype='int32',
            padding='post', truncating='pre', value=0.)
        if len(get_int)>82:
            break
    total_generated.append(get_int)
    all_posible.extend(total_generated)
    return all_posible

def predict_smile(all_posible,val):
    new_compound=[]
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
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
    def __init__(self,state,parentNode=None,reward=None):
        self.state = state
        self.childNodes = []
        self.parentNode=parentNode
        self.wins = 0
        self.visits = 0
        self.virtual_loss=0
        self.num_thread_visited=0
        self.reward=0
        self.check_childnode=[]
        self.expanded_nodes=[]
        self.childucb=[]
def selection(node):
    ucb=[]
    for i in range(len(node.childNodes)):
        #print (node.childNodes[i].wins, node.childNodes[i].visits,node.childNodes[i].num_thread_visited)
        ucb.append((node.childNodes[i].wins+node.childNodes[i].virtual_loss)/
        (node.childNodes[i].visits+node.childNodes[i].num_thread_visited)+
        1.0*sqrt(2*log(node.visits+node.num_thread_visited)
        /(node.childNodes[i].visits+node.childNodes[i].num_thread_visited)))
    print ("ucb:",ucb)
    node.childucb=ucb
    m = np.amax(ucb)
    indices = np.nonzero(ucb == m)[0]
    ind=pr.choice(indices)
    s=node.childNodes[ind]
    s.num_thread_visited+=1
    node.num_thread_visited+=1

    return s,node

def expansion(node, model):
    st_time=time.time()
    state=node.state
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F',
            '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]',
            's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]',
            '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6',
            '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]',
            '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]',
            '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
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
    x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    predictions=model.predict(x_pad)
    preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
    preds = np.log(preds) / 1.0
    preds = np.exp(preds) / np.sum(np.exp(preds))
    sort_index = np.argsort(-preds)
    i=0
    sum_preds=preds[sort_index[i]]
    all_nodes.append(sort_index[i])
    while sum_preds<=0.95:
        i+=1
        all_nodes.append(sort_index[i])
        sum_preds+=preds[sort_index[i]]

    all_nodes=list(set(all_nodes))
    fi_time=time.time()-st_time
    #print ("exp time:",fi_time)
    node.check_childnode.extend(all_nodes)
    node.expanded_nodes.extend(all_nodes)

    return node


def addnode(node,m):
    node.expanded_nodes.remove(m)
    added_nodes=[]
    added_nodes.extend(node.state)
    added_nodes.append(val[m])
    node.num_thread_visited+=1
    n=Node(state=added_nodes,parentNode=node)
    n.num_thread_visited+=1
    node.childNodes.append(n)
    return node,n

def simulation(chem_model,state):
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n',
            '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]',
            '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]',
            '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]',
            '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]',
            '[PH+]', '[PH]', '8', '[S@@+]']
    all_posible=chem_kn_simulation(chem_model,state,val)
    generate_smile=predict_smile(all_posible,val)
    new_compound=make_input_smile(generate_smile)
    kao=[]
    try:
        m = Chem.MolFromSmiles(str(new_compound[0]))
    except:
        m=None
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
        SA_score_norm=SA_score#(SA_score-SA_mean)/SA_std
        logp_norm=logp#(logp-logP_mean)/logP_std
        cycle_score_norm=cycle_score#(cycle_score-cycle_mean)/cycle_std
        score_one = SA_score_norm + logp_norm + cycle_score_norm
        score=score_one/(1+abs(score_one))
    else:
        score=-1000/(1+1000)
    return score

def update_local_node(node,score):
    node.visits+=1
    node.wins+=score
    node.reward=score
    return node


def backpropagation(pnode,cnode):
    pnode.wins+=cnode.reward
    pnode.visits+=1
    pnode.num_thread_visited-=1
    for i in range(len(pnode.childNodes)):
        if cnode.state[-1]==pnode.childNodes[i].state[-1]:
            pnode.childNodes[i].wins+=cnode.reward
            pnode.childNodes[i].num_thread_visited-=1
            pnode.childNodes[i].visits+=1
    return pnode

def TDS_UCT(chem_model,num_cores):
    start_time=time.time()
    q=Queue() ##used for storing received jobs
    q_sim=Queue() ##used for managing rollout simulation
    while time.time()-start_time<=1800:
        ret=comm.Iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
        if ret==True:
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            cur_status=status
            node=message
            if cur_status.Get_tag()==0:
                if hsm.search_table(node.state)==None: #if node is not in the hash table
                    print ("not in table:",node.state)
                    if node.state==['&']:
                        node=expansion(node,chem_model)
                        m=random.choice(node.expanded_nodes)
                        node,n=addnode(node,m)
                        hsm.insert(Item(node.state,node))
                        _,dest=hsm.hashing(n.state)
                        comm.bsend(n,dest=dest,tag=0)
                    else:
                        if len(node.state)<81:
                            st_time=time.time()
                            score=simulation(chem_model,node.state)
                            fi_time=time.time()-st_time
                            node=update_local_node(node,score)#backpropagation on local memory
                            hsm.insert(Item(node.state,node))
                            #print ("rank"+" "+str(rank)+" "+"finished simulation with"+
                            #        " "+str(score)+" "+str(fi_time))
                            _,dest=hsm.hashing(node.state[0:-1])
                            comm.bsend(node,dest=dest,tag=1)
                        else:
                            score=-1000
                            node=update_local_node(node,score)#backpropagation on local memory
                            hsm.insert(Item(node.state,node))
                            #print ("rank"+" "+str(rank)+" "+"finished simulation with"+
                            #        " "+str(score)+" "+str(fi_time))
                            _,dest=hsm.hashing(node.state[0:-1])
                            comm.bsend(node,dest=dest,tag=1)

                else:#if node already in the local hashtable
                    node=hsm.search_table(node.state)
                    print ("in table:",node.state)
                    if node.state==['&']:
                        if node.expanded_nodes!=[]:
                            m=random.choice(node.expanded_nodes)
                            node,n=addnode(node,m)
                            hsm.insert(Item(node.state,node))
                            _,dest=hsm.hashing(n.state)
                            comm.bsend(n,dest=dest,tag=0)
                        else:
                            childnode,node=selection(node)
                            hsm.insert(Item(node.state,node))
                            _,dest=hsm.hashing(childnode.state)
                            comm.bsend(childnode,dest=dest,tag=0)
                    else:
                        if len(node.state)<81:
                            if node.expanded_nodes!=[]:
                                m=random.choice(node.expanded_nodes)
                                node,n=addnode(node,m)
                                hsm.insert(Item(node.state,node))
                                _,dest=hsm.hashing(n.state)
                                comm.bsend(n,dest=dest,tag=0)
                            else:
                                if node.check_childnode==[]:
                                    node=expansion(node,chem_model)
                                    m=random.choice(node.expanded_nodes)
                                    node,n=addnode(node,m)
                                    hsm.insert(Item(node.state,node))
                                    _,dest=hsm.hashing(n.state)
                                    comm.bsend(n,dest=dest,tag=0)
                                else:
                                    childnode,node=selection(node)
                                    hsm.insert(Item(node.state,node))
                                    _,dest=hsm.hashing(childnode.state)
                                    comm.bsend(childnode,dest=dest,tag=0)
                        else:
                            score=-1000
                            node=update_local_node(node,score)#backpropagation on local memory
                            hsm.insert(Item(node.state,node))
                            _,dest=hsm.hashing(node.state[0:-1])
                            comm.bsend(node,dest=dest,tag=1)
            if cur_status.Get_tag()==1:
                print ("reporting check:",node.state)
                local_node=hsm.search_table(node.state[0:-1])
                if local_node.state==['&']:
                    local_node=backpropagation(local_node,node)
                    hsm.insert(Item(local_node.state,local_node))
                    _,dest=hsm.hashing(local_node.state)
                    comm.bsend(local_node,dest=dest,tag=0)
                else:
                    local_node=backpropagation(local_node,node)
                    hsm.insert(Item(local_node.state,local_node))
                    _,dest=hsm.hashing(local_node.state[0:-1])
                    comm.bsend(local_node,dest=dest,tag=1)

if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    status=MPI.Status()
    SEARCH, REPORT= 0, 1
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
            'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
            '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
            '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
            '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
            '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
    chem_model=loaded_model()
    graph = tf.get_default_graph()
    gau_file_index=0
    """initialization of the chemical trees and grammar trees"""
    root=['&']
    child_char=None
    rootnode = Node(state=root)
    """
    start distributing jobs to all ranks
    """
    mem=np.zeros(1024*10*1024)#8192)
    random.seed(1)
    MPI.Attach_buffer(mem)
    num_cores=2
    hsm=HashTable()
    #q1=Queue()
    num_job=num_cores
    hsm=HashTable()
    if rank==0:
        _,rootdest=hsm.hashing(['&'])
        for i in range(num_job):
            comm.bsend(rootnode, dest=rootdest, tag=0)
    if rank!=None:
        TDS_UCT(chem_model,num_cores)
