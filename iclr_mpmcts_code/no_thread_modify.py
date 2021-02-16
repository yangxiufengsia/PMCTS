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


def noneed():
    if node.check_childnode==[]:
        node=expansion(node,chem_model)
        m=random.choice(node.expanded_nodes)
        node=addnode(node,m)
        #node=update_vt_search(node)
        hs.insert(Item(make_string(node.state),node))
        #score=simulation(chem_model,node.state,node)
    else:
        if len(node.check_childnode)==len(node.childNodes):
            childnode=selection(node)
            comm.bsend(childnode,dest=hs.hashing(make_string(childnode.state)),tag=0)
        else:
            m=random.choice(node.expanded_nodes)
            node=addnode(node,m)
            hs.insert(Item(make_string(node.state),node))
    return 0

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
            #print (c)
            hash += pow(self.alphabetSize, len(key)-i-1) * self.char2int1(c)

        #print (hash % self.tableSize)
        return int(hash % self.tableSize)

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
       # print ("Getting item(s) with key '" + key + "'")
        hash = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                return self.hashTable[hash]
        print (" NOT IN TABLE!")
        return None

    def delete(self,key):
        #print ("Deleting item with key '" + key + "'")
        hash = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                del self.hashTable[hash][i]
                self.entriesCount -= 1
                return
        #print (" NOT IN TABLE!")


    def getNumEntries(self):
        return self.entriesCount

    def search_table(self,key):
        slot=self.get(key)
        if slot!=None:
            for i in range(len(slot)):
                if slot[i].key==key:
                    return slot[i].value
            #else:
                #return None
        else:
            return None

def make_string(str_list):
    str=''.join(str_list)
    return str

"""
-----------------------tree search part-----------------------------------------
"""





"""Define some functions used for RNN"""
def chem_kn_simulation(model,state,val):

    all_posible=[]

    end="\n"

    position=[]
    position.extend(state)
    #position.append(added_nodes)
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
    #print ("all_possible child:",all_posible)


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
        self.reward=None
        self.check_childnode=[]
        self.expanded_nodes=[]
        self.childucb=[]


def selection(node):
    ucb=[]
    for i in range(len(node.childNodes)):
        ucb.append((node.childNodes[i].wins+node.childNodes[i].virtual_loss)/
        (node.childNodes[i].visits+node.childNodes[i].num_thread_visited)+
        0.5*sqrt(2*log(node.visits+node.num_thread_visited)
        /(node.childNodes[i].visits+node.childNodes[i].num_thread_visited)))
    #print ("check ucb:",ucb)
    node.childucb=ucb
    m = np.amax(ucb)
    indices = np.nonzero(ucb == m)[0]

    ind=pr.choice(indices)
    s=node.childNodes[ind]
    s.num_thread_visited+=1
    node.num_thread_visited+=1

    return s,node

def expansion(node, model):
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
    for i in range(60):
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
    node.check_childnode.extend(all_nodes)
    node.expanded_nodes.extend(all_nodes)
    
    #added_nodes=[]
    #print ("added possible nodes:",all_nodes)
    for i in range(len(all_nodes)):
        #state=node.state
        added_nodes=[]
        added_nodes.extend(state)
        added_nodes.append(val[all_nodes[i]])
        #added_nodes.append(val[all_nodes[i]])
        #state.append(val[all_nodes[i]])
        n=Node(state=added_nodes,parentNode=node)
        node.childNodes.append(n)
        #node.childNodes.state.append(val[all_nodes[i]])
    ##node.visit+=1
    #node.threads+=1

    return node

def addnode(node,m):
    node.expanded_node.remove(m)
    added_nodes=[]
    added_nodes.extend(node.state)
    added_nodes.append(val[m])
    node.num_thread_visited+=1
    n=Node(state=added_nodes,parentNode=node)
    n.num_thread_visited+=1
    node.childNodes.append(n)
        
    return node





def simulation(chem_model,state):
    #time.sleep(10)
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', 
            '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', 
            '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', 
            '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', 
            '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]',
            '[PH+]', '[PH]', '8', '[S@@+]']
    all_posible=chem_kn_simulation(chem_model,state,val)
    generate_smile=predict_smile(all_posible,val)
    new_compound=make_input_smile(generate_smile)
    #score=[]
    kao=[]
    try:
        m = Chem.MolFromSmiles(str(new_compound[0]))
        #print (str(new_compound[0]))
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
        #score.append(score_one)
        score=score_one/(1+abs(score_one))
        with open('compound.csv','a') as file:
            file.write(str(new_compound[0]))
            file.write('\n')
        with open('score.csv','a') as file:
            file.write(str(score))
            file.write('\n')

    else:
        #score.append(-1000)
        score=-1000/(1+1000)
    #score.append(new_compound[0])
    #score.append(rank)
    #with open('compound.csv','a') as file:
    #    file.write(str(new_compound[0]))
    #    file.write('\n')
    #q1.put([score, state])
    return score

    #return score

def update_tree_grow(node):
    node.visits+=len(node.childNodes)
    #node.num_threads+=1
    for i in range(len(node.childNodes)):
        node.childNodes[i].visits+=1
        #node.childNodes[i].num_threads+=1
    return node


def update_reward(node,r):
    node.wins+=r

    return node

def update_report_reward(node,reward,child_char):
    node.wins+=reward
    for i in range(len(node.childNodes)):
        if child_char==node.childNodes[i].state[-1]:
            node.childNodes[i].wins+=reward
            
    return node

def TDS_UCT(chem_model,num_cores):
    #start_time=time.time()

    hs=HashTable(num_cores)
    #q=Queue() ##used for simulation
    #q1=Queue() ##used for saving simulation results
    print ("check all ranks:", rank)
    checks=0
    while True:
        ret=comm.Iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)

        if ret==True:
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            q.put(message)
            cur_status=status
            if cur_status.Get_tag()==0:
                if hs.search_table(make_string(message[0]))==None:
                    #print ("growing the tree with child UCB:",node.childucb)
                    node=Node(state=message[0])
                    node=expansion(node,chem_model)
                    node=update_tree_grow(node)
                    #hs.insert(Item(make_string(node.state),node))
                    childnode,node=selection(node)
                    comm.bsend([childnode.state,childnode.wins,childnode.state[-1]],
                            dest=hs.hashing(make_string(childnode.state)),tag=0)
                    hs.insert(Item(make_string(node.state),node))
                    print ("growing the tree with child UCB:",node.childucb)
                    #print ("current working rank " + str(rank)+" "+"with branch:"+" "+str(node.state)
                    #        +" "+"with queue size:"+" "+ str(q.qsize())+" "+"child UCB"+
                    #        " "+str(node.childucb)+" "+ "received message from " +
                    #        " "+ str(cur_status.Get_source()) +" "+ "is sending:"+" "
                    #        +str(childnode.state)+" "+"to rank "+" "+
                    #        str(hs.hashing(make_string(childnode.state))))
                    st_time=time.time()
                    score=simulation(chem_model,node.state)
                    fi_time=time.time()-st_time
                    print ("rank"+" "+str(rank)+" "+"finished simulation with"+" "+str(score)
                            +" "+str(fi_time))
                    node=update_reward(node,score)
                    hs.insert(Item(make_string(node.state),node))
                    if node.state==["&"]:
                        comm.bsend([node.state,score,node.state[-1]],
                                dest=hs.hashing(make_string(node.state)),tag=0)
                        
                    else:
                        comm.bsend([node.state[0:-1],score,node.state[-1]],
                                dest=hs.hashing(make_string(node.state[0:-1])),tag=1)
                else:
                    #print ("updating the tree")
                    node=hs.search_table(make_string(message[0]))
                    node=update_tree_grow(node)
                   # hs.insert(Item(make_string(node.state),node))
                    childnode,node=selection(node)
                    hs.insert(Item(make_string(node.state),node))
                    print ("updating the tree with child UCB:",node.childucb)

                    comm.bsend([childnode.state,childnode.wins,childnode.state[-1]],
                            dest=hs.hashing(make_string(childnode.state)),tag=0)
                    #print ("current working rank " + str(rank)+" "+"with branch:"+" "+str(node.state)
                    #        +" "+"with queue size:"+" "+ str(q.qsize())+" "+"child UCB"+
                    #        " "+str(node.childucb)+" "+ "received message from " +
                    #        " "+ str(cur_status.Get_source()) +" "+ "is sending:"+" "
                    #        +str(childnode.state)+" "+"to rank "+" "+
                    #        str(hs.hashing(make_string(childnode.state))))

                    up_st_time=time.time()
                    score=simulation(chem_model,node.state)
                    up_fi_time=time.time()-up_st_time
                    print ("rank"+" "+str(rank)+" "+"finished simulation with"+" "+str(score)
                            +" "+str(up_fi_time))
                    node=update_reward(node,score)
                    hs.insert(Item(make_string(node.state),node))
                    if node.state==["&"]:
                        comm.bsend([node.state,score,childnode.state[-1]],
                            dest=hs.hashing(make_string(node.state)),tag=0)
                    else:
                        comm.bsend([node.state[0:-1],score,node.state[-1]],
                                    dest=hs.hashing(make_string(node.state[0:-1])),tag=1)
                            
            elif cur_status.Get_tag()==1:
                print ("reporting")
                node=hs.search_table(make_string(message[0]))

                if node.state==['&']:
                    print ("check correctness:",node.state)
                    node=update_report_reward(node,message[1],message[2])
                    hs.insert(Item(make_string(node.state),node))
                    comm.bsend([node.state,message[1],message[2]],dest=hs.hashing(make_string(node.state)),tag=0)
            
                else:
                    node=update_report_reward(node,message[1],message[2])
                    print ("check correctness:",node.state)
                    hs.insert(Item(make_string(node.state),node))
                    comm.bsend([node.state[0:-1],message[1],message[2]],dest=hs.hashing(make_string(node.state[0:-1])),tag=1)
            
          



if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    #size=comm.size
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
    reward=0
    child_char=None
    rootnode = Node(state=root)
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
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    mem=np.zeros(1024*10*1024)#8192)
    MPI.Attach_buffer(mem)
    num_cores=12
    hsm=HashTable(num_cores)
    q1=Queue()
    num_job=3*num_cores
    #print ("check node:",rootnode.state)
    print ("root rank:",hsm.hashing(root))
    #if rank==0:
    #if rank==2:
    print ("fuck the code")
    if rank==0:
        for i in range(num_job):
            print ("root rank:",hsm.hashing(root))
            
            comm.bsend([root,reward,child_char], dest=int(hsm.hashing(root)), tag=SEARCH)
         
    #else:
    if rank!=None:
        TDS_UCT(chem_model,num_cores)
