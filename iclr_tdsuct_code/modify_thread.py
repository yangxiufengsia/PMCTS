from subprocess import Popen, PIPE
from math import *
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

class HashTable:
    'Common base class for a hash table'
    tableSize    = 0
    entriesCount = 0
    alphabetSize = 2*26
    hashTable    = []
    def __init__(self, size):
        self.tableSize = size
        self.hashTable = [[] for i in range(size)]
        self.S=82
        self.P=64
        self.zobristnum =[[0]*self.P for i in range(self.S)]
        for i in range(self.S):
            for j in range(self.P):
                self.zobristnum[i][j]=randint(0, 65536)

    def hashing(self, board):
        val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F',
            '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]',
            's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]',
            '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6',
            '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]',
            '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]',
            '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

        #global zobristnum
        hashing_value = 0;
        for i in range(self.S):
            piece = None
            if i<=len(board)-1:
                if board[i] in val:
                    piece = val.index(board[i])
            if(piece != None):
                hashing_value ^=self.zobristnum[i][piece]

        hash_key=format(hashing_value, '016b')[0:12]
        #hash_key=str(bin(hashing_value)).zfill(16)[0:12]
        #print (hash_key)
        hash_key=int(hash_key, 2)
        #core_dest=str(bin(hashing_value)).zfill(16)[-3]
        core_dest=format(hashing_value, '016b')[-2:]
        core_dest=int(core_dest, 2)
        return hash_key,core_dest


    def insert(self,item):
        hash,_ = self.hashing(item.key)
        # print(hash)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == item.key:
                del self.hashTable[hash][i]
                self.entriesCount -= 1
        self.hashTable[hash].append(item)
        self.entriesCount += 1
        return hash

    def get(self,key):
        hash,dest = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                return self.hashTable[hash]
        print (" NOT IN TABLE!")
        return None

    def delete(self,key):
        hash = self.hashing(key)
        for i,it in enumerate(self.hashTable[hash]):
            if it.key == key:
                del self.hashTable[hash][i]
                self.entriesCount -= 1
                return

    def getNumEntries(self):
        return self.entriesCount

    def search_table(self,key):
        slot=self.get(key)
        if slot!=None:
            for i in range(len(slot)):
                if slot[i].key==key:
                    return slot[i].value
        else:
            return None


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
    all_possible.extend('fuck')
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
    for i in range(30):
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





def simulation(chem_model,state,q1):
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
    q1.put([score, state])

    #return score


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
    #start_time=time.time()

    #hs=HashTable(num_cores)
    q=Queue() ##used for simulation
    q_sim=Queue() ##used for saving simulation results
    print ("check all ranks:", rank)
    check_flag=0
    while True:
        ret=comm.Iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
        if ret==True:
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            #print ("check received message:",message)
            #q.put(message)
            #node=Node(state=message[0],reward=message[1])
            node=message
            cur_status=status
            #node,cur_status=q.get()
            if cur_status.Get_tag()==0:
                if hsm.search_table(node.state)==None:
                    #print ("growing the tree with child UCB:",node.childucb)

                    if node.state==['&']:
                        node=expansion(node,chem_model)
                        print ("first root:",node.expanded_nodes)
                        m=random.choice(node.expanded_nodes)
                        node,n=addnode(node,m)
                        hsm.insert(Item(node.state,node))
                        _,dest=hsm.hashing(n.state)
                        #print ("check searching message:",node.state,n.state,dest,rank)
                        comm.bsend(n,dest=dest,tag=0)
                    else:
                        if len(node.state)<81:
                            q.put(message)
                            #print ("not in table:",node.state)
                            if check_flag==0:
                                check_flag=1
                                job=q.get()
                                hsm.insert(Item(job.state,job))
                                #st_time=time.time()
                                sim_thread=Thread(target=simulation,args=(chem_model,job.state,q_sim))
                                sim_thread.start()
                                #fi_time=time.time()-st_time
                            if q_sim.qsize()!=0:
                                score,state=q_sim.get()
                                check_flag==0
                                node=hsm.search_table(state)
                                node=update_local_node(node,score)#backpropagation on local memory
                                hsm.insert(Item(node.state,node))
                                #print ("rank"+" "+str(rank)+" "+"finished simulation with"+" "+str(score)+" "+str(fi_time))
                                _,dest=hsm.hashing(node.state[0:-1])
                                comm.bsend(node,dest=dest,tag=1)


                        else:
                            score=-10
                            node=update_local_node(node,score)#backpropagation on local memory
                            hsm.insert(Item(node.state,node))
                            #print ("rank"+" "+str(rank)+" "+"finished simulation with"+" "+str(score)+" "+str(fi_time))
                            _,dest=hsm.hashing(node.state[0:-1])
                            comm.bsend(node,dest=dest,tag=1)

                else:#if node already in the local hashtable
                    node=hsm.search_table(node.state)
                    print ("in table:",node.state,node.expanded_nodes)
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
                            score=-10
                            node=update_local_node(node,score)#backpropagation on local memory
                            hsm.insert(Item(node.state,node))
                            print ("rank"+" "+str(rank)+" "+"finished simulation with"+" "+str(score)+" "+str(fi_time))
                            _,dest=hsm.hashing(node.state[0:-1])
                            comm.bsend(node,dest=dest,tag=1)
            if cur_status.Get_tag()==1:
                print ("reporting check:",node.state, rank)
                #print ("reporting node:",node.state)
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

    hsm=HashTable(65536)
    #hsm=comm.bcast(hsm,root=0)
    #_,ddd=hsm.hashing(['&'])
    hsm=comm.bcast(hsm,root=0)

    num_job=3*num_cores
    #print ("check node:",rootnode.state)
    print ("root rank:",hsm.hashing(root))
    #if rank==0:
    #if rank==2:
    print ("fuck the code")
    if rank==0:
        _,rootdest=hsm.hashing(['&'])
        for i in range(num_job):
            print ("root rank:",hsm.hashing(root))

            comm.bsend(rootnode, dest=rootdest, tag=SEARCH)

    #else:
    if rank!=None:
        TDS_UCT(chem_model,num_cores)
