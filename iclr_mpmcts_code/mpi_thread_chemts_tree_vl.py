from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
import random as pr
from copy import deepcopy
from types import IntType, ListType, TupleType, StringTypes
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






class chemical:

    def __init__(self):

        self.position=['&']
    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):

        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:

    def __init__(self, position = None, parent = None, state = None, nodelock=threading.Lock()):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        self.depth=0
        self.expanded=[]
        self.nodeadded=[]
        self.random_node=[]
        self.all_posible=[]
        self.generate_smile=[]
        self.node_index=[]
        self.valid_smile=[]
        self.new_compound=[]
        self.nodelock=nodelock
        self.ucb=[]
        self.core_id=[]
        self.virtual_loss=0
        self.num_thread_visited=0



    def Selectnode(self):
        #self.nodelock.acquire()

        ucb=[]
        #print "current node's virtual_loss:",self.num_thread_visited,self.virtual_loss
        for i in range(len(self.childNodes)):
            #print "current node's childrens' virtual_loss:",self.childNodes[i].num_thread_visited,self.childNodes[i].virtual_loss
            ucb.append((self.childNodes[i].wins+self.childNodes[i].virtual_loss)/
            (self.childNodes[i].visits+self.childNodes[i].num_thread_visited)+
            1.0*sqrt(2*log(self.visits+self.num_thread_visited)/(self.childNodes[i].visits+self.childNodes[i].num_thread_visited)))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        #print "which thread's ucb:",threading.currentThread().getName()
        #print ucb
        #self.nodelock.release()
        return s

    def Addnode(self, m):

        #n = Node(position = m, parent = self, state = s)
        self.nodeadded.remove(m)
        n = Node(position = m, parent = self)
        self.childNodes.append(n)
        return n



    def Update(self, result):
        #self.nodelock.acquire()
        #print "update visits:",self.visits
        self.visits += 1
        self.wins += result
        #self.nodelock.release()

    def delete_virtual_loss(self):
        self.num_thread_visited=0
        self.virtual_loss=0

    def expanded_node1(self, model, state, val):
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

        ex_time=time.time()

        for i in range(1):
            global graph
            with graph.as_default():
                predictions=model.predict(x_pad)
                #print "shape of RNN",predictions.shape
                preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
                preds = np.log(preds) / 1.0
                preds = np.exp(preds) / np.sum(np.exp(preds))
                #next_probas = np.random.multinomial(1, preds, 1)
                next_probas=np.argsort(preds)[-5:]
		next_probas=list(next_probas)
		#next_int=np.argmax(next_probas)
                #get_int.append(next_int)
                #all_nodes.append(next_int)

        #all_nodes=list(set(all_nodes))
	if 0 in next_probas:
	   next_probas.remove(0)
        all_nodes=next_probas
	#print all_nodes

        self.expanded=all_nodes
        #print self.expanded
	exfi_time=time.time()-ex_time
	#print exfi_time


    def expanded_node(self, model,state,val):
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
	    #ex_time=time.time()
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
        print all_nodes

        self.expanded=all_nodes
        #print self.expanded
	#exfi_time=time.time()-ex_time
	#print exfi_time

    def node_to_add(self, all_nodes,val):
        added_nodes=[]
        for i in range(len(all_nodes)):
            added_nodes.append(val[all_nodes[i]])

        self.nodeadded=added_nodes

        #print "childNodes of current node:", self.nodeadded

    def random_node_to_add(self, all_nodes,val):
        added_nodes=[]
        for i in range(len(all_nodes)):
            added_nodes.append(val[all_nodes[i]])

        self.random_node=added_nodes





        #print "node.nodeadded:",self.nodeadded





"""Define some functions used for RNN"""


def expanded_node(self, model,state,val):
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
    #ex_time=time.time()
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
    print all_nodes





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
    ##keep R3 as Ch3, keep R1 as CH3, optimize R2
    #beta_first=["[Li]","O","/", "C", '(','C',')','=','C','(']
    #beta_last=[')','\\','C','(','C',')','=','O']

    ##keep R3 as CF3, keep R2 as CH3, optimize R1
    #beta_first=["[Li]","O","/", "C", "("]
    #beta_last=[')','=','C','(',"C",')','\\','C','(',"C","(",'F',')','(','F',')','F',')','=','O']


    ##keep R3 as CF3, keep R2 as CF3, optimize R1
    #beta_first=["[Li]","O","/", "C", "("]
    #beta_last=[')','=','C','(',"C","(",'F',')','(','F',')','F',')','\\','C','(',"C","(",'F',')','(','F',')','F',')','=','O']

    ## keep R1 as CF3, optimize R3
    #beta_first=["[Li]","O","/", "C", "(", "C","(",'F',')', '(','F',')','F',')','=','C','\\','C','(']
    #beta_last=[')','=','O']

    ## keep R3 as CF3, otpimize R1

    beta_first=["[Li]","O","/", "C", "("]
    beta_last=[')','=','C','\\','C','(',"C","(",'F',')','(','F',')','F',')','=','O']

    ## keep R1 as CH3, optimize R3
    #beta_first=["[Li]","O","/", "C", "(", "C",')','=','C','\\','C','(']
    #beta_last=[')','=','O']


    ## keep R3 as bene, optimize R1
    #beta_first=["[Li]","O","/", "C", "("]
    #beta_last=[')','=','C','\\','C','(','c','1','c','c','c','c','c','1',')','=','O']

    for i in range(len(all_posible)):
        total_generated=all_posible[i]

        generate_smile=[]

        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        beta_first.extend(generate_smile)
        beta_first.extend(beta_last)
        #new_compound.append(generate_smile)
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



def ChemTS_run(rootnode,result_queue,lock,chem_model):
    """----------------------------------------------------------------------"""
    """----------------------------------------------------------------------"""
    global maxnum
    global gau_file_index
    global ind_mol
    start_time=time.time()
    while time.time()-start_time<3600:
        node = rootnode
        state=['&']
        """selection step"""
        node_pool=[]
        lock.acquire()


        while node.expanded!=[] and node.nodeadded==[] and len(node.childNodes)==len(node.expanded):
            node.num_thread_visited+=1
            node.virtual_loss+=0
            node = node.Selectnode()
            state.append(node.position)
        depth.append(len(state))
	#lock.release()

        """this if condition makes sure the tree not exceed the maximum depth"""
        if len(state)>20:
            re=-10


            #lock.acquire()
            while node != None:
                node.Update(re)
                node.delete_virtual_loss()
                node = node.parentNode
            lock.release()
        else:
            """expansion step"""
            #lock.acquire()
            if node.expanded==[]:
                #maxnum+=1
                #ind_mol+=1
                node.expanded_node(chem_model,state,val)
                node.node_to_add(node.expanded,val)
                node.random_node_to_add(node.expanded,val)
                node.num_thread_visited+=1
                node.virtual_loss+=0
                #print "node.nodeadded:",node.nodeadded
                if node.nodeadded!=[]:
                    m=random.choice(node.nodeadded)
                    node=node.Addnode(m)
                    node.num_thread_visited+=1
                    node.virtual_loss+=0


            else:
                node.num_thread_visited+=1
                node.virtual_loss+=0
                if node.nodeadded!=[]:
                    m=random.choice(node.nodeadded)
                    node=node.Addnode(m)
                    node.num_thread_visited+=1
                    node.virtual_loss+=0
            #print "m is:",m
            maxnum+=1
            ind_mol+=1
            lock.release()
            """simulation step"""
            lock.acquire()
            dest_core=random.choice(free_core_id)
            free_core_id.remove(dest_core)
            comm.send([state,m,ind_mol], dest=dest_core, tag=START)
            lock.release()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
			#with open('/home/yang/DP-ChemTS/csvresult.csv','wb') as file:
			 #   for line in text:
			#	    file.write(line)
			#		file.write('\n')
            lock.acquire()
			#result_test.append[data]
            free_core_id.append(data[3])
            #with open('/home/yang/csvfile.csv','wb') as file:
            #    for line in result_test:
            #        file.write(str(line))
            #        file.write('\n')
            lock.release()

            tag = status.Get_tag()
            if tag == DONE:
                if data[0]!=-1000:
                    lock.acquire()
                    all_com_beta.append(data[2])
                    all_com_gap.append(data[0])
                    all_com_lumo.append(data[1])
                    with open('/home/yang/ex1/all_com_beta.csv','wb') as file:
                        for line4 in all_com_beta:
                            file.write(str(line4))
                            file.write('\n')
                    with open('/home/yang/ex1/all_com_gap.csv','wb') as file:
                        for line5 in all_com_gap:
                            file.write(str(line5))
                            file.write('\n')
                    with open('/home/yang/ex1/all_com_lumo.csv','wb') as file:
                        for line6 in all_com_lumo:
                            file.write(str(line6))
                            file.write('\n')
                    lock.release()
                    if 3.9<=data[0]<=4.1 and data[1]<=-2.5:
                        re=1.0
                        lock.acquire()
                        wave_compounds.append(data[2])
                        com_gap.append(data[0])
                        com_lumo.append(data[1])
                        with open('/home/yang/ex1/com_beta.csv','wb') as file:
                            for line1 in wave_compounds:
                                file.write(str(line1))
                                file.write('\n')
                        with open('/home/yang/ex1/com_gap.csv','wb') as file:
                            for line2 in com_gap:
                                file.write(str(line2))
                                file.write('\n')
                        with open('/home/yang/ex1/com_lumo.csv','wb') as file:
                            for line3 in com_lumo:
                                file.write(str(line3))
                                file.write('\n')

                        lock.release()
                    else:
                        re=0.0
                    #re=(0.01*data[0])/(1.0+abs(0.01*data[0]))

                if data[0]==-1000:
                    re=-1
                if m=='\n':
                    re=-4.0

            lock.acquire()
            """backpropation step"""
            while node!= None:
                #print "node.parentNode:",node.parentNode
                node.Update(re)
                node.delete_virtual_loss()
                node = node.parentNode
            lock.release()


    result_queue.put([wave_compounds])


def gaussion_workers(chem_model,val):
    while True:
        simulation_time=time.time()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag==START:
            state=task[0]
            m=task[1]
            ind=task[2]
            all_posible=chem_kn_simulation(chem_model,state,val,m)
            generate_smile=predict_smile(all_posible,val)
            new_compound=make_input_smile(generate_smile)
            score=[]
            kao=[]

            try:
                m = Chem.MolFromSmiles(str(new_compound[0]))
            except:
                m=None
            #if m!=None and len(task[i])<=81:
            if m!=None:
                stable=tansfersdf(str(new_compound[0]),ind)
                if stable==1.0:
                    try:
                        SDFinput = 'CheckMolopt'+str(ind)+'.sdf'
                        calc_sdf = GaussianDFTRun('B3LYP', '6-31G', 1, 'uv homolumo', SDFinput, 0)
                        outdic = calc_sdf.run_gaussian()
                        wavelength=outdic['uv'][0]
						#gap=outdic['gap'][0]
						#lumo=outdic['gap'][1]

                    except:
					    wavelength=None
                else:
                    wavelength=None
                if wavelength!=None and wavelength!=[]:
                    wavenum=wavelength[0]
                    gap=outdic['gap'][0]
                    lumo=outdic['gap'][1]
                else:
                    wavenum=-1000
                    gap=-1000
                    lumo=-1000
            else:
                wavenum=-1000
                gap=-1000
                lumo=-1000
            #score.append(wavenum)
            score.append(gap)
            score.append(lumo)
            score.append(new_compound[0])
            score.append(rank)

            comm.send(score, dest=0, tag=DONE)
            simulation_fi_time=time.time()-simulation_time
            print "simulation_fi_time:",simulation_fi_time
        if tag==EXIT:
            MPI.Abort(MPI.COMM_WORLD)

    comm.send(None, dest=0, tag=EXIT)





if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    size=comm.size
    rank=comm.rank
    status=MPI.Status()
    READY, START, DONE, EXIT = 0, 1, 2, 3

    #smile_old=zinc_data_with_bracket_original()
    #val,smile=zinc_processed_with_bracket(smile_old)
    #print val
    val=['\n', '&', 'C', 'O', '(', 'F', ')', '1', '2', '=', '#', '[C@H]', '[C@@H]', '3', '[O-]', '[C@@]', '[C]', '[CH]', '/', '[C@]', '[CH2]', '4', '[O+]', '[O]', '5']
    #val= ['\n', '&', 'C', '#', '/', '1', '=', '(', 'O', ')', '[O-]', 'c', '[C@@H]', '[C@H]', 'F', '2', '3', 'o', '4', '5', '[C@@]', '[C@]', '6', '[o+]', '[O+]', '7']
    #val=['\n','&','/','=','#','(',')',"C",'c','O','o','[o+]','[O+]','F','[C@@H]','[C@H]','[C]', '[CH]','[C@@]','[C@]','[O]','[O-]','[CH2]',"1","2","3","4","5"]
    #val=['\n', '&', 'C', '[C@@H]', '(', 'N', ')', 'O', '=', '1', '/', 'c', 'n', '[nH]', '[C@H]', '2', '[NH]', '[C]', '[CH]', '[N]', '[C@@]', '[C@]', 'o', '[O]', '3', '#', '[O-]', '[n+]', '[N+]', '[CH2]', '[n]']
    #val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

    chem_model=loaded_model()
    graph = tf.get_default_graph()
    chemical_state = chemical()

    num_simulations=499
    thread_pool=[]
    lock=Lock()
    gau_file_index=0

    """initialization of the chemical trees and grammar trees"""
    root=['&']
    rootnode = Node(position= root)
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
    result_queue=Queue()
    free_core_id=range(1,num_simulations+1)
    #free_core_id=[40,80,120,160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760,800,840,880,920,960,1000]

    if rank==0:
        for thread_id in range(num_simulations):
            thread_best = Thread(target=ChemTS_run,args=(rootnode,result_queue,lock,chem_model))
            thread_pool.append(thread_best)

        for i in range(num_simulations):
            thread_pool[i].start()
        for i in range(num_simulations):
            thread_pool[i].join()
        for i in range(num_simulations):
            result.append(result_queue.get())
        comm.Abort()
        for i in range(len(free_core_id)):
            comm.send(None, dest=i+1, tag=EXIT)
    #elif rank%40==0 and rank!=0:
    else:
	    gaussion_workers(chem_model,val)
    #else:
       # while True:
           # time.sleep(30)
