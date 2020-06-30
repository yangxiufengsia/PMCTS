from subprocess import Popen, PIPE
import cProfile
from math import *
import time
import random
import numpy as np
import random as pr
from copy import deepcopy
import itertools
import math
import tensorflow as tf
import argparse
from load_model import loaded_model
from keras.preprocessing import sequence
from keras.preprocessing import sequence
import sys
from threading import Thread, Lock, RLock
from queue import *
from mpi4py import MPI
from rdkit.Chem import rdmolops
from collections import deque
from random import randint
from zobrist_hash import Item, HashTable
from search_tree  import Node, backpropagation
import csv
from write_to_csv import wcsv
from enum import Enum
from check_ucbpath import update_selection_ucbtable, compare_ucb, backtrack

class JobType(Enum):
    '''
    defines JobType tag values
    values higher than PRIORITY_BORDER (128) mean high prority tags
    FINISH is not used in this implementation. It will be needed for games.
    '''
    SEARCH = 0
    BACKPROPAGATION = 1
    PRIORITY_BORDER = 128
    TIMEUP = 254
    FINISH = 255
    @classmethod
    def is_high_priority(self, tag):
        return tag >= self.PRIORITY_BORDER.value

def d_mcts(chem_model):
    comm.barrier()
    start_time = time.time()
    allscore = []
    allmol = []
    _, rootdest = hsm.hashing(['&'])
    jobq = deque()
    timeup = False
    if rank == rootdest:
        root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
        for i in range(3 * nprocs):
            temp = deepcopy(root_job_message)
            root_job = (JobType.SEARCH.value, temp)
            jobq.appendleft(root_job)
    while not timeup:
        if rank == 0:
            if time.time()-start_time > 600:
                timeup = True
                for dest in range(1, nprocs):
                    dummy_data = tag = JobType.TIMEUP.value
                    comm.bsend(dummy_data, dest=dest, tag=JobType.TIMEUP.value)
        while True:
            ret = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if ret == False:
                break
            else:
                message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                cur_status = status
                tag = cur_status.Get_tag()
                job = (tag, message)
                if JobType.is_high_priority(tag):
                    jobq.append(job)
                else:
                    jobq.appendleft(job)
        jobq_non_empty = bool(jobq)
        if jobq_non_empty:
            (tag, message) = jobq.pop()
            if tag == JobType.SEARCH.value:
                if hsm.search_table(message[0]) == None:
                    node = Node(state=message[0])
                    if node.state == ['&']:
                        node.expansion(chem_model)
                        m = random.choice(node.expanded_nodes)
                        n = node.addnode(m)
                        hsm.insert(Item(node.state, node))
                        _, dest = hsm.hashing(n.state)
                        send_stime = time.time()
                        comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                               n.num_thread_visited, n.path_ucb]),
                                               dest=dest,
                                               tag=JobType.SEARCH.value)
                    else:
                        if len(node.state) < 81:
                            score, mol = node.simulation(chem_model, node.state)
                            allscore.append(score)
                            allmol.append(mol)
                            node.update_local_node(score)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(node.state[0:-1])
                            comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                   node.num_thread_visited, node.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.BACKPROPAGATION.value)
                        else:
                            score = -1
                            node.update_local_node(node, score)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(node.state[0:-1])
                            comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                   node.num_thread_visited, node.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.BACKPROPAGATION.value)

                else:  # if node already in the local hashtable
                    node = hsm.search_table(message[0])
                    if node.state == ['&']:
                        # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                        if node.expanded_nodes != []:
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(n.state)
                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                   n.num_thread_visited, n.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.SEARCH.value)
                        else:
                            ind, childnode = node.selection_for_dmcts()
                            hsm.insert(Item(node.state, node))
                            ucb_table = update_selection_ucbtable(node, ind)
                            _, dest = hsm.hashing(childnode.state)
                            comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                   childnode.visits, childnode.num_thread_visited,
                                                   ucb_table]),
                                                   dest=dest,tag=JobType.SEARCH.value)
                    else:
                        node.path_ucb = message[5]
                        print("check ucb:", node.wins, node.visits, node.num_thread_visited)
                        if len(node.state) < 81:
                            if node.state[-1] != '\n':
                                if node.expanded_nodes != []:
                                    m = random.choice(node.expanded_nodes)
                                    n = node.addnode(m)
                                    hsm.insert(Item(node.state, node))
                                    _, dest = hsm.hashing(n.state)
                                    comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                           n.num_thread_visited, n.path_ucb]),
                                                           dest=dest,
                                                           tag=JobType.SEARCH.value)
                                else:
                                    if node.check_childnode == []:
                                        node.expansion(chem_model)
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        hsm.insert(Item(node.state, node))
                                        _, dest = hsm.hashing(n.state)
                                        comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                               n.num_thread_visited, n.path_ucb]),
                                                               dest=dest,
                                                               tag=JobType.SEARCH.value)
                                    else:
                                        ind, childnode = node.selection_for_dmcts()
                                        hsm.insert(Item(node.state, node))
                                        ucb_table = update_selection_ucbtable(node, ind)
                                        _, dest = hsm.hashing(childnode.state)
                                        comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                               childnode.visits, childnode.num_thread_visited, ucb_table]),
                                                               dest=dest,
                                                               tag=JobType.SEARCH.value)
                            else:
                                score, mol = node.simulation(chem_model, node.state)
                                score = -1
                                allscore.append(score)
                                allmol.append(mol)
                                node.update_local_node(score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                       node.num_thread_visited, node.path_ucb]),
                                                       dest=dest,
                                                       tag=JobType.BACKPROPAGATION.value)
                        else:
                            score = -1
                            node.update_local_node(score)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(node.state[0:-1])
                            comm.bsend(np.asarray([node.state, node.reward, node.wins,
                                                   node.visits, node.num_thread_visited, node.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.BACKPROPAGATION.value)

            elif tag == JobType.BACKPROPAGATION.value:
                node = Node(state=message[0])
                node.reward = message[1]
                local_node = hsm.search_table(message[0][0:-1])
                if local_node.state == ['&']:
                    local_node = backpropagation(local_node, node)
                    hsm.insert(Item(local_node.state, local_node))
                    _, dest = hsm.hashing(local_node.state)
                    comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                           local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                           dest=dest,
                                           tag=JobType.SEARCH.value)
                else:
                    local_node = backpropagation(local_node, node)
                    local_node = backtrack(local_node, node)
                    back_flag = compare_ucb(local_node)
                    hsm.insert(Item(local_node.state, local_node))
                    if back_flag == 1:
                        _, dest = hsm.hashing(local_node.state[0:-1])
                        comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                               local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                               dest=dest,
                                               tag=JobType.BACKPROPAGATION.value)
                    if back_flag == 0:
                        _, dest = hsm.hashing(local_node.state)
                        comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                               local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                               dest=dest,
                                               tag=JobType.SEARCH.value)
            elif tag == JobType.TIMEUP.value:
                timeup = True

    wcsv(allscore, 'OUTPUT/logp_dmcts_scoreForProcess' + str(rank))
    wcsv(allmol,'OUTPUT/logp_dmcts_generatedMoleculesForProcess' + str(rank))



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    status = MPI.Status()
    SEARCH, REPORT = 0, 1
    val = ['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
           'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
           '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
           '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
           '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
           '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
    chem_model = loaded_model()
    graph = tf.get_default_graph()
    gau_file_index = 0
    """
    start distributing jobs to all ranks
    """
    mem = np.zeros(1024*10*1024)  # 8192)
    random.seed(2)
    MPI.Attach_buffer(mem)
    hsm = HashTable(nprocs)
    _, rootdest = hsm.hashing(['&'])
    d_mcts(chem_model)
