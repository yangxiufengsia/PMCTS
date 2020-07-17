from subprocess import Popen, PIPE
import cProfile
from math import *
import time
import random
import numpy as np
import random as pr
from copy import deepcopy
import itertools
import tensorflow as tf
import argparse
from load_model import loaded_logp_model, loaded_wave_model
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
from search_tree  import Tree_Node
import csv
from write_to_csv import wcsv
from enum import Enum
from parallel_mcts import p_mcts


if __name__ == "__main__":
    """
    Initialize MPI environment
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    status = MPI.Status()
    mem = np.zeros(1024 * 10 * 1024)
    MPI.Attach_buffer(mem)

    """
    Load the pre-trained RNN model
    """
    chem_model = loaded_logp_model()
    print (chem_model)
    graph = tf.get_default_graph()
    property="logP"
    node=Tree_Node(state=['&'], property=property)

    """
    Initialize HashTable
    """
    random.seed(3)
    hsm = HashTable(nprocs, node.val, node.max_len, len(node.val))

    """
    Design molecules with desired properties:
    currently available properties: logP (rdkit) and wavelength (DFT)
    """
    score,mol=p_mcts.H_MCTS(chem_model, hsm)
    #p_mcts.D_MCTS(chem_model, hsm)
    wcsv(allscore, 'OUTPUT/logp_dmcts_scoreForProcess' + str(rank))
    wcsv(allmol,'OUTPUT/logp_dmcts_generatedMoleculesForProcess' + str(rank))
