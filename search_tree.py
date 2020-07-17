from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from keras.preprocessing import sequence
import sascorer
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
from rollout import chem_kn_simulation, predict_smile, make_input_smile
import numpy as np
from math import log, sqrt
import random as pr
from RDKitText import tansfersdf
from SDF2GauInput import GauTDDFT_ForDFT
from GaussianRunPack import GaussianDFTRun

class property_simulator:
    """
    logp property
    """
    def __init__(self, property):
        self.property=property
        #print (self.property)
        if self.property=="logP":
            self.val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
                'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
                '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
                '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
                '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
                '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
            self.max_len=82
        if self.property=="wavelength":
            self.val=['\n', '&', 'C', '[C@@H]', '(', 'N', ')', 'O', '=', '1', '/', 'c', 'n', '[nH]',
                '[C@H]', '2', '[NH]', '[C]', '[CH]', '[N]', '[C@@]', '[C@]', 'o', '[O]', '3', '#',
                '[O-]', '[n+]', '[N+]', '[CH2]', '[n]']
            self.max_len=42

    def simulation(self, chem_model, state, rank, gauid):
        all_posible = chem_kn_simulation(chem_model, state, self.val, self.max_len)
        generate_smile = predict_smile(all_posible, self.val)
        new_compound = make_input_smile(generate_smile)
        if self.property=="logP":
            score,mol=self.logp_evaluator(new_compound, rank)
        if self.property=="wavelength":
            score,mol=self.wavelength_evaluator(new_compound, rank)

        return score, mol

    def logp_evaluator(self, new_compound, rank):
        ind=rank
        try:
            m = Chem.MolFromSmiles(str(new_compound[0]))
        except BaseException:
            m = None
        if m is not None:
            try:
                logp = Descriptors.MolLogP(m)
            except BaseException:
                logp = -1000
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[0]))
            cycle_list = nx.cycle_basis(
                nx.Graph(
                    rdmolops.GetAdjacencyMatrix(
                        MolFromSmiles(
                            new_compound[0]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            SA_score_norm = SA_score  # (SA_score-SA_mean)/SA_std
            logp_norm = logp  # (logp-logP_mean)/logP_std
            cycle_score_norm = cycle_score  # (cycle_score-cycle_mean)/cycle_std
            score_one = SA_score_norm + logp_norm + cycle_score_norm
            score = score_one / (1 + abs(score_one))
        else:
            score = -1000 / (1 + 1000)
        return score, new_compound[0]

    def wavelength_evaluator(self, new_compound, ind):
        ind=rank
        try:
            m = Chem.MolFromSmiles(str(new_compound[0]))
        except:
            m= None
        if m!= None:
            stable = tansfersdf(str(new_compound[0]),ind)
            if stable == 1.0:
                try:
                    SDFinput = 'CheckMolopt'+str(ind)+'.sdf'
                    calc_sdf = GaussianDFTRun('B3LYP', '3-21G*', 1, 'uv homolumo', SDFinput, 0)
                    outdic = calc_sdf.run_gaussian()
                    wavelength = outdic['uv'][0]
                except:
                    wavelength = None
            else:
                wavelength = None
            if wavelength != None and wavelength != []:
                wavenum = wavelength[0]
                gap = outdic['gap'][0]
                lumo = outdic['gap'][1]
                score = 0.01*wavenum/(1+0.01*abs(wavenum))
            else:
                score = -1
        else:
            score = -1
        return score, new_compound[0]


class Tree_Node(property_simulator):
    """
    define the node in the tree
    """
    def __init__(self, state, parentNode=None, property=property):
       
        #print (property_simulator.val)
        #self.val = property_simulator.val
        #self.max_len = property_simulator.max_len
        self.state = state
        self.childNodes = []
        self.parentNode = parentNode
        self.wins = 0
        self.visits = 0
        self.virtual_loss = 0
        self.num_thread_visited = 0
        self.reward = 0
        self.check_childnode = []
        self.expanded_nodes = []
        self.path_ucb = []
        self.childucb = []
        property_simulator.__init__(self, property)
    def selection(self):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append((self.childNodes[i].wins +
                        self.childNodes[i].virtual_loss) /
                       (self.childNodes[i].visits +
                        self.childNodes[i].num_thread_visited) +
                       1.0 *
                       sqrt(2 *log(self.visits +self.num_thread_visited) /
                            (self.childNodes[i].visits +
                                self.childNodes[i].num_thread_visited)))
        self.childucb = ucb
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = pr.choice(indices)
        self.childNodes[ind].num_thread_visited += 1
        self.num_thread_visited += 1

        return ind, self.childNodes[ind]

    def expansion(self, model):
        state = self.state
        all_nodes = []
        end = "\n"
        position = []
        position.extend(state)
        total_generated = []
        new_compound = []
        get_int_old = []
        for j in range(len(position)):
            get_int_old.append(self.val.index(position[j]))
        get_int = get_int_old
        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = sequence.pad_sequences(x, maxlen=self.max_len, dtype='int32',
                                       padding='post', truncating='pre', value=0.)
        predictions = model.predict(x_pad)
        preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        sort_index = np.argsort(-preds)
        i = 0
        sum_preds = preds[sort_index[i]]
        all_nodes.append(sort_index[i])
        while sum_preds <= 0.95:
            i += 1
            all_nodes.append(sort_index[i])
            sum_preds += preds[sort_index[i]]
        self.check_childnode.extend(all_nodes)
        self.expanded_nodes.extend(all_nodes)

    def addnode(self, m):
        self.expanded_nodes.remove(m)
        added_nodes = []
        added_nodes.extend(self.state)
        added_nodes.append(self.val[m])
        self.num_thread_visited += 1
        n = Tree_Node(state=added_nodes, parentNode=self)
        n.num_thread_visited += 1
        self.childNodes.append(n)
        return  n

    def update_local_node(self, score):
        self.visits += 1
        self.wins += score
        self.reward = score

    def backpropagation(self, cnode):
        self.wins += cnode.reward
        self.visits += 1
        self.num_thread_visited -= 1
        self.reward = cnode.reward
        for i in range(len(self.childNodes)):
            if cnode.state[-1] == self.childNodes[i].state[-1]:
                self.childNodes[i].wins += cnode.reward
                self.childNodes[i].num_thread_visited -= 1
                self.childNodes[i].visits += 1


