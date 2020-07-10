

import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import AtomInfo
from SDF2xyzV2 import Read_sdf

def tansfersdf(com,index):
    m2 = Chem.MolFromSmiles(com)
    AllChem.EmbedMolecule(m2)
    m3 = Chem.AddHs(m2)
    AllChem.EmbedMolecule(m3)
    Chem.MolToMolFile(m3,'logp_CheckMol'+str(index)+'.sdf')
    try:
        opt = AllChem.UFFOptimizeMolecule(m3,maxIters=200)
    except:
        opt=None
    if opt!=None:
        Chem.MolToMolFile(m3,'logp_CheckMolopt'+str(index)+'.sdf')
        SpinMulti=Read_sdf('logp_CheckMolopt'+str(index)+'.sdf')
    else:
        SpinMulti=0

    return SpinMulti

