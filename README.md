This project requires installation of following softwares:

1. Platform tested: Linux and MacOS
2. Python (version 3.7.4) 
3. MPI (openmpi: conda install -c conda-forge openmpi)
4. mpi4py (version 3.0.3, pip install mpi4py==3.0.3)
5. RDkit (version >=2019.09.3, conda install -c rdkit rdkit)
6. Keras (version 2.0.5, pip install keras==2.0.5)
7. Tensorflow (version 1.15.2, pip install tensorflow==1.15.2)
8. Nltk (pip install nltk)
9. Networkx (pip install networkx) 
10. Gaussian 16 (a commercial software used for wavelength calculations, http://gaussian.com/gaussian16/)

To Run parallel MCTS algorithms example (logp property):

> mpiexec -n 4 python example_usage.py

> where 4 is the number of cores or processes to use. You can use more cores by changing 4 to 1024 for example. 

