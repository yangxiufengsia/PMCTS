from mpi4py import MPI
import time
import random
import numpy as np
import numpy
#import mpiunittest as unittest
#import arrayimpl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#MPI.Attach_buffer(mem)
#sbuf = array( s, typecode, s)
#rbuf = array(-1, typecode, s)
#mem=[None]*1000#np.array()
mem= np.zeros(8192)
#mem  = array( 0, typecode, 2*(s+MPI.BSEND_OVERHEAD)).as_raw()
MPI.Attach_buffer(mem)
if rank == 0:
    data = [0,0,0]
    #data=np.array(data,dtype=str)
    #print (data)
    #print (data.shape)
    #data = numpy.arange(1000, dtype='i')
    #print (data.shape)
    for i in range(36):
        print (i)
        comm.bsend(data, dest=1, tag=11)
        #print ()
        #re.wait()
if rank!=None:
    #data = np.zeros(3)
    #data = numpy.arange(1000, dtype='i')
    #charlist = [' ']*3
    #data=np.array(charlist)
    #data = np.empty(3,dtype=str)
    while True:
        ret=comm.Iprobe(source=MPI.ANY_SOURCE, tag=11)
        if ret==True:
            data1=comm.recv(source=MPI.ANY_SOURCE, tag=11)
            dest1=random.choice([0,1,2,3,4,5,6,7,8,9,10,11])
            comm.bsend(data1,dest=dest1,tag=11)
            print ("rank:"+str(rank)+" "+"is sending message to:"+" "+str(dest1))
            #re.wait()

