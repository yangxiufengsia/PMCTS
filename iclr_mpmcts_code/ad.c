// Find out rank, size

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    int world_rank;
    int world_size;
    int flag=0;
    int count=0;
    //MPI_Init(argc, **argv);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) 
{
        int number[]={0,0, 0};
         //printf (sizeof(number));
        int a=0;
        for( a=1 ; a < 36; a++ ){
            MPI_Send(
               /* data = */ &number, 
              /* count   = */ 1, 
             /* datatype = */ MPI_INT, 
             /* destination  = */ 1, 
            /* tag          = */ 0, 
         /* communicator = */ MPI_COMM_WORLD);
        }

} 
//if (world_rank!=NULL)
{int number[3];
     //printf(sizeof(number));
    while (1){
        MPI_Iprobe(MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&flag,&status);
        //MPI_Get_count( &status, MPI_INT, &count );
        if (flag)
        {

            int dest1=rand() % (11 +1 );

            MPI_Recv(
                   /* data   = */ &number, 
                   /* count        = */ 1, 
                   /* datatype     = */ MPI_INT, 
                   /* source       = */ MPI_ANY_SOURCE, 
                   /* tag          = */ 0, 
                   /* communicator = */ MPI_COMM_WORLD, 
                   /* status       = */ MPI_STATUS_IGNORE);


            MPI_Send(&number,1,MPI_INT,dest1,0,MPI_COMM_WORLD);
            printf("Process received number %d\n",number);                                    
        }
       }
}
}
