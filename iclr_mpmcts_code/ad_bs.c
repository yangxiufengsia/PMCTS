// Find out rank, size

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv)
{
  int world_rank;
  int world_size;
  int *buf, *bbuf;
  int bufsize, bsize;
  //int kkk[3];
  int flag=0;
  int count=0;

  // MPI_Init(NULL, NULL);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  bufsize = 1000 * sizeof(int);
  buf = (int *)malloc( bufsize );
  MPI_Buffer_attach( buf, bufsize );

  if (world_rank == 0) {
    int number=0;
    //printf (sizeof(number));
    int a=0;
    for( a=1 ; a < 5; a++ ){
      MPI_Bsend(
          /* buf = */ &number, 
          /* count   = */ 1, 
          /* datatype = */ MPI_INT, 
          /* destination  = */ 1, 
          /* tag          = */ 0, 
          /* communicator = */ MPI_COMM_WORLD);
    }
  } 

  if (world_rank != NULL) 
  {
    int number;
    //printf(sizeof(number));
    int x;
    while (1){
      MPI_Iprobe(MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&flag,&status);
      /* MPI_Get_count( &status, MPI_INT, &count ); */
      /* if (count==1) { */
      if (flag) {

        // int dest1=rand() % (11 +1);
        int dest1=rand() % (12);

        MPI_Recv(
            /* data   = */ &number, 
            /* count        = */ 1, 
            /* datatype     = */ MPI_INT, 
            /* source       = */ MPI_ANY_SOURCE, 
            /* tag          = */ 0, 
            /* communicator = */ MPI_COMM_WORLD, 
            /* status       = */ MPI_STATUS_IGNORE);
        x=x+number;

        MPI_Bsend(&number,1,MPI_INT,dest1,0,MPI_COMM_WORLD);
        printf("Process %d received number %d\n", world_rank, number);
      }
      //usleep(1000);
    }
  }

  MPI_Finalize();
}
