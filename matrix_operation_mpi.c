#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#ifdef _OPENMP 
  #include <omp.h>
#endif

#define MIN 0.0    //min value of random value
#define MAX 10.0   //max value of random value 
#define OMPTHREADS 64  //max number of omp threads 
#define NUM_RUNS 10  //number of runs to test algorithm 

//---- global varible definition----
int n;    //matrix size
double wstart, wend;  //variable for wall clock time
double wtimeSeq, wtimeMpi, wtimePar;  //safe different value of wstart - wend
double avg_time_seqCS, avg_time_seqT, avg_time_mpiCS, avg_time_mpiT, avg_time_parallel, avg_speedup, avg_efficiency, avg_weak;  //avg values of analyses parameters 
double total_time_parallel;  
double data_transfered;    //amount of data are transfered 
double bandwidthSeq, bandwidthMpi, bandwidthPar;    //bandwidth


//---- functions declartion ----
double random_float(double min, double max);  //initialize matrix with random value
bool checkPowerTwo(const int n);  //check if n is power of 2
void printMat (double **matrix);  //print matrix
bool checkMat (double **m1, double *m2, double **m3);  //check correctness of each matrix transpose
//sequential functions 
double** matTranspose(double **matrix);
bool checkSym(double **matrix);
//mpi functions 
void matTransposeMPI(double *M, double *T, int myrank, int size);
bool checkSymMPI(double *M, int myrank, int size);
//Explicit Parallelization with OpenMP functions
double** matTransposeOMP(double **matrix);
bool checkSymOMP(double **matrix);

//---- bonus function declaration ----
double** matTransposeBlock(double **matrix);
void matTransposeBlockMPI(double *M, double *T, int myrank, int size); 

//
//
//    MAIN
//
//

int main(int argc, char **argv){
    srand(time(NULL));
    double **M, **TS, **TP;
    double *M_mpi, *T_mpi, *Tblock;  //flatten array for MPI 
    
    int myrank, size;  //myrank: determines the label of the calling process | size:  determines the number of processes
  	int lib_version, lib_subversion;
  	    
    //MPI environment 
    MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	//MPI_Get_version(&lib_version, &lib_subversion);
   
     //check if input value n is a power of 2
     
   if (argc < 2 || !checkPowerTwo(atoi(argv[1]))) {
       if(myrank == 0){
       	  fprintf(stderr,"usage: scriptname <integer size_matrix> or <not power of 2>\n");
       	  return -1;
       }
   }
    

    //check if n is divisible by the number of processes
    if (atoi(argv[1]) % size != 0) {
        if (myrank == 0) {
            fprintf(stderr, "Error! matrix size must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    //get matrix size
    n = atoi(argv[1]);
    
    //define amount data transfered
    data_transfered = 2.0 * n * n * sizeof(double);
    
    M_mpi = (double*)malloc(n*n*sizeof(double));
    T_mpi = (double*)malloc(n*n*sizeof(double));
    //Tblock = (double*)malloc(n*n*sizeof(double));
    
    if(myrank == 0){
        //allocate space for matrix
        M = (double**)malloc(n*sizeof(double*));
        
        for(int i=0; i<n; i++){
            M[i] = (double*)malloc(n*sizeof(double));         
        }
        
        //Initialize random n x n matrix
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                M[i][j] = random_float(MIN, MAX);
            }
        }
        
        //Initialize M_mpi
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                M_mpi[i*n+j] = M[i][j];
            }
        }
    }
    
    MPI_Bcast(M_mpi, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    
    
    if(myrank == 0){
      printf("\n\t\t________ COMPARE IMPLEMENTATIONS (matrix:%dx%d) ________\n\n",n, n);
      printf("\t-MPI version with processors = %d\n", size);
	  	printf("\t-Running each simulation %d times\n\n", NUM_RUNS);
    
    //Task 1: Sequential implementation
    //Wall time of checkSym
        printf("1. Sequential Implementation Times:\n");
        total_time_parallel = 0.0;
        for(int run = 0; run < NUM_RUNS; run++){
            wstart=MPI_Wtime();
            
            checkSym(M);
            
            wend=MPI_Wtime();
            wtimeSeq = wend-wstart;
            total_time_parallel += wtimeSeq; 
        }
        
        avg_time_seqCS = total_time_parallel / NUM_RUNS; 
        
        printf("\t-checkSym function: average (on %d runs) wall clock time  = %8.4g sec\n", NUM_RUNS, avg_time_seqCS);
        
        
        //wall clock time of matTranspose
        TS = matTranspose(M);
        
        bandwidthSeq = data_transfered / (avg_time_seqT * 1e9);// GB/s
        printf("\t-matTranspose function: average (on %d runs) wall clock time = %8.4g sec\n", NUM_RUNS, avg_time_seqT);
        printf("\t\t-bandwidth imp. Sequential: %lf GB/s\n\n", bandwidthSeq);
    }
    
    
    //Task 2: MPI implementation
    
    if(myrank == 0){
        printf("2. MPI Parallelization with MPI Average Times:\n");
        printf("\t-checkSymMPI function analyses (on %d runs): \n", NUM_RUNS);
        printf("Num_Processes | Avg_MPI_Time | Avg_Speedup | Avg_Efficiency | Bandwidth | Weak_Scalability \n");
    }
    
    //wall clock time of checkSymMPI

    total_time_parallel = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        //start
        wstart = MPI_Wtime();

        checkSymMPI(M_mpi, myrank,size);

        wend = MPI_Wtime();
        wtimeMpi = wend - wstart;
        total_time_parallel += wtimeMpi;
        
    }

    // Compute average parallel time, speedup, and efficiency
    avg_time_mpiCS = total_time_parallel / NUM_RUNS;
    avg_speedup = avg_time_seqCS / avg_time_mpiCS;
    avg_efficiency = avg_speedup / size;
    avg_weak = avg_time_seqT / n*avg_time_mpiCS;
    bandwidthMpi = data_transfered / (avg_time_mpiCS * 1e9);

    // Print results for each process mpi count
    if(myrank == 0){
        printf("%13d | %12f | %11f | %13.3f%% | %9.4f | %5.12f\n", size, avg_time_mpiCS, avg_speedup, avg_efficiency * 100, bandwidthMpi, avg_weak);
    }
    
    
    //wall clock time of matTransposeMPI
    if(myrank == 0){
        printf("\n\t-matTransposeMPI function analyses (on %d runs): \n", NUM_RUNS);
    }
    matTransposeMPI(M_mpi, T_mpi, myrank, size); 
    
    
    if(myrank ==0){
     //Task 3: Explicit Parallelization with OpenMP
    //wall clock time of checkSym
        printf("3. Explicit Parallelization with OpenMP Average Times:\n");
        
#ifdef _OPENMP
        printf("\t-checkSymOMP function analyses (on %d runs): \n", NUM_RUNS);
        printf("Num_Threads | Avg_Parallel_Time | Avg_Speedup | Avg_Efficiency | Weak_Scalability\n"); 
        // Test performance with thread counts of 1, 2, 4, 8, 16, 32, and 64
        for (int num_threads = 2; num_threads <= 64; num_threads *= 2){
            omp_set_num_threads(num_threads);
    
            total_time_parallel = 0.0;
    
            for (int run = 0; run < NUM_RUNS; run++) {
                //start
                wstart = MPI_Wtime();
    
                checkSymOMP(M);
    
                wend = MPI_Wtime();
                wtimePar = wend - wstart;
                total_time_parallel += wtimePar;
            }
    
            // Compute average parallel time, speedup, and efficiency
            avg_time_parallel = total_time_parallel / NUM_RUNS;
            avg_speedup = avg_time_seqCS / avg_time_parallel;
            avg_efficiency = avg_speedup / num_threads;
            avg_weak = avg_time_seqT / n*avg_time_parallel;
    
            // Print results for each thread count
            printf("%11d | %17f | %11f | %13.3f%% | %5.120f\n", num_threads, avg_time_parallel, avg_speedup, avg_efficiency * 100, avg_weak);
        }
        
        printf("\n\t-matTransposeOMP function analyses (on %d runs): \n", NUM_RUNS);
        
        // wall clock time of matTranspose
        TP = matTransposeOMP(M);
#endif
    }
    
    
    /*
    //check if each matrix transpose goes well in each approach
    if(myrank == 0){
        if(checkMat(TS, T_mpi, TP)){
            printf("Matrix transpose goes well in each implemetation!\n");
        }else{
            printf("ERROR!something goes wrong some implemetation.\n");
        }
    }
    */
    
    
    //deallocate memory
    if(myrank == 0){
        for(int i=0; i<n; i++){
            free(M[i]);
            free(TS[i]);
        }
        
        free(M);
        free(TS);
    }
    free(M_mpi);
    free(T_mpi);
    
    
    MPI_Finalize();
    
    return 0;
}



//
//
//_____________FUNCTIONS IMPLEMENTATION___________
//
//



double random_float(double min, double max) {
    //return (float)(rand()) / (float)(rand());  //without range [min, max]
    return min + (double)rand() / RAND_MAX * (max - min);  //with range [min, max]
}

bool checkPowerTwo(const int n){
    
    if(n <= 0){
        return false;
    }
    
    return (n & (n-1)) == 0;
}

void printMat (double **matrix){
    //print matrix
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
          printf("%f\t", matrix[i][j]);
      }
      printf("\n");
    }
}

void printMatMPI (double *matrix){
    //print matrix
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
          printf("%f\t", matrix[i*n+j]);
      }
      printf("\n");
    }
}

bool checkMat (double **m1, double *m2, double **m3){
    int i, j;
    
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(m1[i][j] != m2[i*n+j] ||  m2[i*n+j] != m3[i][j]){
                return false; 
            }
        }
    }
    
    return true; 
}

double** matTranspose(double **matrix){
    double **T = (double**)malloc(n*sizeof(double*));
    
    for(int i=0; i<n; i++){
        T[i] = (double*)malloc(n*sizeof(double)); 
    }
    
    total_time_parallel = 0.0; 
    for(int run = 0; run < NUM_RUNS; run++){
        //start time
        wstart=MPI_Wtime();
        
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                T[i][j] = matrix[j][i];
            }
        }
        //end time
        wend=MPI_Wtime();
        wtimeSeq = wend - wstart;
        total_time_parallel += wtimeSeq;
    }
    
    avg_time_seqT = total_time_parallel / NUM_RUNS;
    return T;
}

bool checkSym(double **matrix){
    bool sym = true;
    
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(matrix[i][j] != matrix[j][i]){
                sym = false;
            }
        }
    }
    
    return sym;
}

void matTransposeMPI(double *M, double *T, int myrank, int size) {
    int row = n / size;
    int start = row * myrank;
    int end = start + row;
    double* temp = (double*)malloc(n * n * sizeof(double));

    if (myrank == 0) {
        printf("Num_Processes | Avg_MPI_Time | Avg_Speedup | Avg_Efficiency | Bandwidth | Weak_Scalability\n");
    }

    total_time_parallel = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        // Start time
        wstart = MPI_Wtime();

        for (int i = start; i < end; i++) {
            for (int j = 0; j < n; j++) {
                temp[i * n + j] = M[j * n + i];
            }
        }

        MPI_Gather(temp + start * n, row * n, MPI_DOUBLE, T, row * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Allgather(temp + start * n, row * n, MPI_DOUBLE, T, row * n, MPI_DOUBLE, MPI_COMM_WORLD); // Distribute matrix transpose at each process

        // End time
        wend = MPI_Wtime();
        wtimeMpi = wend - wstart;
        total_time_parallel += wtimeMpi;
    }

    // Compute average parallel time, speedup, and efficiency
    avg_time_mpiT = total_time_parallel / NUM_RUNS;
    avg_speedup = avg_time_seqT / avg_time_mpiT;
    avg_efficiency = avg_speedup / size;
    avg_weak = avg_time_seqT / n*avg_time_mpiT;
    bandwidthMpi = data_transfered / (avg_time_mpiT * 1e9); // GB/s

    if (myrank == 0) {
        printf("%13d | %12f | %11f | %13.3f%% | %9.4f | %5.12f\n\n", size, avg_time_mpiT, avg_speedup, avg_efficiency * 100, bandwidthMpi, avg_weak*100);
    }

    free(temp);
}





bool checkSymMPI(double *M, int myrank, int size){
    int row = n/size; 
    int start = row * myrank;
    int end = start + row;
    
    bool symPro = true;
    bool symRed;
    
    for(int i=start; i<end; i++){
        for(int j=0; j<n; j++){
            if(M[i*n + j] != M[j*n+i]){
                symPro = false;
            }
        }
    }
    
    MPI_Allreduce(&symPro, &symRed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    
    return symRed;
    
}

double** matTransposeOMP(double **matrix){
    double **T = (double**)malloc(n*sizeof(double*));

    for(int i=0; i<n; i++){
        T[i] = (double*)malloc(n*sizeof(double)); 
    }
#ifdef _OPENMP  
    // Test performance with thread counts of 1, 2, 4, 8, 16, 32, and 64
    printf("Num_Threads | Avg_Parallel_Time | Avg_Speedup | Avg_Efficiency | Bandwidth | Weak_Scalability\n"); 
    for(int num_threads=2; num_threads <=OMPTHREADS; num_threads *= 2){
        
        omp_set_num_threads(num_threads);
        total_time_parallel = 0.0;
        
        for(int run = 0; run < NUM_RUNS; run++){
            //start time
            wstart=MPI_Wtime();
            
            #pragma omp parallel for collapse(2) schedule (guided, 2)
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    T[i][j] = matrix[j][i];
                }
            }
            
            //end time
            wend=MPI_Wtime();
            wtimePar = wend - wstart;
            total_time_parallel += wtimePar;
        }
        
        // Compute average parallel time, speedup, and efficiency
        avg_time_parallel = total_time_parallel / NUM_RUNS;
        avg_speedup = avg_time_seqT / avg_time_parallel;
        avg_weak = avg_time_seqT / n*n*avg_time_parallel;
        avg_efficiency = avg_speedup / num_threads;
        bandwidthPar = data_transfered / (avg_time_parallel * 1e9);  // GB/s
        
        printf("%11d | %17f | %11f | %13.3f%% | %9.4f | %5.12f\n", 
               num_threads, avg_time_parallel, avg_speedup, avg_efficiency * 100, bandwidthPar, avg_weak);
        
     }
#else
    
    printf("error _OPENMP is not defined"); 
    
#endif

    return T;
}


bool checkSymOMP(double **matrix) {
    bool sym = true;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) shared(sym) schedule (guided, 2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                sym = false;
            }   
        }
    }
    
    return sym;
#else
    printf("error _OPENMP is not defined\n");
    return false;
#endif
}

/*
double** matTransposeBlock(double **matrix){
    double **T = (double**)malloc(n*sizeof(double*)); 
    int block;
    
    for(int i=0; i<n; i++){
        T[i] = (double*)malloc(n*sizeof(double)); 
    }
    
    block = n/sqrt(4); //size of block
    
    for(int i=0; i<n; i+=block){
        for(int j=0; j<n; j+=block){
            for(int k=0; k<block; k++){
                for(int t=0; t<block; t++){
                    T[i+k][j+t] = matrix[j+k][i+t];
                }
            }
        }
    }
    
    return T;
}

void matTransposeBlockMPI(double *M, double *T, int myrank, int size) {
    int block = n / size;
    int start = block * myrank;
    double* temp = (double*)malloc(block * block * sizeof(double));
    
    // Trasponi i blocchi locali
    for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
            temp[j * block + i] = M[(start + i) * n + (start + j)];
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
    
    // Raccogli i blocchi trasposti in T
    MPI_Allgather(temp, block * block, MPI_DOUBLE, T, block * block, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); 
    
    // Ricomponi la matrice T
    for (int i = 0; i < size; i++) {
        int row_start = (i / block) * block;
        int col_start = (i % block) * block;
        for (int j = 0; j < block; j++) {
            for (int k = 0; k < block; k++) {
                T[(row_start + j) * n + (col_start + k)] = temp[i * block * block + j * block + k];
            }
        }
    }
    
    if(myrank==0){
        printf("FINAL:\n");
        printMatMPI(T);
    }
    
    free(temp);
}

*/

