# Intro_PARCO_H2
Parallelizing matrix operations using MPI

# -Requirements
GNU Compiler: Ensure that versions 5.4, 7.5, or 9.1 are available on the cluster.
MODULE: mpich-3.2.1--gcc-9.1.0 and  gcc-9.1.0 
SSH Client: Use PuTTY or MobaXterm for Windows. For macOS and Linux, the built-in SSH client is sufficient.
VPN: To access the University network from an external network, establish a secure connection using the Virtual Private Network (VPN).

# -Instructions for Compilation and Execution

1) Download Files: Obtain the commands.pbs and matrix_operation_mpi.c files. These will be used for job submission and program execution.
2) Access the HPC Cluster:
   a) Use a VPN to establish a secure connection to the Trento University network
   b) Open your SSH client and connect to the cluster with the following command:
     ssh username@hpc.unitn.it
Enter your university credentials when prompted.
3) Upload Files: Navigate to your desired directory on the cluster or create a new one.
Upload commands.pbs and matrix_operation_mpi.c to the chosen directory using the file transfer feature of your SSH client (e.g., drag-and-drop in MobaXterm or scp for Linux/macOS).
4) Reserve a Node and Enter an Interactive Session:
   a) Move to the directory containing your files:
   cd path_to_your_directory
   b) Request an interactive session on a node with the following specifications: 64 cores, 64
    MPI processes and 1 MB of memory. Submit the request using:
    [username@hpc-head-n1 ~]$ qsub -I commands.pbs
5) Modify the Program Input (Optional): Once inside the node, you can modify the commands.pbs file to adjust the matrix size and the number of processes used by the program. Open the file in a text editor, and locate the section where these parameters are defined. Replace the default settings (64) processes and a matrix size of (4096) with your desired values.
    mpirun -np 64 ./matrix_operation_mpi.o 4096 >> $OUTPUT_FILE
6) To execute code without file commands.pbs:
a) Load the required module (mpich-3.2.1--gcc-9.1.0) using the appropriate command:
module load mpich-3.2.1--gcc-9.1.0
b) Compile the program:
mpicc -fopenmp matrix_operation_mpi.c -o matrix_operation_mpi.o
c)  Execute the program, specifying the number of processes (num_process) and the matrix size (size_matrix):
mpirun -np num_process ./matrix_operation_mpi.o size_matrix
8) Submit the job to the queue for compile and execution:
   [username@hpc-head-n1 ~]$ qsub commands.pbs
9) View the Results: once the program completes, its output will be saved in a file named results.txt in the same directory.

# -View the paper
Download the Parallelizing_matrix_operations_using_MPI.pdf

