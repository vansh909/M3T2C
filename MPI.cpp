#include <iostream>
#include <cstdlib>
#include <chrono>
#include <time.h>
#include "mpi.h"

using namespace std::chrono;
using namespace std;

// Variables
const int MATRIX_SIZE = 400;
int **MatrixA, **MatrixB, **MatrixC;

// Functions

// Function to initialize the matrices and fill it with random number
void init_matrix(int **&matrix, int rows, int cols, bool initialize);

// Function for the head to do the calculation
void head(int total_processes);

// Function for the node to do the calculation
void node(int total_processes, int rank);

// Function to multiply matrix A and matrix B
void multiply(int** matrixA, int** matrixB, int** matrixC, int num_rows);

// Function to print the matrix
void print_matrix(int** matrix, int rows, int cols);

int main(int argc, char **argv)
{
    srand(time(0));

    int total_processes; // Total number of processes
    int rank;            // Rank of each of the processes

    // Initializing the MPI Environment
    MPI_Init(&argc, &argv);

    // Determines the total number of processes running in parallel
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    // Determines the rank of the calling processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) head(total_processes);
    
    else node(total_processes, rank);

    // Terminate the MPI Environment
    MPI_Finalize();
}

void head(int total_processes)
{
    init_matrix(MatrixA, MATRIX_SIZE, MATRIX_SIZE, true);
    init_matrix(MatrixB, MATRIX_SIZE, MATRIX_SIZE, true);
    init_matrix(MatrixC, MATRIX_SIZE, MATRIX_SIZE, false);

    // Calculate the decomposition
    int num_rows = MATRIX_SIZE / total_processes;  // Number of rows per process
    int broadcast_size = (MATRIX_SIZE * MATRIX_SIZE);  // Number of elements to broadcast
    int scatter_gather_size = (MATRIX_SIZE * MATRIX_SIZE) / total_processes; // Number of elements to scatter

    // Starts the timer
    auto start = high_resolution_clock::now();

    // Start distributing the data
    // Scatter matrix A to all the nodes
    MPI_Scatter(&MatrixA[0][0], scatter_gather_size, MPI_INT, &MatrixA, 0, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast matrix B to all the nodes
    MPI_Bcast(&MatrixB[0][0], broadcast_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply matrix A and matrix B into matrix C
    multiply(MatrixA, MatrixB, MatrixC, num_rows);

    // Wait and gather all nodes, then put M3 data into main
    MPI_Gather(MPI_IN_PLACE, scatter_gather_size, MPI_INT, &MatrixC[0][0], scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Get the current time when it's stopped
    auto stop = high_resolution_clock::now();

    // Returns the time it takes for the function to run
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Execution time(MPI): " << duration.count() << " Microseconds" << endl;
}

void node(int total_processes, int rank)
{
    // Calculate the decomposition
    int num_rows = MATRIX_SIZE / total_processes;  // Number of rows per process
    int broadcast_size = (MATRIX_SIZE * MATRIX_SIZE);  // Number of elements to broadcast
    int scatter_gather_size = (MATRIX_SIZE * MATRIX_SIZE) / total_processes; // Number of elements to scatter

    // Initialize arrays which will receive MatrixA, MatrixB, and MatrixC
    init_matrix(MatrixA, MATRIX_SIZE, MATRIX_SIZE, true);
    init_matrix(MatrixB, MATRIX_SIZE, MATRIX_SIZE, false);
    init_matrix(MatrixC, MATRIX_SIZE, MATRIX_SIZE, false);

    // Receive MatrixA and MatrixB from the head
    MPI_Scatter(NULL, scatter_gather_size, MPI_INT, &MatrixA[0][0], scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MatrixB[0][0], broadcast_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply the given matrices
    multiply(MatrixA, MatrixB, MatrixC, num_rows);

    // Gather the calculated matrix
    MPI_Gather(&MatrixC[0][0], scatter_gather_size, MPI_INT, NULL, scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);

}

void init_matrix(int **&matrix, int rows, int cols, bool initialize)
{   
    // Allocate the memory to the arrays
    matrix = (int **) malloc(sizeof(int*) *rows * cols);
    int* temp_matrix = (int *) malloc(sizeof(int) * cols * rows);

    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        matrix[i] = &temp_matrix[i * cols];
    }

    if (!initialize) return;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = rand() % 100;
        }
    }
}   

// Function to multiply matrix A and matrix B
void multiply(int** matrixA, int** matrixB, int** matrixC, int num_rows)
{
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            matrixC[i][j] = 0;

            for (int k = 0; k < MATRIX_SIZE; ++k)
            {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void print_matrix(int** matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("--------------\n");
}
