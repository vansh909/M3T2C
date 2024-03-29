#include <iostream>
#include <cstdlib>
#include <chrono>
#include <time.h>
#include <CL/cl.h>
#include "mpi.h"

using namespace std::chrono;
using namespace std;

// Variables
const int MATRIX_SIZE = 400;
int **MatrixA, **MatrixB, **MatrixC;

// OpenCL Variables
cl_mem bufMatrixA, bufMatrixB, bufMatrixC;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err;

// Functions

// Function to initialize the matrices and fill them with random numbers
void initialize_matrix(int **&matrix, int rows, int cols, bool initialize);

// Function for the head to perform the calculation
void head(int total_processes);

// Function for the node to perform the calculation
void node(int total_processes, int rank);

// Function to print the matrix
void print_matrix(int **matrix, int rows, int cols);

// OpenCL Function Prototypes

cl_device_id create_device();

void setup_opencl_device_context_queue_kernel(char *filename, char *kernelname);

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

void setup_kernel_memory(int num_rows);

void copy_kernel_args();

void free_memory();

void start_opencl(int num_rows);

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

    if (rank == 0)
        head(total_processes);
    else
        node(total_processes, rank);

    // Terminate the MPI Environment
    MPI_Finalize();
}

void head(int total_processes)
{
    initialize_matrix(MatrixA, MATRIX_SIZE, MATRIX_SIZE, true);
    initialize_matrix(MatrixB, MATRIX_SIZE, MATRIX_SIZE, true);
    initialize_matrix(MatrixC, MATRIX_SIZE, MATRIX_SIZE, false);

    // Calculate the decomposition
    int num_rows = MATRIX_SIZE / total_processes;                     // Number of rows per process
    int broadcast_size = (MATRIX_SIZE * MATRIX_SIZE);                // Number of elements to broadcast
    int scatter_gather_size = (MATRIX_SIZE * MATRIX_SIZE) / total_processes; // Number of elements to scatter

    // Starts the timer
    auto start = high_resolution_clock::now();

    // Start distributing the data
    // Scatter matrix A to all the nodes
    MPI_Scatter(&MatrixA[0][0], scatter_gather_size, MPI_INT, &MatrixA, 0, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all the nodes
    MPI_Bcast(&MatrixB[0][0], broadcast_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply matrix A and matrix B into MatrixC using OpenCL
    start_opencl(num_rows);

    // Wait and gather all nodes, then put MatrixC data into main
    MPI_Gather(MPI_IN_PLACE, scatter_gather_size, MPI_INT, &MatrixC[0][0], scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Get the current time when it's stopped
    auto stop = high_resolution_clock::now();

    // Returns the time it takes for the function to run
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: " << duration.count() << " Microseconds" << endl;

    // Free the memory for device, kernel, queue, etc
    free_memory();
}

void node(int total_processes, int rank)
{
    // Calculate the decomposition
    int num_rows = MATRIX_SIZE / total_processes;                     // Number of rows per process
    int broadcast_size = (MATRIX_SIZE * MATRIX_SIZE);                // Number of elements to broadcast
    int scatter_gather_size = (MATRIX_SIZE * MATRIX_SIZE) / total_processes; // Number of elements to scatter

    // Initialize arrays which will receive MatrixA, MatrixB, and MatrixC
    initialize_matrix(MatrixA, MATRIX_SIZE, MATRIX_SIZE, true);
    initialize_matrix(MatrixB, MATRIX_SIZE, MATRIX_SIZE, false);
    initialize_matrix(MatrixC, MATRIX_SIZE, MATRIX_SIZE, false);

    // Receive MatrixA and MatrixB from the head
    MPI_Scatter(NULL, scatter_gather_size, MPI_INT, &MatrixA[0][0], scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MatrixB[0][0], broadcast_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply the given matrices using OpenCL
    start_opencl(num_rows);

    // Gather the calculated matrix
    MPI_Gather(&MatrixC[0][0], scatter_gather_size, MPI_INT, NULL, scatter_gather_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Free the memory for device, kernel, queue, etc
    free_memory();
}

void initialize_matrix(int **&matrix, int rows, int cols, bool initialize)
{
    // Allocate memory to the arrays
    matrix = (int **)malloc(sizeof(int *) * rows);
    int *temp_matrix = (int *)malloc(sizeof(int) * rows * cols);

    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        matrix[i] = &temp_matrix[i * cols];
    }

    if (!initialize)
        return;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = rand() % 100;
        }
    }
}

void print_matrix(int **matrix, int rows, int cols)
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

// OpenCL Functions

void free_memory()
{
    //free the buffers
    clReleaseMemObject(bufMatrixA);
    clReleaseMemObject(bufMatrixB);
    clReleaseMemObject(bufMatrixC);

    //free OpenCL objects
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(MatrixA);
    free(MatrixB);
    free(MatrixC);
}

void copy_kernel_args()
{
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&MATRIX_SIZE);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufMatrixA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufMatrixB);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufMatrixC);

    if (err < 0)
       	perror("Couldn't create a kernel argument");
}

void setup_kernel_memory(int num_rows)
{
    bufMatrixA = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * num_rows * sizeof(int), NULL, NULL);
    bufMatrixB = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, NULL);
    bufMatrixC = clCreateBuffer(context, CL_MEM_READ_WRITE, MATRIX_SIZE * num_rows * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufMatrixA, CL_TRUE, 0, MATRIX_SIZE * num_rows * sizeof(int), &MatrixA[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufMatrixB, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), &MatrixB[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufMatrixC, CL_TRUE, 0, MATRIX_SIZE * num_rows * sizeof(int), &MatrixC[0], 0, NULL, NULL);
}

void setup_opencl_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    	perror("Couldn't create a context");

    program = build_program(context, device_id, filename);

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
       	perror("Couldn't create a command queue");

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
       	perror("Couldn't create a kernel");
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
       	perror("Couldn't create the program");
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    	perror("Couldn't identify a platform");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        printf("GPU not found\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    	perror("Couldn't access any devices");

    return dev;
}

void start_opencl(int num_rows)
{
    size_t global[3] = {(size_t) num_rows, (size_t) MATRIX_SIZE, (size_t) MATRIX_SIZE};    

    setup_opencl_device_context_queue_kernel((char *)"./MatrixMultiply.cl", (char *)"matrix_multiply");

    setup_kernel_memory(num_rows);

    copy_kernel_args();

    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);

    clEnqueueReadBuffer(queue, bufMatrixC, CL_TRUE, 0, MATRIX_SIZE * num_rows * sizeof(int), &MatrixC[0], 0, NULL, NULL);
}
