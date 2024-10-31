#include <mpi.h>
#include <iostream>

constexpr int RANK_MASTER = 0;
constexpr int N = 5; // Tamaño de la matriz

struct Matrix {
    int data[N][N];
};

void print_matrix(const char* label, const Matrix& matrix, int rank) {
    std::cout << "Proceso " << rank << " - " << label << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix.data[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix matrix = {};

    if (rank == RANK_MASTER) {
        std::srand(std::time(NULL));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                matrix.data[i][j] = std::rand() % 10;   // Números del 1 al 10
            }
        }

		print_matrix("Matriz original", matrix, rank);
    }

    MPI_Datatype upper_triangle, lower_triangle;

    // Definir el tipo de dato para la matriz triangular superior
    int block_lengths_upper[N];
    MPI_Aint displacements_upper[N];
    MPI_Aint base_address;

    MPI_Get_address(&matrix.data[0][0], &base_address);

    for (int i = 0; i < N; ++i) {
        block_lengths_upper[i] = N - i;
        MPI_Aint address;
        MPI_Get_address(&matrix.data[i][i], &address);
        displacements_upper[i] = address - base_address;
    }

    MPI_Type_create_hindexed(N, block_lengths_upper, displacements_upper, MPI_INT, &upper_triangle);
    MPI_Type_commit(&upper_triangle);

    // Definir el tipo de dato para la matriz triangular inferior
    int block_lengths_lower[N];
    MPI_Aint displacements_lower[N];

    for (int i = 0; i < N; ++i) {
        block_lengths_lower[i] = i + 1;
        MPI_Aint address;
        MPI_Get_address(&matrix.data[i][0], &address);
        displacements_lower[i] = address - base_address;
    }

    MPI_Type_create_hindexed(N, block_lengths_lower, displacements_lower, MPI_INT, &lower_triangle);
    MPI_Type_commit(&lower_triangle);

    if (rank == RANK_MASTER) {
        MPI_Send(&matrix, 1, upper_triangle, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&matrix, 1, lower_triangle, 2, 0, MPI_COMM_WORLD);
    }
    else {
        print_matrix("Matriz antes de recibir", matrix, rank);
        if (rank == 1) {
            MPI_Recv(&matrix, 1, upper_triangle, RANK_MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 2) {
            MPI_Recv(&matrix, 1, lower_triangle, RANK_MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        print_matrix("Matriz después de recibir", matrix, rank);
    }

    MPI_Type_free(&upper_triangle);
    MPI_Type_free(&lower_triangle);
    MPI_Finalize();
    return 0;
}
