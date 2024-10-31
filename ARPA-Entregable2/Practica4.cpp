#include <mpi.h>
#include <iostream>

using namespace std;

constexpr int RANK_MASTER = 0;
constexpr int TAM_MATRIX = 4;   // Tamaño de la matriz
constexpr int NUM_DATOS = 1;    // Número de datos a enviar
constexpr int TAG = 0;          // Etiqueta del mensaje

struct Matrix {
    int data[TAM_MATRIX][TAM_MATRIX];
};

static void generateMatrix(Matrix& matrix) {
	srand(time(NULL));
	for (int i = 0; i < TAM_MATRIX; ++i) {
		for (int j = 0; j < TAM_MATRIX; ++j) {
			// Números del 1 al 9, para que no parezca que los procesos 
            // no cogen los datos correctamente por los 0.
			matrix.data[i][j] = rand() % 9 + 1;
		}
	}
}

static void printLine() {
    printf("+");
    for (int j = 0; j < TAM_MATRIX - 1; ++j) {
        printf("----+");
    }
    printf("----+\n");
}

static void printMatrix(Matrix matrix) {
    // Imprimir la línea superior
    printLine();

    for (int i = 0; i < TAM_MATRIX; ++i) {
        // Imprimir los valores de la fila
        printf("|");
        for (int j = 0; j < TAM_MATRIX; ++j) {
            printf(" %2d |", matrix.data[i][j]);
        }
        printf("\n");

        // Imprimir las líneas intermedias e inferior de la fila
        printLine();
    }
}

int main(int argc, char* argv[]) {
    int rank, size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 3) {
		if (rank == RANK_MASTER) {
			printf("El número de procesos debe ser mínimo de 3.\n");
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

    Matrix matrix{};

    if (rank == RANK_MASTER) {
		generateMatrix(matrix);
		printf("Matriz original:\n");
		printMatrix(matrix);
    }

    MPI_Datatype upper_triangle, lower_triangle;

    // Definir el tipo de dato para la matriz triangular superior
    int block_lengths_upper[TAM_MATRIX]{};
    MPI_Aint displacements_upper[TAM_MATRIX]{};
    MPI_Aint base_address;

    MPI_Get_address(&matrix.data[0][0], &base_address);

    for (int i = 0; i < TAM_MATRIX; ++i) {
        block_lengths_upper[i] = TAM_MATRIX - i;
        MPI_Aint address;
        MPI_Get_address(&matrix.data[i][i], &address);
        displacements_upper[i] = address - base_address;
    }

    MPI_Type_create_hindexed(TAM_MATRIX, block_lengths_upper, displacements_upper, MPI_INT, &upper_triangle);
    MPI_Type_commit(&upper_triangle);

    // Definir el tipo de dato para la matriz triangular inferior
    int block_lengths_lower[TAM_MATRIX]{};
    MPI_Aint displacements_lower[TAM_MATRIX]{};

    for (int i = 0; i < TAM_MATRIX; ++i) {
        block_lengths_lower[i] = i + 1;
        MPI_Aint address;
        MPI_Get_address(&matrix.data[i][0], &address);
        displacements_lower[i] = address - base_address;
    }

    MPI_Type_create_hindexed(TAM_MATRIX, block_lengths_lower, displacements_lower, MPI_INT, &lower_triangle);
    MPI_Type_commit(&lower_triangle);

    if (rank == RANK_MASTER) {
        MPI_Send(&matrix, NUM_DATOS, upper_triangle, 1, TAG, MPI_COMM_WORLD);
        MPI_Send(&matrix, NUM_DATOS, lower_triangle, 2, TAG, MPI_COMM_WORLD);
    }
    else {
		printf("Matriz antes de recibir los datos en el proceso %d:\n", rank);
		printMatrix(matrix);
        if (rank == 1) {
            MPI_Recv(&matrix, NUM_DATOS, upper_triangle, RANK_MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 2) {
            MPI_Recv(&matrix, NUM_DATOS, lower_triangle, RANK_MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
		printf("Matriz después de recibir los datos en el proceso %d:\n", rank);
        printMatrix(matrix);
    }

    MPI_Type_free(&upper_triangle);
    MPI_Type_free(&lower_triangle);
    MPI_Finalize();
    return 0;
}
