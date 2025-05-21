// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <iostream>

#include <gendil/gendil.hpp>

int main(int argc, char** argv) {

#ifdef GENDIL_USE_MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from MPI rank " << rank 
              << " out of " << size << " processes.\n";

    // Simple MPI communication test (rank 0 sends to rank 1)
    if(size >= 2) {
        if(rank == 0) {
            double send_data = 3.14159;
            MPI_Send(&send_data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            std::cout << "Rank 0 sent data: " << send_data << " to Rank 1\n";
        } else if(rank == 1) {
            double recv_data = 0.0;
            MPI_Recv(&recv_data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Rank 1 received data: " << recv_data << " from Rank 0\n";
        }
    }

    MPI_Finalize();
#else
    std::cout << "GenDiL MPI support is disabled. Enable with -DGENDIL_USE_MPI.\n";
#endif

    return 0;
}