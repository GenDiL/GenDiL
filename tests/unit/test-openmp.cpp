// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <iostream>
#include <omp.h>

#include <gendil/gendil.hpp>

int main(int argc, char** argv) {

    int nthreads;
    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            std::cout << "Number of threads: " << nthreads << std::endl;
        }
    }

    return 0;
}
