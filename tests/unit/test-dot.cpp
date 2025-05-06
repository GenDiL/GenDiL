// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

using namespace gendil;

int main(int argc, char ** argv)
{
    constexpr Integer dim = 3;
    std::array< Real, dim > u = { 1.0, 2.0, 3.0 };
    std::array< Real, dim > v = { 4.0, 5.0, 6.0 };
    const Real dot_product = Dot( u, v );
    std::cout << "Dot product: " << dot_product << std::endl;
    if (dot_product == 32.0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
