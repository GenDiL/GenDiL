// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil
{

template <typename T, T... ints>
void print(std::integer_sequence<T, ints...> int_seq)
{
   ((std::cout << ints << ' '), ...);
}

}
