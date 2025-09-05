// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>
#include <array>
#include <cassert>
#include <iostream>

using namespace gendil;

// Test ConstexprLoop<End> == ConstexprFor<0, End>
constexpr int test_constexpr_loop()
{
  int sum = 0;
  ConstexprLoop<4>([&](auto i) {
    sum += i; // 0 + 1 + 2 + 3 = 6
  });
  return sum;
}

// Test with lambda accepting auto (template param version)
void test_template_lambda()
{
  std::array<int, 3> out = {};
  ConstexprLoop<3>([&](auto i) {
    out[i] = i + 10;
  });
  assert((out == std::array<int, 3>{10, 11, 12}));
}

// Main function to aggregate tests
int main()
{
  static_assert(test_constexpr_loop() == 6, "ConstexprLoop test failed");

  test_template_lambda();

  std::cout << "All ConstexprFor/ConstexprLoop tests passed.\n";
  return 0;
}
