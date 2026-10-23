// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>
#include <array>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{
template<class L, class R>
concept HasDot = requires(const L& lhs, const R& rhs) { Dot(lhs, rhs); };

using Recursive2 = SerialRecursiveArray<Real, 2>;
using RecursiveMatrix = SerialRecursiveArray<Real, 2, 2>;
using Array2 = std::array<Real, 2>;
using Array3 = std::array<Real, 3>;
using ArrayMatrix = std::array<Array2, 2>;
static_assert(HasDot<Recursive2, Array2> && HasDot<Array2, Recursive2>);
static_assert(!HasDot<Recursive2, Array3> && !HasDot<Array3, Recursive2>);
static_assert(!HasDot<RecursiveMatrix, Array2> && !HasDot<Array2, RecursiveMatrix>);
static_assert(!HasDot<Recursive2, ArrayMatrix> && !HasDot<ArrayMatrix, Recursive2>);
static_assert(!HasDot<RecursiveMatrix, ArrayMatrix> && !HasDot<ArrayMatrix, RecursiveMatrix>);

bool CheckMixedStorage()
{
   SerialRecursiveArray<double, 2> two;
   two(0) = 1.5; two(1) = -2;
   const std::array<float, 2> weights{2, 0.25f};
   static_assert(std::is_same_v<decltype(Dot(two, weights)), double>);
   static_assert(std::is_same_v<decltype(Dot(weights, two)), double>);
   if (Dot(two, weights) != 2.5 || Dot(weights, two) != 2.5) { return false; }

   SerialRecursiveArray<int, 3> three;
   three(0) = 1; three(1) = -2; three(2) = 3;
   const std::array<float, 3> fractions{0.5f, 1.25f, -0.5f};
   // The scalar product determines the result type, rather than either input
   // container's element type or an unconditional conversion to Real.
   static_assert(std::is_same_v<decltype(Dot(three, fractions)), float>);
   static_assert(std::is_same_v<decltype(Dot(fractions, three)), float>);
   return Dot(three, fractions) == -3.5f && Dot(fractions, three) == -3.5f;
}

bool CheckExistingOverloads()
{
   const std::array<Real, 3> u{1, 2, 3}, v{4, 5, 6};
   const Real raw_u[]{1, 2, 3}, raw_v[]{4, 5, 6};
   const std::array<Real, 0> empty{};
   if (Dot(u, v) != 32 || Dot(raw_u, raw_v) != 32 ||
       Dot(raw_u, v) != 32 || Dot(empty, empty) != 0) { return false; }

   SerialRecursiveArray<Real, 3> recursive_u, recursive_v;
   for (size_t i = 0; i < 3; ++i)
   {
      recursive_u(i) = u[i]; recursive_v(i) = v[i];
   }
   static_assert(std::is_same_v<decltype(Dot(recursive_u, recursive_v)), Real>);
   if (Dot(recursive_u, recursive_v) != 32) { return false; }

   // Existing recursive-array Dot contracts every entry, including rank two.
   SerialRecursiveArray<Real, 2, 3> matrix;
   for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 3; ++j) { matrix(i, j) = Real(3 * i + j + 1); }
   if (Dot(matrix, matrix) != 91) { return false; }

   Vector x(3), y(3);
   auto* x_values = x.WriteHostData();
   auto* y_values = y.WriteHostData();
   for (size_t i = 0; i < 3; ++i) { x_values[i] = u[i]; y_values[i] = v[i]; }
   const auto borrowed = MakeVectorView(std::as_const(y), size_t{0}, size_t{3});
   static_assert(HostReadableVector<decltype(borrowed)>);
   return Dot(x, y) == 32 && Dot(x, borrowed) == 32 &&
          Dot(borrowed, x) == 32 && Dot(borrowed, borrowed) == 77;
}
} // namespace

int main()
{
   if (!CheckMixedStorage())
   {
      std::cerr << "Mixed-storage Dot values or promotion are incorrect\n";
      return 1;
   }
   if (!CheckExistingOverloads())
   {
      std::cerr << "An existing Dot overload changed behavior\n";
      return 2;
   }
   std::cout << "Dot preserves scalar products across supported vector storage\n";
   return 0;
}
