// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <array>
#include <iostream>
#include <type_traits>
#include <utility>

namespace
{
using namespace gendil;

template<size_t Dim, size_t... I>
constexpr Point<Dim> CoordinatePoint(const std::array<Real, Dim>& values, std::index_sequence<I...>)
{
   return Point<Dim>{values[I]...};
}

template<size_t Dim, size_t... I>
constexpr Point<Dim> ScalarPointPack(const std::array<Real, Dim>& values, std::index_sequence<I...>)
{
   // Tensor-product quadrature assembles multidimensional points this way.
   return Point<Dim>{Point<1>{values[I]}...};
}

template<size_t Dim>
constexpr Point<Dim> AcceptPoint(Point<Dim> point) { return point; }

template<size_t Dim>
constexpr bool CheckPointConstruction(std::array<Real, Dim> values)
{
   const std::array<Real, Dim> constant = values;
   const Point<Dim> direct(values), from_const(constant), temporary{std::array<Real, Dim>(values)};
   const Point<Dim> converted = values;
   const Point<Dim> const_converted = constant;
   const Point<Dim> temporary_converted = std::array<Real, Dim>(values);
   const Point<Dim> copied = direct;
   Point<Dim> to_move = direct;
   const Point<Dim> moved = std::move(to_move);
   const Point<Dim> coordinates = CoordinatePoint(values, std::make_index_sequence<Dim>{});
   const Point<Dim> packed = ScalarPointPack(values, std::make_index_sequence<Dim>{});
   const Point<Dim> argument = AcceptPoint<Dim>(constant);
   const Point<Dim> zero{};
   for (size_t i = 0; i < Dim; ++i)
   {
      if (zero[i] != 0) return false;
      for (const auto& point : {direct, from_const, temporary, converted, const_converted,
                               temporary_converted, copied, moved, coordinates, packed, argument})
         if (point[i] != values[i]) return false;
   }
   return true;
}

static_assert(CheckPointConstruction(std::array<Real, 1>{2.5}));
static_assert(CheckPointConstruction(std::array<Real, 2>{2.5, -1.25}));
static_assert(CheckPointConstruction(std::array<Real, 3>{2.5, -1.25, .75}));
static_assert(Point<2>({1., 2.})[0] == 1. && Point<2>({1., 2.})[1] == 2.);
constexpr Point<1> scalar_point = 3.5;
static_assert(static_cast<Real>(scalar_point) == 3.5);
static_assert(std::is_convertible_v<Point<1>, Real>);
static_assert(!std::is_convertible_v<Point<2>, Real>);
static_assert(!std::is_convertible_v<Point<3>, Real>);

constexpr bool CheckScalarConstruction(int integer, float single, long double extended)
{
   // Preserve the Real-parameter conversions of the former 1D specialization.
   const Point<1> from_integer(integer);
   const Point<1> from_single(single);
   const Point<1> from_extended(extended);
   const Point<1> implicit_integer = integer;
   return from_integer[0] == Real(integer) && from_single[0] == Real(single) &&
          from_extended[0] == Real(extended) && implicit_integer[0] == Real(integer);
}

static_assert(CheckScalarConstruction(2, 1.25f, 2.5L));

template<size_t Dim>
bool CheckMesh(const CartesianMesh<Dim>& mesh,
               const std::array<GlobalIndex, Dim>& sizes,
               const std::array<Real, Dim>& spacing,
               const std::array<Real, Dim>& origin,
               const std::array<int, Dim>& periodic)
{
   GlobalIndex cells = 1;
   bool ok = true;
   for (size_t d = 0; d < Dim; ++d)
   {
      cells *= sizes[d];
      ok = mesh.Size(d) == sizes[d] && mesh.h[d] == spacing[d] &&
         mesh.mesh_origin[d] == origin[d] &&
         mesh.is_periodic_boundary[d] == bool(periodic[d]) && ok;
   }
   ok = mesh.GetNumberOfCells() == cells && ok;
   for (GlobalIndex linear = 0; linear < cells; ++linear)
   {
      const auto cell = mesh.GetCell(linear);
      GlobalIndex remaining = linear;
      for (size_t d = 0; d < Dim; ++d)
      {
         const GlobalIndex index = remaining%sizes[d];
         remaining /= sizes[d];
         ok = cell.origin[d] == origin[d]+spacing[d]*index && cell.h[d] == spacing[d] && ok;
      }
   }
   return ok;
}

template<size_t Dim>
bool CheckMeshOrigins()
{
   std::array<GlobalIndex, Dim> sizes{};
   std::array<Real, Dim> spacing{}, origin{};
   std::array<int, Dim> axis_periodic{};
   for (size_t d = 0; d < Dim; ++d)
   {
      sizes[d] = d+2;
      spacing[d] = .25*(d+1);
      origin[d] = .5-1.25*d;
      axis_periodic[d] = d%2 == 0;
   }
   const std::array<Real, Dim> constant_origin = origin;
   const Point<Dim> point_origin = CoordinatePoint(origin, std::make_index_sequence<Dim>{});
   bool ok = true;
   for (int variant = 0; variant < 3; ++variant)
   {
      std::array<int, Dim> expected_periodic{};
      for (size_t d = 0; d < Dim; ++d)
         expected_periodic[d] = variant == 0 ? 0 : variant == 1 ? 1 : axis_periodic[d];
      const auto check_origin = [&](auto&& supplied_origin)
      {
         const auto mesh = [&]()
         {
            if (variant == 0)
               return CartesianMesh<Dim>(sizes, spacing, std::forward<decltype(supplied_origin)>(supplied_origin));
            if (variant == 1)
               return CartesianMesh<Dim>(sizes, spacing, std::forward<decltype(supplied_origin)>(supplied_origin), true);
            return CartesianMesh<Dim>(sizes, spacing, std::forward<decltype(supplied_origin)>(supplied_origin), axis_periodic);
         }();
         return CheckMesh(mesh, sizes, spacing, origin, expected_periodic);
      };
      ok = check_origin(origin) && ok;
      ok = check_origin(constant_origin) && ok;
      ok = check_origin(std::array<Real, Dim>(origin)) && ok;
      ok = check_origin(point_origin) && ok;
      const auto brace_mesh = [&]<size_t... I>(std::index_sequence<I...>)
      {
         if (variant == 0) return CartesianMesh<Dim>(sizes, spacing, {origin[I]...});
         if (variant == 1) return CartesianMesh<Dim>(sizes, spacing, {origin[I]...}, true);
         return CartesianMesh<Dim>(sizes, spacing, {origin[I]...}, axis_periodic);
      }(std::make_index_sequence<Dim>{});
      ok = CheckMesh(brace_mesh, sizes, spacing, origin, expected_periodic) && ok;
      if (!ok) std::cerr << Dim << "D CartesianMesh origin, constructor variant " << variant << " failed\n";
   }
   return ok;
}
}

int main()
{
   bool ok = CheckPointConstruction(std::array<Real, 1>{-.5}) &&
      CheckPointConstruction(std::array<Real, 2>{-.5, 2.25}) &&
      CheckPointConstruction(std::array<Real, 3>{-.5, 2.25, 4.});
   Point<1> one = 1.25;
   const Real scalar = one;
   const Point<2> nested({1., 2.});
   ok = scalar == 1.25 && nested[0] == 1 && nested[1] == 2 && ok;
   ok = CheckScalarConstruction(-3, .75f, -1.25L) && ok;
   if (!ok) std::cerr << "Point array conversion or existing constructor regression\n";
   ok = CheckMeshOrigins<1>() && ok;
   ok = CheckMeshOrigins<2>() && ok;
   ok = CheckMeshOrigins<3>() && ok;
   return ok ? 0 : 1;
}
