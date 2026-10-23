// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

namespace
{
using namespace gendil;
using Coordinates = std::array<Real, 2>;
using Matrix = std::array<Coordinates, 2>;
using QuadIndex = TensorIndex<2>;

struct TestKernel
{
   template<Integer Dim>
   using register_dimensions = std::make_index_sequence<Dim>;
};

// One quadrature point per component is sufficient to exercise the actual
// gradient reader, physical mapping, and component-wise quadrature writer.
struct GradientTensor
{
   Coordinates values{-999, -999};
   Real& operator()(Integer, Integer, Integer direction) { return values[direction]; }
   Real operator()(Integer, Integer, Integer direction) const { return values[direction]; }
};

struct VectorRow
{
   Empty values;
   std::tuple<GradientTensor, GradientTensor> gradients;
};

VectorRow Field(const Matrix& gradient)
{
   return {{}, {GradientTensor{gradient[0]}, GradientTensor{gradient[1]}}};
}

bool Close(Real actual, Real expected, const char* label)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(actual-expected) > 1e-12*(1+std::abs(expected)))
   {
      std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
      return false;
   }
   return true;
}

bool CheckMapped(const VectorRow& row, Real seed, const Matrix& inverse,
                 const char* label)
{
   bool ok = true;
   const Matrix actual{std::get<0>(row.gradients).values,
                       std::get<1>(row.gradients).values};
   // A physical scalar seed s for div(v) is s*I. Applying the gradient
   // transpose mapping gives row c, reference direction d = s*inv_J[c][d].
   for (size_t c = 0; c < 2; ++c)
   for (size_t d = 0; d < 2; ++d)
      ok = Close(actual[c][d], seed*inverse[c][d], label) && ok;
   return ok;
}

template<class Channel, class Pullback, class Point, class Fields>
bool CheckPhysical(const Pullback& pullback, const Point& point,
                   const Fields& fields, Real seed, const char* label)
{
   const auto value = pullback.template Get<Channel>()(
      TestKernel{}, Empty{}, Empty{}, Empty{}, point, fields);
   bool ok = true;
   for (size_t c = 0; c < 2; ++c)
   for (size_t d = 0; d < 2; ++d)
      ok = Close(GetValue(value, c, d), c == d ? seed : 0, label) && ok;
   return ok;
}

template<class Pullback, class Point, class Fields>
bool CheckCurrent(const Pullback& pullback, const Point& point,
                  const Fields& fields, Real seed, const Matrix& inverse,
                  const char* label)
{
   static_assert(Pullback::template Contains<GradientChannel>());
   static_assert(!Pullback::template Contains<ValueChannel>());
   VectorRow output;
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, point, fields,
                        pullback, QuadIndex{}, output);
   return CheckPhysical<GradientChannel>(pullback, point, fields, seed, label) &&
          CheckMapped(output, seed, inverse, label);
}

template<class Pullback, class Point, class Fields>
bool CheckBothSides(const Pullback& pullback, const Point& point,
                    const Fields& fields, Real minus_seed, Real plus_seed,
                    const Matrix& minus_inverse, const Matrix& plus_inverse,
                    const char* label)
{
   static_assert(Pullback::template Contains<GradientMinusChannel>());
   static_assert(Pullback::template Contains<GradientPlusChannel>());
   static_assert(!Pullback::template Contains<ValueMinusChannel>());
   static_assert(!Pullback::template Contains<ValuePlusChannel>());
   VectorRow minus_output, plus_output;
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, point, fields,
                        pullback, QuadIndex{}, minus_output, plus_output);
   bool ok = CheckPhysical<GradientMinusChannel>(pullback, point, fields, minus_seed, label);
   ok = CheckPhysical<GradientPlusChannel>(pullback, point, fields, plus_seed, label) && ok;
   ok = CheckMapped(minus_output, minus_seed, minus_inverse, label) && ok;
   return CheckMapped(plus_output, plus_seed, plus_inverse, label) && ok;
}

bool CheckDivergenceChannels()
{
   const Matrix minus_inverse{{{2, 3}, {5, 7}}};
   const Matrix plus_inverse{{{11, 13}, {17, 19}}};
   QuadraturePointContext<QuadIndex, Coordinates, Matrix> cell{};
   cell.inv_J_mesh = minus_inverse;
   cell.weight = 2;
   cell.det_J = 3;
   BoundaryFacetQuadraturePointContext<QuadIndex, Coordinates, Matrix, Coordinates> boundary{};
   boundary.inv_J_mesh = minus_inverse;
   boundary.weight = 2;
   boundary.det_J_facet = 5;
   TwoSidedFacetQuadraturePointContext<QuadIndex, Coordinates, Matrix, Matrix, Coordinates> interior{};
   interior.inv_J_mesh_minus = minus_inverse;
   interior.inv_J_mesh_plus = plus_inverse;
   interior.weight = 2;
   interior.det_J_facet = 5;

   const VectorTestSpace<"v"> v;
   const auto cell_pullback = MakeCellPullback(integrate(Cells<"mesh">{}, 3*div(v)));
   const auto boundary_pullback = MakeBoundaryFacetPullback(
      integrate(BoundaryFacets<"mesh">{}, 3*div(v)));
   const auto boundary_average = MakeBoundaryFacetPullback(
      integrate(BoundaryFacets<"mesh">{}, 3*average(div(v))));
   const auto boundary_jump = MakeBoundaryFacetPullback(
      integrate(BoundaryFacets<"mesh">{}, 3*jump(div(v))));
   const auto local_average = MakeLocalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 3*average(div(v))));
   const auto local_jump = MakeLocalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 3*jump(div(v))));
   const auto global_average = MakeGlobalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 3*average(div(v))));
   const auto global_jump = MakeGlobalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 3*jump(div(v))));

   bool ok = CheckCurrent(cell_pullback, cell, Empty{}, 18, minus_inverse, "cell divergence");
   ok = CheckCurrent(boundary_pullback, boundary, Empty{}, 30, minus_inverse, "boundary divergence") && ok;
   ok = CheckCurrent(boundary_average, boundary, Empty{}, 30, minus_inverse, "boundary average divergence") && ok;
   ok = CheckCurrent(boundary_jump, boundary, Empty{}, 30, minus_inverse, "boundary jump divergence") && ok;
   ok = CheckCurrent(local_average, interior, Empty{}, 15, minus_inverse, "local average divergence") && ok;
   ok = CheckCurrent(local_jump, interior, Empty{}, 30, minus_inverse, "local jump divergence") && ok;
   ok = CheckBothSides(global_average, interior, Empty{}, 15, 15,
      minus_inverse, plus_inverse, "global average divergence") && ok;
   ok = CheckBothSides(global_jump, interior, Empty{}, 30, -30,
      minus_inverse, plus_inverse, "global jump divergence") && ok;

   const auto explicit_minus = MakeGlobalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 2*minus(div(v))));
   const auto explicit_plus = MakeGlobalInteriorFacetPullback(
      integrate(InteriorFacets<"mesh">{}, 5*plus(div(v))));
   static_assert(decltype(explicit_minus)::template Contains<GradientMinusChannel>());
   static_assert(!decltype(explicit_minus)::template Contains<GradientPlusChannel>());
   static_assert(!decltype(explicit_plus)::template Contains<GradientMinusChannel>());
   static_assert(decltype(explicit_plus)::template Contains<GradientPlusChannel>());
   InterpolatedField<Empty, Empty> inactive;
   VectorRow minus_output, plus_output;
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, interior, Empty{},
                        explicit_minus, QuadIndex{}, minus_output, inactive);
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, interior, Empty{},
                        explicit_plus, QuadIndex{}, inactive, plus_output);
   ok = CheckPhysical<GradientMinusChannel>(explicit_minus, interior, Empty{}, 20, "explicit minus seed") && ok;
   ok = CheckPhysical<GradientPlusChannel>(explicit_plus, interior, Empty{}, 50, "explicit plus seed") && ok;
   ok = CheckMapped(minus_output, 20, minus_inverse, "explicit minus divergence") && ok;
   ok = CheckMapped(plus_output, 50, plus_inverse, "explicit plus divergence") && ok;

   const VectorTrialSpace<"u"> u;
   const FiniteElementField<"w"> w;
   const auto minus_field_values = MakeStaticMap(
      MakeEntry<NameTag<"u">>(Field(Matrix{{{1, 2}, {3, 4}}})),
      MakeEntry<NameTag<"w">>(Field(Matrix{{{-2, 5}, {7, -3}}})));
   const auto plus_field_values = MakeStaticMap(
      MakeEntry<NameTag<"u">>(Field(Matrix{{{-1, 3}, {2, -4}}})),
      MakeEntry<NameTag<"w">>(Field(Matrix{{{4, -2}, {-3, 6}}})));
   struct FacetFields
   {
      decltype(minus_field_values) minus_fields;
      decltype(plus_field_values) plus_fields;
   } fields{minus_field_values, plus_field_values};
   auto evaluate = [&](const auto& expression, const auto& point, const auto& field_values)
   {
      return expression(TestKernel{}, Empty{}, Empty{}, Empty{}, point, field_values);
   };
   // Explicit physical partials: minus u = (8,43), plus u = (28,-42);
   // minus w = (11,14), plus w = (18,63). Taking the reference trace,
   // transposing the gradient, or reusing the minus Jacobian gives other values.
   ok = Close(evaluate(div(u), cell, minus_field_values), 51, "trial physical divergence") && ok;
   ok = Close(evaluate(div(w), cell, minus_field_values), 25, "bound physical divergence") && ok;
   ok = Close(evaluate(div(w), boundary, minus_field_values), 25, "bound boundary divergence") && ok;
   ok = Close(evaluate(minus(div(u)), interior, fields), 51, "minus trial divergence") && ok;
   ok = Close(evaluate(plus(div(u)), interior, fields), -14, "plus trial divergence") && ok;
   ok = Close(evaluate(average(div(u)), interior, fields), 18.5, "average trial divergence") && ok;
   ok = Close(evaluate(jump(div(u)), interior, fields), 65, "jump trial divergence") && ok;
   ok = Close(evaluate(minus(div(w)), interior, fields), 25, "minus bound divergence") && ok;
   ok = Close(evaluate(plus(div(w)), interior, fields), 81, "plus bound divergence") && ok;
   ok = Close(evaluate(average(div(w)), interior, fields), 53, "average bound divergence") && ok;
   ok = Close(evaluate(jump(div(w)), interior, fields), -56, "jump bound divergence") && ok;

   const auto trial_form = MakeGlobalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{},
      average(div(u))*average(div(v))));
   const auto bound_form = MakeGlobalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{},
      average(div(w))*average(div(v))));
   ok = CheckBothSides(trial_form, interior, fields, 92.5, 92.5,
      minus_inverse, plus_inverse, "global trial/test divergence") && ok;
   return CheckBothSides(bound_form, interior, fields, 265, 265,
      minus_inverse, plus_inverse, "global bound/test divergence") && ok;
}
} // namespace

int main()
{
   return CheckDivergenceChannels() ? 0 : 1;
}
