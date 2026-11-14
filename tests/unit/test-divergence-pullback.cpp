// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

namespace divergence_pullback_test
{
using namespace gendil;
using Coordinates = std::array<Real, 2>;
using Matrix = std::array<Coordinates, 2>;
using QuadIndex = TensorIndex<2>;

struct NeighborFace
{
   IdentityOrientation<2> orientation;
   GENDIL_HOST_DEVICE
   const auto& GetOrientation() const { return orientation; }
};

template<int Sign>
using GeometryFace = LocalFaceInfo<Empty, NeighborFace,
   std::integral_constant<int, Sign>>;

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
   GENDIL_HOST_DEVICE
   Real& operator()(Integer, Integer, Integer direction) { return values[direction]; }
   GENDIL_HOST_DEVICE
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

GENDIL_HOST_DEVICE
bool Close(Real actual, Real expected, const char* label)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(actual-expected) > 1e-12*(1+std::abs(expected)))
   {
#ifndef GENDIL_DEVICE_CODE
      std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
#endif
      return false;
   }
   return true;
}

GENDIL_HOST_DEVICE
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

template<class Channel, class Pullback, class Point, class Fields, class Entity = Empty>
GENDIL_HOST_DEVICE
bool CheckPhysical(const Pullback& pullback, const Point& point,
                   const Fields& fields, Real seed, const char* label, const Entity& entity = {})
{
   const auto value = pullback.template Get<Channel>()(
      TestKernel{}, Empty{}, Empty{}, entity, point, fields);
   bool ok = true;
   for (size_t c = 0; c < 2; ++c)
   for (size_t d = 0; d < 2; ++d)
      ok = Close(GetValue(value, c, d), c == d ? seed : 0, label) && ok;
   return ok;
}

template<class Pullback, class Point, class Fields, class Entity = Empty>
GENDIL_HOST_DEVICE
bool CheckCurrent(const Pullback& pullback, const Point& point,
                  const Fields& fields, Real seed, const Matrix& inverse,
                  const char* label, const Entity& entity = {})
{
   static_assert(Pullback::template Contains<GradientChannel>());
   static_assert(!Pullback::template Contains<ValueChannel>());
   VectorRow output;
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, entity, point, fields,
                        pullback, QuadIndex{}, output);
   return CheckPhysical<GradientChannel>(pullback, point, fields, seed, label, entity) &&
          CheckMapped(output, seed, inverse, label);
}

template<class Pullback, class Point, class Entity = Empty>
GENDIL_HOST_DEVICE
bool CheckInactive(const Pullback& pullback, const Point& point, const Entity& entity = {})
{
   static_assert(!Pullback::template Contains<ValueChannel>());
   static_assert(!Pullback::template Contains<GradientChannel>());
   VectorRow output;
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, entity, point, Empty{},
                        pullback, QuadIndex{}, output);
   const Matrix actual{std::get<0>(output.gradients).values,
                       std::get<1>(output.gradients).values};
   bool ok = true;
   for (size_t c = 0; c < 2; ++c)
   for (size_t d = 0; d < 2; ++d)
      ok = Close(actual[c][d], -999, "inactive local divergence trace") && ok;
   return ok;
}

template<int Sign>
auto MakeLocalDivergencePullbacks()
{
   // DSL construction stays on the host; kernels evaluate the captured pullbacks.
   const VectorTestSpace<"v"> v;
   return std::tuple{
      MakeLocalInteriorFacetPullback<Sign>(
         integrate(InteriorFacets<"mesh">{}, 3*average(div(v)))),
      MakeLocalInteriorFacetPullback<Sign>(
         integrate(InteriorFacets<"mesh">{}, 3*jump(div(v)))),
      MakeLocalInteriorFacetPullback<Sign>(
         integrate(InteriorFacets<"mesh">{}, 2*minus(div(v)))),
      MakeLocalInteriorFacetPullback<Sign>(
         integrate(InteriorFacets<"mesh">{}, 5*plus(div(v))))};
}

template<int Sign, class Pullbacks>
GENDIL_HOST_DEVICE
bool CheckLocalDivergence(Real entry, const Pullbacks& pullbacks)
{
   // The output always belongs to current, regardless of normal orientation.
   // Unequal, nondiagonal maps catch accidentally using the neighbor test map.
   const Matrix current_inverse{{{entry, 3}, {5, 7}}};
   const Matrix neighbor_inverse{{{11, 13}, {17, 19}}};
   LocalFacetQuadraturePointData<QuadIndex, Coordinates, Matrix, Matrix, Coordinates> raw{};
   raw.inv_J_mesh_current = current_inverse;
   raw.inv_J_mesh_neighbor = neighbor_inverse;
   raw.weight = 2;
   raw.det_J_facet = 5;
   const LocalFacetContext facet{GeometryFace<Sign>{}};
   const auto& point = raw;
   const auto& average_pullback = std::get<0>(pullbacks);
   const auto& jump_pullback = std::get<1>(pullbacks);
   const auto& minus_pullback = std::get<2>(pullbacks);
   const auto& plus_pullback = std::get<3>(pullbacks);
   bool ok = CheckCurrent(average_pullback, point, Empty{}, 15, current_inverse,
      "oriented local average divergence", facet);
   ok = CheckCurrent(jump_pullback, point, Empty{}, Sign*30, current_inverse,
      "oriented local jump divergence", facet) && ok;
   if constexpr (Sign == 1)
   {
      ok = CheckCurrent(minus_pullback, point, Empty{}, 20, current_inverse,
         "local minus divergence for normal orientation +1", facet) && ok;
      return CheckInactive(plus_pullback, point, facet) && ok;
   }
   else
   {
      ok = CheckCurrent(plus_pullback, point, Empty{}, 50, current_inverse,
         "local plus divergence for normal orientation -1", facet) && ok;
      return CheckInactive(minus_pullback, point, facet) && ok;
   }
}

using RuntimeFace = LocalFaceInfo<Empty, Empty, int>;

template<class MinusPullbacks, class PlusPullbacks>
GENDIL_HOST_DEVICE
bool CheckLocalDivergence(Real entry, int sign,
                         const MinusPullbacks& current_minus,
                         const PlusPullbacks& current_plus)
{
   bool ok = false;
   WithLocalFacetNormalOrientation(RuntimeFace{{}, {}, sign}, [&](const auto& face)
   {
      if constexpr (decltype(face.GetNormalOrientation())::value == 1)
         ok = CheckLocalDivergence<1>(entry, current_minus);
      else
         ok = CheckLocalDivergence<-1>(entry, current_plus);
   });
   return ok;
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
   LocalFacetQuadraturePointData<QuadIndex, Coordinates, Matrix, Matrix, Coordinates> local_data{};
   local_data.inv_J_mesh_current = minus_inverse;
   local_data.inv_J_mesh_neighbor = plus_inverse;
   local_data.weight = 2;
   local_data.det_J_facet = 5;
   const LocalFacetContext local_facet{GeometryFace<1>{}};
   const auto& local = local_data;

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
   ok = CheckCurrent(local_average, local, Empty{}, 15, minus_inverse, "local average divergence", local_facet) && ok;
   ok = CheckCurrent(local_jump, local, Empty{}, 30, minus_inverse, "local jump divergence", local_facet) && ok;
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
} // namespace divergence_pullback_test

int main()
{
   using namespace divergence_pullback_test;
   bool ok = CheckDivergenceChannels();
   const auto current_minus = MakeLocalDivergencePullbacks<1>();
   const auto current_plus = MakeLocalDivergencePullbacks<-1>();
   ok = CheckLocalDivergence<1>(2.25, current_minus) &&
      CheckLocalDivergence<-1>(-1.5, current_plus) &&
      CheckLocalDivergence(2.25, 1, current_minus, current_plus) &&
      CheckLocalDivergence(-1.5, -1, current_minus, current_plus) && ok;
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   Vector input(2), result(2);
   auto* input_host = input.WriteHostData();
   input_host[0] = -1.;
   input_host[1] = 1.;
   const auto* signs = input.ReadDeviceData();
   auto* values = result.WriteDeviceData();
   Policy::CandidateBlockLoop(2, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(2))
      {
         const Real entry = .25 + signs[i];
         values[i] = CheckLocalDivergence<1>(entry, current_minus) &&
            CheckLocalDivergence<-1>(entry, current_plus) &&
            CheckLocalDivergence(entry, static_cast<int>(signs[i]), current_minus, current_plus);
      }
   });
   GENDIL_DEVICE_SYNC;
   const auto* host = result.ReadHostData();
   const bool device_ok = host[0] == 1. && host[1] == 1.;
   if (!device_ok) { std::cerr << "Device local divergence pullback failed.\n"; }
   ok = device_ok && ok;
#endif
   return ok ? 0 : 1;
}
