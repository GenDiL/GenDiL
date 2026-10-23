// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string_view>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace
{

template<class T>
constexpr std::string_view TypeSignature()
{
#if defined(_MSC_VER)
   return __FUNCSIG__;
#else
   return __PRETTY_FUNCTION__;
#endif
}

template<class T>
void PrintType(const std::string_view label)
{
   std::cout << label << ":\n" << TypeSignature<T>() << "\n\n";
}

bool SignatureContains(
   const std::string_view signature,
   const std::string_view expected)
{
   if (signature.find(expected) != std::string_view::npos)
   {
      return true;
   }

   std::cerr << "Type signature does not contain '" << expected << "'.\n";
   return false;
}

#if defined(GENDIL_USE_DEVICE)
template<Integer NumQuad1D>
using GlobalFaceKernelPolicy =
   DeviceKernelConfiguration<ThreadBlockLayout<NumQuad1D>, 1, 2>;
#else
template<Integer>
using GlobalFaceKernelPolicy = SerialKernelConfiguration;
#endif

} // namespace

int main()
{
   // The right part has twice as many cells along the shared interface, which
   // makes the interface h-nonconforming. The p=1 and p=3 finite elements make
   // the resulting mixed space p-nonconforming as well.
   constexpr GlobalIndex nx_left = 2;
   constexpr GlobalIndex ny_left = 2;
   constexpr GlobalIndex nx_right = 2;
   constexpr GlobalIndex ny_right = 4;
   static_assert(ny_right == 2 * ny_left);

   Cartesian2DMesh left_mesh(
      Real{0.5},
      Real{0.5},
      nx_left,
      ny_left,
      Point<2>{0.0, 0.0});
   Cartesian2DMesh right_mesh(
      Real{0.5},
      Real{0.25},
      nx_right,
      ny_right,
      Point<2>{1.0, 0.0});

   auto left_interior_faces =
      MakeCartesianInteriorFaceConnectivity<2>({nx_left, ny_left});
   auto right_interior_faces =
      MakeCartesianInteriorFaceConnectivity<2>({nx_right, ny_right});
   NonconformingCartesianIntermeshFaceConnectivity<2, 2> interface_faces(
      {nx_left, ny_left},
      {nx_right, ny_right});

   auto partition = MakePartition(
      MakeCellPart(left_mesh),
      MakeCellPart(right_mesh),
      MakeInteriorFacePart<0, 0>(left_interior_faces),
      MakeInteriorFacePart<1, 1>(right_interior_faces),
      MakeInteriorFacePart<0, 1>(interface_faces));

   using PartitionType = std::remove_cvref_t<decltype(partition)>;
   static_assert(PartitionType::num_cell_parts == 2);
   static_assert(PartitionType::num_interior_face_parts == 5);
   static_assert(PartitionType::num_boundary_face_parts == 0);

   auto p1 =
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   auto p3 =
      MakeLobattoFiniteElement(FiniteElementOrders<3, 3>{});
   auto hp_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{p1, p3},
      DGDirectSumNumbering{});

   using HpSpaceType = std::remove_cvref_t<decltype(hp_space)>;
   using LeftFiniteElement =
      typename std::tuple_element_t<
         0,
         typename HpSpaceType::cell_spaces_type>::finite_element_type;
   using RightFiniteElement =
      typename std::tuple_element_t<
         1,
         typename HpSpaceType::cell_spaces_type>::finite_element_type;
   static_assert(HpSpaceType::num_cell_spaces == 2);
   static_assert(!std::is_same_v<LeftFiniteElement, RightFiniteElement>);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;

   auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 2>& x) -> Real
      {
         return Real{1.0} + Real{0.1} * x[0];
      });
   auto tau = MakeCoefficient<"penalty", InverseFacetSize>(
      [] GENDIL_HOST_DEVICE (const Real& h_inv) -> Real
      {
         return Real{16.0} * h_inv;
      });
   constexpr Real sigma = -1.0;

   auto sipdg_form =
      integrate(cells, mu * dot(grad(u), grad(v)))
      + integrate(
           interior_facets,
           -average(mu * dot(grad(u), Normal{})) * jump(v)
           + sigma * jump(u) * average(mu * dot(grad(v), Normal{}))
           + tau * mu * jump(u) * jump(v));

   auto context = MakeWeakFormContext(
      MakeTrialField<"u">(hp_space),
      MakeIntegrationDomain<"mesh">(partition));

   constexpr Integer num_quad_1d = 6;
   auto integration_rule = MakeIntegrationRule(
      IntegrationRuleNumPoints<num_quad_1d, num_quad_1d>{});
   auto op = MakeGenericOperator<GlobalFaceKernelPolicy<num_quad_1d>>(
      sipdg_form,
      context,
      integration_rule);

   PrintType<PartitionType>("partition type");
   PrintType<decltype(sipdg_form)>("SIPDG weak-form type");
   PrintType<decltype(op)>("hp-adaptive SIPDG operator type");

   const std::string_view operator_signature = TypeSignature<decltype(op)>();
   bool success = true;
   success =
      SignatureContains(operator_signature, "PullbackGenericOperator") &&
      success;
   success =
      SignatureContains(operator_signature, "MixedFiniteElementSpace") &&
      success;
   success =
      SignatureContains(
         operator_signature,
         "NonconformingCartesianIntermeshFaceConnectivity") &&
      success;

   // A globally constant state has zero gradient and zero jump, so the
   // interior-only SIPDG action should vanish on this nonconforming partition.
   Vector x(GetAlgebraicDofExtent(hp_space));
   Vector y(GetAlgebraicDofExtent(hp_space));
   x = 1.0;
   y = 0.0;
   op(x, y);

   const Real* y_data = y.ReadHostData();
   Real max_abs_value = 0.0;
   for (Integer i = 0; i < y.Size(); ++i)
   {
      max_abs_value = std::max(max_abs_value, std::abs(y_data[i]));
   }
   if (max_abs_value > 1.0e-11)
   {
      std::cerr << "Constant-state SIPDG residual is nonzero: "
                << max_abs_value << "\n";
      success = false;
   }

   return success ? 0 : 1;
}
