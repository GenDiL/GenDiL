// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>
#include <type_traits>

using namespace gendil;

namespace {

#if defined(GENDIL_USE_DEVICE)
using GlobalFaceKernelPolicy =
   DeviceKernelConfiguration<ThreadBlockLayout<4>, 1, 2>;
#else
using GlobalFaceKernelPolicy = SerialKernelConfiguration;
#endif

template<class T, class = void>
struct has_interior_face_spaces : std::false_type {};

template<class T>
struct has_interior_face_spaces<
   T,
   std::void_t<decltype(std::declval<const T&>().InteriorFaceSpaces())>>
   : std::true_type {};

template<class T, class = void>
struct has_boundary_face_spaces : std::false_type {};

template<class T>
struct has_boundary_face_spaces<
   T,
   std::void_t<decltype(std::declval<const T&>().BoundaryFaceSpaces())>>
   : std::true_type {};

template<class T, class = void>
struct has_get_interior_face_fes : std::false_type {};

template<class T>
struct has_get_interior_face_fes<
   T,
   std::void_t<
      decltype(std::declval<const T&>().
         template GetInteriorFaceFiniteElementSpace<0>())>>
   : std::true_type {};

template<class T, class = void>
struct has_get_boundary_face_fes : std::false_type {};

template<class T>
struct has_get_boundary_face_fes<
   T,
   std::void_t<
      decltype(std::declval<const T&>().
         template GetBoundaryFaceFiniteElementSpace<0>())>>
   : std::true_type {};

template<class T>
inline constexpr bool has_interior_face_spaces_v =
   has_interior_face_spaces<T>::value;

template<class T>
inline constexpr bool has_boundary_face_spaces_v =
   has_boundary_face_spaces<T>::value;

template<class T>
inline constexpr bool has_get_interior_face_fes_v =
   has_get_interior_face_fes<T>::value;

template<class T>
inline constexpr bool has_get_boundary_face_fes_v =
   has_get_boundary_face_fes<T>::value;

auto MakeFixture()
{
   constexpr GlobalIndex num_cells = 3;
   const Real h = 1.0 / num_cells;

   Cartesian1DMesh mesh0(h, num_cells);
   Cartesian1DMesh mesh1(h, num_cells);
   auto fe1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   CartesianIntermeshFaceConnectivity<1, 1>
      interface_faces({num_cells}, {num_cells});
   auto boundary_faces0 =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});
   auto boundary_faces1 =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});
   auto partition =
      MakePartition(
         MakeCellPart(mesh0),
         MakeCellPart(mesh1),
         MakeInteriorFacePart<0, 1>(interface_faces),
         MakeBoundaryFacePart<0>(boundary_faces0),
         MakeBoundaryFacePart<1>(boundary_faces1));

   return std::tuple{mesh0, mesh1, fe1, fe2, partition};
}

bool TestMixedSpaceOwnsOnlyCellsAndPartition()
{
   auto fixture = MakeFixture();
   const auto& mesh0 = std::get<0>(fixture);
   const auto& fe1 = std::get<2>(fixture);
   const auto& fe2 = std::get<3>(fixture);
   const auto& partition = std::get<4>(fixture);

   auto space0 = MakeFiniteElementSpace(mesh0, fe1, L2Restriction{0});
   const GlobalIndex shift = space0.GetNumberOfFiniteElementDofs();
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe1, fe2},
         std::tuple{L2Restriction{0}, L2Restriction{shift}});

   using Mixed = std::remove_cvref_t<decltype(mixed)>;
   using Partition = std::remove_cvref_t<decltype(partition)>;
   static_assert(is_mixed_finite_element_space_v<Mixed>);
   static_assert(Mixed::num_cell_spaces == 2);
   static_assert(Mixed::num_interior_face_parts == Partition::num_interior_face_parts);
   static_assert(Mixed::num_boundary_face_parts == Partition::num_boundary_face_parts);
   static_assert(std::is_same_v<typename Mixed::partition_type, Partition>);
   static_assert(!has_interior_face_spaces_v<Mixed>);
   static_assert(!has_boundary_face_spaces_v<Mixed>);
   static_assert(!has_get_interior_face_fes_v<Mixed>);
   static_assert(!has_get_boundary_face_fes_v<Mixed>);

   return
      mixed.GetNumberOfCellFiniteElementSpaces() == 2 &&
      mixed.GetNumberOfInteriorFaces() ==
         std::get<0>(partition.InteriorFaceParts()).face_mesh.GetNumberOfFaces() &&
      mixed.GetNumberOfFiniteElementDofs() ==
         mixed.GetCellFiniteElementSpace<0>().GetNumberOfFiniteElementDofs() +
            mixed.GetCellFiniteElementSpace<1>().GetNumberOfFiniteElementDofs();
}

bool TestInteriorExecutionBatchAndFieldBindings()
{
   auto fixture = MakeFixture();
   const auto& fe1 = std::get<2>(fixture);
   const auto& fe2 = std::get<3>(fixture);
   const auto& partition = std::get<4>(fixture);

   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe1, fe2},
         DGDirectSumNumbering{});
   auto coeff_mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe2, fe1},
         DGDirectSumNumbering{});

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeFiniteElementField<"mu">(coeff_mixed, Empty{}),
      MakeIntegrationDomain<"mesh">(mixed));

   Integer num_batches = 0;
   bool success = true;

   ForEachInteriorFaceFiniteElementSpace(
      ctx,
      InteriorFacets<"mesh">{},
      [&] (const auto& batch)
      {
         ++num_batches;
         using Batch = std::remove_cvref_t<decltype(batch)>;
         static_assert(is_interior_face_execution_batch_v<Batch>);
         static_assert(Batch::minus_cell_part_index == 0);
         static_assert(Batch::plus_cell_part_index == 1);

         auto restricted =
            MakeRestrictedWeakFormContext<"u", "u">(
               ctx,
               InteriorFacets<"mesh">{},
               batch);

         using UBinding =
            std::remove_cvref_t<
               decltype(restricted.template fe_field<"u">().space)>;
         using MuBinding =
            std::remove_cvref_t<
               decltype(restricted.template fe_field<"mu">().space)>;
         static_assert(is_interior_face_field_binding_v<UBinding>);
         static_assert(is_interior_face_field_binding_v<MuBinding>);
         static_assert(is_two_space_interior_face_field_binding_v<UBinding>);
         static_assert(is_two_space_interior_face_field_binding_v<MuBinding>);
         static_assert(
            std::is_same_v<
               typename UBinding::face_part_type,
               typename Batch::face_part_type>);

         auto ir = MakeIntegrationRule(IntegrationRuleNumPoints<4>{});
         auto op_ctx = MakeFacetOperatorContext(restricted, ir, batch);
         (void)op_ctx;
      });

   success = success && num_batches == mixed.GetPartition().GetNumberOfInteriorFaceParts();
   return success;
}

bool TestBoundaryExecutionBatchAndFieldBindings()
{
   auto fixture = MakeFixture();
   const auto& fe1 = std::get<2>(fixture);
   const auto& fe2 = std::get<3>(fixture);
   const auto& partition = std::get<4>(fixture);

   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe1, fe2},
         DGDirectSumNumbering{});

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"mesh">(mixed));

   Integer num_batches = 0;
   bool saw_cell_part_0 = false;
   bool saw_cell_part_1 = false;

   ForEachBoundaryFaceFiniteElementSpace(
      ctx,
      BoundaryFacets<"mesh">{},
      [&] (const auto& batch)
      {
         ++num_batches;
         using Batch = std::remove_cvref_t<decltype(batch)>;
         static_assert(is_boundary_face_execution_batch_v<Batch>);
         if constexpr (Batch::cell_part_index == 0)
         {
            saw_cell_part_0 = true;
         }
         if constexpr (Batch::cell_part_index == 1)
         {
            saw_cell_part_1 = true;
         }

         auto restricted =
            MakeRestrictedWeakFormContext<"u", "u">(
               ctx,
               BoundaryFacets<"mesh">{},
               batch);
         using UBinding =
            std::remove_cvref_t<
               decltype(restricted.template fe_field<"u">().space)>;
         static_assert(is_boundary_face_field_binding_v<UBinding>);
         static_assert(
            std::is_same_v<
               typename UBinding::face_part_type,
               typename Batch::face_part_type>);

         auto ir = MakeIntegrationRule(IntegrationRuleNumPoints<4>{});
         auto op_ctx = MakeFacetOperatorContext(restricted, ir, batch);
         (void)op_ctx;
      });

   return
      num_batches == mixed.GetPartition().GetNumberOfBoundaryFaceParts() &&
      saw_cell_part_0 &&
      saw_cell_part_1;
}

bool TestPartitionMixedGenericOperatorSmoke()
{
   auto fixture = MakeFixture();
   const auto& fe1 = std::get<2>(fixture);
   const auto& fe2 = std::get<3>(fixture);
   const auto& partition = std::get<4>(fixture);

   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe1, fe2},
         DGDirectSumNumbering{});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto form =
      integrate(InteriorFacets<"mesh">{}, jump(u) * jump(v))
    + integrate(BoundaryFacets<"mesh">{}, u * v);
   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"mesh">(mixed));
   auto ir = MakeIntegrationRule(IntegrationRuleNumPoints<4>{});
   auto op =
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         form,
         ctx,
         ir);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   Vector y(mixed.GetNumberOfFiniteElementDofs());
   x = 1.0;
   y = 0.0;
   op(x, y);

   return true;
}

} // namespace

int main()
{
   bool success = true;
   success = TestMixedSpaceOwnsOnlyCellsAndPartition() && success;
   success = TestInteriorExecutionBatchAndFieldBindings() && success;
   success = TestBoundaryExecutionBatchAndFieldBindings() && success;
   success = TestPartitionMixedGenericOperatorSmoke() && success;

   if (!success)
   {
      std::cerr << "Mixed integration-domain ownership test failed.\n";
      return 1;
   }
   return 0;
}
