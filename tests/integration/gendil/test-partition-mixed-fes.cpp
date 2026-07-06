// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace {

bool TestOneCellPartPartition()
{
   constexpr GlobalIndex num_cells = 3;
   Cartesian1DMesh mesh(1.0 / num_cells, num_cells);

   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   auto boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});

   auto partition =
      MakePartition(
         MakeCellPart(mesh),
         MakeInteriorFacePart<0, 0>(interior_faces),
         MakeBoundaryFacePart<0>(boundary_faces));

   using PartitionType = decltype(partition);
   static_assert(PartitionType::num_cell_parts == 1);
   static_assert(PartitionType::num_interior_face_parts == 1);
   static_assert(PartitionType::num_boundary_face_parts == 2);
   static_assert(
      !is_tuple_v<
         typename std::tuple_element_t<
            0,
            typename PartitionType::interior_face_parts_type>::
               face_mesh_type>);
   static_assert(
      !is_tuple_v<
         typename std::tuple_element_t<
            0,
            typename PartitionType::boundary_face_parts_type>::
               face_mesh_type>);

   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe},
         std::tuple{L2Restriction{0}});

   using MixedType = decltype(mixed);
   static_assert(MixedType::num_cell_spaces == 1);
   static_assert(MixedType::num_interior_face_parts == 1);
   static_assert(MixedType::num_boundary_face_parts == 2);
   static_assert(
      std::is_same_v<
         typename MixedType::partition_type,
         PartitionType>);

   return
      mixed.GetCellFiniteElementSpace<0>().restriction.shift == 0 &&
      mixed.GetNumberOfCellFiniteElementSpaces() == 1 &&
      mixed.GetPartition().GetNumberOfInteriorFaceParts() == 1 &&
      mixed.GetPartition().GetNumberOfBoundaryFaceParts() == 2;
}

bool TestTwoCellPartPartition()
{
   constexpr GlobalIndex num_cells = 2;
   Cartesian1DMesh mesh_left(1.0 / num_cells, num_cells);
   Cartesian1DMesh mesh_right(1.0 / num_cells, num_cells);

   auto left_interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   auto right_interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   CartesianIntermeshFaceConnectivity<1, 1>
      interface_faces({num_cells}, {num_cells});
   auto left_boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});
   auto right_boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});

   auto partition =
      MakePartition(
         MakeCellPart(mesh_left),
         MakeCellPart(mesh_right),
         MakeBoundaryFacePart<1>(right_boundary_faces),
         MakeInteriorFacePart<0, 0>(left_interior_faces),
         MakeInteriorFacePart<1, 1>(right_interior_faces),
         MakeInteriorFacePart<0, 1>(interface_faces),
         MakeBoundaryFacePart<0>(left_boundary_faces));

   using PartitionType = decltype(partition);
   static_assert(PartitionType::num_cell_parts == 2);
   static_assert(PartitionType::num_interior_face_parts == 3);
   static_assert(PartitionType::num_boundary_face_parts == 4);
   static_assert(
      std::tuple_element_t<
         2,
         typename PartitionType::interior_face_parts_type>::
            minus_cell_index == 0);
   static_assert(
      std::tuple_element_t<
         2,
         typename PartitionType::interior_face_parts_type>::
            plus_cell_index == 1);
   static_assert(
      std::tuple_element_t<
         0,
         typename PartitionType::boundary_face_parts_type>::
            cell_index == 1);
   static_assert(
      std::tuple_element_t<
         2,
         typename PartitionType::boundary_face_parts_type>::
            cell_index == 0);

   auto fe_p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe_p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});

   auto u_cell0_unshifted =
      MakeFiniteElementSpace(mesh_left, fe_p1, L2Restriction{0});
   const GlobalIndex u_cell1_shift =
      u_cell0_unshifted.GetNumberOfFiniteElementDofs();

   auto u_mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe_p1, fe_p2},
         std::tuple{L2Restriction{0}, L2Restriction{u_cell1_shift}});
   auto u_dg_mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe_p1, fe_p2},
         DGDirectSumNumbering{});
   auto mu_mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{fe_p1, fe_p1},
         DGDirectSumNumbering{});

   using UMixed = decltype(u_mixed);
   using MuMixed = decltype(mu_mixed);
   static_assert(UMixed::num_cell_spaces == 2);
   static_assert(UMixed::num_interior_face_parts == 3);
   static_assert(UMixed::num_boundary_face_parts == 4);
   static_assert(
      decltype(u_dg_mixed)::num_cell_spaces == UMixed::num_cell_spaces);
   static_assert(
      decltype(u_dg_mixed)::num_interior_face_parts ==
      UMixed::num_interior_face_parts);
   static_assert(
      decltype(u_dg_mixed)::num_boundary_face_parts ==
      UMixed::num_boundary_face_parts);
   static_assert(MuMixed::num_cell_spaces == 2);
   static_assert(MuMixed::num_interior_face_parts == 3);
   static_assert(MuMixed::num_boundary_face_parts == 4);
   static_assert(
      std::is_same_v<
         typename std::tuple_element_t<
            2,
            typename UMixed::interior_face_parts_type>::face_mesh_type,
         typename std::tuple_element_t<
            2,
            typename MuMixed::interior_face_parts_type>::face_mesh_type>);
   static_assert(
      !std::is_same_v<
         typename std::tuple_element_t<
            1,
            typename UMixed::cell_spaces_type>::finite_element_type,
         typename std::tuple_element_t<
            1,
            typename MuMixed::cell_spaces_type>::finite_element_type>);

   const GlobalIndex u_cell0_dofs =
      u_mixed.GetCellFiniteElementSpace<0>().
         GetNumberOfFiniteElementDofs();
   const GlobalIndex mu_cell0_dofs =
      mu_mixed.GetCellFiniteElementSpace<0>().
         GetNumberOfFiniteElementDofs();

   bool success = true;
   success =
      (u_mixed.GetCellFiniteElementSpace<0>().restriction.shift == 0) &&
      success;
   success =
      (u_mixed.GetCellFiniteElementSpace<1>().restriction.shift ==
       u_cell0_dofs) &&
      success;
   success =
      (u_dg_mixed.GetCellFiniteElementSpace<0>().restriction.shift ==
       u_mixed.GetCellFiniteElementSpace<0>().restriction.shift) &&
      success;
   success =
      (u_dg_mixed.GetCellFiniteElementSpace<1>().restriction.shift ==
       u_mixed.GetCellFiniteElementSpace<1>().restriction.shift) &&
      success;
   success =
      (u_dg_mixed.GetNumberOfInteriorFaces() ==
       u_mixed.GetNumberOfInteriorFaces()) &&
      success;
   success =
      (u_dg_mixed.GetNumberOfBoundaryFaces() ==
       u_mixed.GetNumberOfBoundaryFaces()) &&
      success;
   success =
      (mu_mixed.GetCellFiniteElementSpace<1>().restriction.shift ==
       mu_cell0_dofs) &&
      success;
   using BoundaryPart0 =
      std::tuple_element_t<0, typename UMixed::boundary_face_parts_type>;
   using BoundaryPart2 =
      std::tuple_element_t<2, typename UMixed::boundary_face_parts_type>;
   static_assert(BoundaryPart0::cell_index == 1);
   static_assert(BoundaryPart2::cell_index == 0);
   success =
      (u_mixed.GetNumberOfFiniteElementDofs() ==
       u_mixed.GetCellFiniteElementSpace<0>().GetNumberOfFiniteElementDofs() +
          u_mixed.GetCellFiniteElementSpace<1>().
             GetNumberOfFiniteElementDofs()) &&
      success;
   success =
      (mu_mixed.GetNumberOfInteriorFaces() ==
       u_mixed.GetNumberOfInteriorFaces()) &&
      success;
   success =
      (mu_mixed.GetNumberOfBoundaryFaces() ==
       u_mixed.GetNumberOfBoundaryFaces()) &&
      success;

   return success;
}

bool TestCellOnlyMixedSpaceOwnsCellOnlyPartition()
{
   constexpr GlobalIndex num_cells = 2;
   Cartesian1DMesh mesh0(0.5, num_cells);
   Cartesian1DMesh mesh1(0.5, num_cells);

   auto fe0 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe1 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   auto fes0 = MakeFiniteElementSpace(mesh0, fe0);
   auto fes1 = MakeFiniteElementSpace(mesh1, fe1);

   auto mixed = MakeMixedFiniteElementSpace(fes0, fes1);
   using Mixed = std::remove_cvref_t<decltype(mixed)>;
   static_assert(Mixed::num_cell_spaces == 2);
   static_assert(Mixed::num_interior_face_parts == 0);
   static_assert(Mixed::num_boundary_face_parts == 0);
   static_assert(
      std::tuple_size_v<typename Mixed::cell_parts_type> == 2);
   static_assert(
      std::tuple_size_v<typename Mixed::interior_face_parts_type> == 0);
   static_assert(
      std::tuple_size_v<typename Mixed::boundary_face_parts_type> == 0);

   return
      mixed.GetPartition().GetNumberOfCellParts() == 2 &&
      mixed.GetNumberOfCellFiniteElementSpaces() == 2 &&
      mixed.GetNumberOfInteriorFaces() == 0 &&
      mixed.GetNumberOfBoundaryFaces() == 0;
}

template<bool P2Minus>
bool TestPartitionBuiltGenericOperator()
{
   constexpr GlobalIndex num_cells = 1;
   Cartesian1DMesh mesh_left(1.0, num_cells);
   Cartesian1DMesh mesh_right(1.0, num_cells);
   CartesianIntermeshFaceConnectivity<1, 1>
      interface_faces({num_cells}, {num_cells});

   auto partition =
      MakePartition(
         MakeCellPart(mesh_left),
         MakeCellPart(mesh_right),
         MakeInteriorFacePart<0, 1>(interface_faces));

   auto fe_p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe_p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});

   auto mixed =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeMixedFiniteElementSpace(
               partition,
               std::tuple{fe_p2, fe_p1},
               DGDirectSumNumbering{});
         }
         else
         {
            return MakeMixedFiniteElementSpace(
               partition,
               std::tuple{fe_p1, fe_p2},
               DGDirectSumNumbering{});
         }
      }();

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"solid"> interior_facets;
   auto form = integrate(interior_facets, jump(u) * jump(v));
   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

   auto op =
      MakeGenericOperator<SerialKernelConfiguration>(
         form,
         ctx,
         integration_rule);

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
   success = TestOneCellPartPartition() && success;
   success = TestTwoCellPartPartition() && success;
   success = TestCellOnlyMixedSpaceOwnsCellOnlyPartition() && success;
   success = TestPartitionBuiltGenericOperator<true>() && success;
   success = TestPartitionBuiltGenericOperator<false>() && success;

   if (!success)
   {
      std::cerr << "Partition mixed finite element space test failed.\n";
      return 1;
   }

   return 0;
}
