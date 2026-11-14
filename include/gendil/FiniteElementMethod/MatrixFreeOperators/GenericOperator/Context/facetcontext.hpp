// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/coefficientjacobian.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetnormal.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetsides.hpp"
#include "gendil/Meshes/Connectivities/Orientations/orientations.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"

namespace gendil {

// Geometry for the second stored endpoint: neighbor locally, plus globally.
template<typename FaceInfo, typename SecondCellView>
struct FacetContext : FaceInfo
{
   SecondCellView second_cell;

   GENDIL_HOST_DEVICE
   FacetContext(const FaceInfo& face_info, const SecondCellView& second_cell_)
      : FaceInfo(face_info), second_cell(second_cell_)
   {}
};

namespace local_facet_detail
{

// GenDiL stores inverse Jacobians with physical rows and reference columns.
// Re-express only callback inputs in the canonical-minus reference frame;
// physical field gradients keep the raw matrix matching their interpolant.
template<class Jacobian, class Orientation>
GENDIL_HOST_DEVICE
constexpr decltype(auto) CanonicalInverseJacobian(
   const Jacobian& raw, const Orientation& orientation)
{
   if constexpr (is_static_identity_orientation_v<Orientation>)
   {
      return (raw);
   }
   else if constexpr (is_tensor_product_orientation_v<Orientation>)
   {
      return [&]<size_t... I>(std::index_sequence<I...>) {
         return std::make_tuple(CanonicalInverseJacobian(
            std::get<I>(raw), orientation.template Get<I>())...);
      }(std::make_index_sequence<Orientation::num_components>{});
   }
   else if constexpr (std::is_arithmetic_v<Jacobian>)
   {
      static_assert(orientation_dimension_v<Orientation> == 1,
         "A scalar inverse Jacobian represents a one-dimensional cell.");
      const auto axis = detail::GetStructuredOrientationAxis<0>(orientation);
      GENDIL_ASSERT(axis == 1 || axis == -1, "Invalid one-dimensional orientation.");
      return axis > 0 ? raw : -raw;
   }
   else if constexpr (StaticVector<Jacobian>)
   {
      // Compact diagonal storage can represent axis reversals, but not an
      // axis permutation. That restriction also applies to raw cell geometry.
      static_assert(static_extent_v<Jacobian> == orientation_dimension_v<Orientation>);
      Jacobian result{};
      ConstexprLoop<static_extent_v<Jacobian>>([&](auto column) {
         const auto axis = detail::GetStructuredOrientationAxis<column.value>(orientation);
         GENDIL_ASSERT(axis == int(column.value) + 1 || axis == -int(column.value) - 1,
            "Diagonal inverse Jacobians require axis-preserving orientations.");
         const auto value = GetValue(raw, column);
         GetValue(result, column) = axis > 0 ? value : -value;
      });
      return result;
   }
   else
   {
      static_assert(StaticMatrix<Jacobian>,
         "A nonidentity cell orientation requires a dense inverse Jacobian.");
      Jacobian result{};
      ConstexprLoop<static_num_cols_v<Jacobian>>([&](auto column) {
         const auto axis = detail::GetStructuredOrientationAxis<column.value>(orientation);
         // Constant source indices avoid spilling register matrices to local
         // memory merely to support a runtime column subscript on a GPU.
         ConstexprLoop<static_num_cols_v<Jacobian>>([&](auto source) {
            constexpr int positive = int(source.value) + 1;
            if (axis == positive || axis == -positive)
            {
               ConstexprLoop<static_num_rows_v<Jacobian>>([&](auto row) {
                  const auto value = GetValue(raw, row, source);
                  GetValue(result, row, column) = axis > 0 ? value : -value;
               });
            }
         });
      });
      return result;
   }
}

template<class FaceInfo, class NeighborCell>
using LocalFacetStorage = std::conditional_t<std::is_same_v<NeighborCell, Empty>,
   FaceInfo, FacetContext<FaceInfo, NeighborCell>>;

// Selecting a trace consumes a local pair, never an already selected side.
template<class PointContext>
concept LocalFacetPointData = requires(const PointContext& point)
{
   point.inv_J_mesh_current;
   point.CurrentSide();
   point.NeighborSide();
};

} // namespace local_facet_detail

// Local execution owns the current test row. This context presents canonical
// minus/plus traces while keeping geometry and gradient mappings in the stored
// current/neighbor order. Selected points own their values; callback Jacobians
// obtain their reference-coordinate orientation only when requested.
template<class FaceInfo, class NeighborCell = Empty>
struct LocalFacetContext
   : local_facet_detail::LocalFacetStorage<FaceInfo, NeighborCell>
{
private:
   using Storage = local_facet_detail::LocalFacetStorage<FaceInfo, NeighborCell>;
   static constexpr int Sign = static_face_normal_orientation_v<FaceInfo>;

public:
   GENDIL_HOST_DEVICE
   constexpr explicit LocalFacetContext(const FaceInfo& face)
      requires std::is_same_v<NeighborCell, Empty>
      : Storage(face) {}

   GENDIL_HOST_DEVICE
   constexpr LocalFacetContext(const FaceInfo& face, const NeighborCell& neighbor_cell)
      requires (!std::is_same_v<NeighborCell, Empty>)
      : Storage(face, neighbor_cell) {}

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) MinusSide() const
   {
      if constexpr (Sign == 1) return this->CurrentSide();
      else return this->NeighborSide();
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) PlusSide() const
   {
      if constexpr (Sign == 1) return this->NeighborSide();
      else return this->CurrentSide();
   }

   GENDIL_HOST_DEVICE
   GlobalIndex ElementIndex() const
   {
      return this->CurrentSide().GetCellIndex();
   }

};

// Callback Jacobians use the canonical-minus reference frame on both traces;
// physical gradients continue to use the raw point-data matrices. Named points
// retain const-reference identity results; transformed results own their values.
template<class FaceInfo, class NeighborCell, class PointContext>
   requires (!is_physical_boundary_v<PointContext> &&
             requires(const PointContext& point) { point.inv_J_mesh; })
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetCoefficientJacobianInverse(
   const LocalFacetContext<FaceInfo, NeighborCell>& facet, PointContext& point)
{
   const auto& raw = point.inv_J_mesh;
   if constexpr (static_face_normal_orientation_v<FaceInfo> == 1) return (raw);
   else
   {
      // Keep custom by-value endpoints and orientations alive until the
      // transformation finishes. Its result never borrows the orientation.
      const auto& neighbor = facet.NeighborSide();
      const auto& orientation = neighbor.GetOrientation();
      return local_facet_detail::CanonicalInverseJacobian(raw, orientation);
   }
}

// Local point data stays in current/neighbor order. Select an owning canonical
// trace using the facet's static orientation, leaving raw geometry unchanged.
template<class FaceInfo, class NeighborCell,
         local_facet_detail::LocalFacetPointData PointContext>
   requires (!is_physical_boundary_v<PointContext>)
GENDIL_HOST_DEVICE
constexpr auto GetMinusSide(
   const LocalFacetContext<FaceInfo, NeighborCell>&,
   const PointContext& point)
{
   if constexpr (static_face_normal_orientation_v<FaceInfo> == 1)
      return point.CurrentSide();
   else
      return point.NeighborSide();
}

template<class FaceInfo, class NeighborCell,
         local_facet_detail::LocalFacetPointData PointContext>
   requires (!is_physical_boundary_v<PointContext>)
GENDIL_HOST_DEVICE
constexpr auto GetPlusSide(
   const LocalFacetContext<FaceInfo, NeighborCell>&,
   const PointContext& point)
{
   if constexpr (static_face_normal_orientation_v<FaceInfo> == 1)
      return point.NeighborSide();
   else
      return point.CurrentSide();
}

template<class FaceInfo>
LocalFacetContext(const FaceInfo&) -> LocalFacetContext<FaceInfo>;

template<class FaceInfo, class NeighborCell>
LocalFacetContext(const FaceInfo&, const NeighborCell&) -> LocalFacetContext<FaceInfo, NeighborCell>;

template<class Channels>
inline constexpr bool global_interior_channels_require_plus_side_jacobian_v =
   std::remove_cvref_t<Channels>::template Contains<GradientPlusChannel>();

// Local geometry requirements refer to the neighbor storage slot.
// The normal orientation determines which weak-form trace reads that slot.
template<class Integrand, int NormalOrientation = 1>
inline constexpr bool local_interior_context_requires_neighbor_jacobian_v =
   NormalOrientation == 1 ? requires_plus_side_jacobian_v<Integrand>
                          : requires_minus_side_jacobian_v<Integrand>;

template<class Integrand, class Channels>
inline constexpr bool global_interior_context_requires_plus_side_jacobian_v =
   requires_plus_side_jacobian_v<Integrand> ||
   global_interior_channels_require_plus_side_jacobian_v<Channels>;

template<typename WeakFormContext, typename Integrand, typename FaceInfo>
GENDIL_HOST_DEVICE
auto MakeLocalInteriorFacetContext(
   const WeakFormContext& wf_ctx,
   const Integrand& integrand,
   const FaceInfo& face_info)
{
   constexpr int Sign = static_face_normal_orientation_v<FaceInfo>;
   if constexpr (local_interior_context_requires_neighbor_jacobian_v<Integrand, Sign>)
   {
      const auto& mesh =
         GetCellIntegrationDomainMesh(integrand, wf_ctx);

      const auto& neighbor_face = face_info.NeighborSide();
      auto neighbor_cell = mesh.GetCell(neighbor_face.GetCellIndex());

      ApplyOrientationToCell(
         neighbor_face.GetOrientation(),
         neighbor_cell);

      return LocalFacetContext{ face_info, neighbor_cell };
   }
   else
   {
      return LocalFacetContext{ face_info };
   }
}

template<
   typename FaceDomain,
   typename Integrand,
   typename Channels,
   typename FaceInfo>
GENDIL_HOST_DEVICE
auto MakeGlobalInteriorFacetContext(
   const FaceDomain& face_domain,
   const Integrand& /*integrand*/,
   const Channels& /*channels*/,
   const FaceInfo& face_info)
{
   if constexpr (
      global_interior_context_requires_plus_side_jacobian_v<
         Integrand,
         Channels>)
   {
      static_assert(
         requires (const FaceDomain& domain)
         {
            domain.GetPlusCellMesh();
         },
         "Global interior facet context requires a face execution space "
         "that exposes GetPlusCellMesh() when plus-side geometry "
         "is needed.");

      const auto& plus_mesh = face_domain.GetPlusCellMesh();

      auto plus_cell =
         plus_mesh.GetCell(face_info.PlusSide().GetCellIndex());

      ApplyOrientationToCell(
         face_info.PlusSide().GetOrientation(),
         plus_cell);

      return FacetContext{ face_info, plus_cell };
   }
   else
   {
      return face_info;
   }
}

} // namespace gendil
