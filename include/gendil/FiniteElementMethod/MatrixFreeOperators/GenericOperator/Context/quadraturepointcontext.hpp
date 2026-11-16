// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/negation.hpp"
#include "gendil/Meshes/cell.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/computefacetgeometry.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/facetquaddata.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"

namespace gendil
{

template<class QuadIndex, class PhysicalCoordinates, class Jacobian>
struct QuadraturePointContext
{
   QuadIndex quad_index;
   PhysicalCoordinates X;
   Jacobian J_mesh;
   Jacobian inv_J_mesh;
   Real det_J;
   Real weight;
};

// =============================================================================
// FacetQuadraturePointContext
// =============================================================================
template<class QuadIndex, class PhysicalCoordinates, class Jacobian, class PhysicalNormal>
struct OneSidedFacetQuadraturePointContext
{
   QuadIndex quad_index;
   PhysicalCoordinates X;
   // Jacobian J_mesh;
   Jacobian inv_J_mesh;
   // Real det_J;
   Real weight;
   Real det_J_facet;          // Surface Jacobian determinant
   Real inverse_normal_cell_size; // Selected element: ||J^{-T} n_ref||
   // Shared canonical unit normal on interiors, outward unit normal on boundaries.
   PhysicalNormal physical_normal;
};

namespace quadrature_context_detail
{
template<class Shape, class Points, class Face, Integer Axis>
GENDIL_HOST_DEVICE Real BoundaryReferenceCoordinate(
   const NonconformingDofToQuad<Shape, Points, Face, Axis>& data, Integer q)
{
   const auto point = data.face.template MapReferenceToFaceCoordinates1d<Axis>(
      Point<1>{Points::GetCoord(q)});
   return point[0];
}

template<class Data>
GENDIL_HOST_DEVICE Real BoundaryReferenceCoordinate(const Data& data, Integer q)
{
   if constexpr (requires { data.coord(q); }) return data.coord(q);
   else if constexpr (requires { typename Data::points; }) return Data::points::GetCoord(q);
   else return Data::GetCoord(q);
}

template<class QData, class Index, class Coordinates>
GENDIL_HOST_DEVICE void BoundaryReferenceCoordinates(const QData& qdata,
   const Index& index, Coordinates& coordinates, size_t& dimension)
{
   if constexpr (is_tensor_product_data_v<QData>)
   {
      [&]<size_t... I>(std::index_sequence<I...>)
      {
         (BoundaryReferenceCoordinates(GetTensorProductEntry<I>(qdata), index,
                                       coordinates, dimension), ...);
      }(std::make_index_sequence<tensor_product_entry_count_v<QData>>{});
   }
   else
   {
      const auto q = index[dimension];
      coordinates[dimension] = BoundaryReferenceCoordinate(qdata, q);
      ++dimension;
   }
}
} // namespace quadrature_context_detail

// Physical boundaries are distinct from one selected side of an interior face.
// Geometry callback descriptors refer to the adjacent parent cell; the surface
// determinant remains separate and supplies the integration measure.
template<class QuadIndex, class PhysicalCoordinates, class Jacobian, class PhysicalNormal>
struct BoundaryFacetQuadraturePointContext
   : OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian, PhysicalNormal>
{
   static constexpr bool is_physical_boundary = true;
   Jacobian J_mesh;
   Real det_J;
   PhysicalCoordinates reference_coordinates;
};

// Fixed minus/plus geometry for global facet expression evaluation.
template<
   typename QuadIndex,
   typename PhysicalCoordinates,
   typename MinusJacobian,
   typename PlusJacobian,
   typename PhysicalNormal >
struct TwoSidedFacetQuadraturePointContext
{
   QuadIndex quad_index;
   PhysicalCoordinates X;
   MinusJacobian inv_J_mesh_minus;
   PlusJacobian inv_J_mesh_plus;
   Real weight;
   Real det_J_facet;          // Minus-side surface Jacobian determinant
   Real inverse_normal_cell_size_minus;
   Real inverse_normal_cell_size_plus;
   PhysicalNormal physical_normal; // Canonical-minus outward unit normal

   GENDIL_HOST_DEVICE
   auto MinusSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, MinusJacobian, PhysicalNormal>
         qc_minus{
            quad_index, X, inv_J_mesh_minus, weight, det_J_facet,
            inverse_normal_cell_size_minus, physical_normal };
      return qc_minus;
   }

   GENDIL_HOST_DEVICE
   auto PlusSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, PlusJacobian, PhysicalNormal>
         qc_plus{
            quad_index, X, inv_J_mesh_plus, weight, det_J_facet,
            inverse_normal_cell_size_plus, physical_normal };
      return qc_plus;
   }
};

// Local geometry stays in current/neighbor order. The facet context selects
// canonical traces without changing the storage identities. The point factory
// orients the shared unit normal once; both selected traces retain that normal.
template<
   typename QuadIndex,
   typename PhysicalCoordinates,
   typename CurrentJacobian,
   typename NeighborJacobian,
   typename PhysicalNormal >
struct LocalFacetQuadraturePointData
{
   QuadIndex quad_index;
   PhysicalCoordinates X;
   CurrentJacobian inv_J_mesh_current;
   NeighborJacobian inv_J_mesh_neighbor;
   Real weight;
   Real det_J_facet;          // Current-cell surface Jacobian determinant
   Real inverse_normal_cell_size_current;
   Real inverse_normal_cell_size_neighbor;
   PhysicalNormal physical_normal; // Shared canonical unit normal

   GENDIL_HOST_DEVICE
   auto CurrentSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, CurrentJacobian, PhysicalNormal>
         qc_current{
            quad_index, X, inv_J_mesh_current, weight, det_J_facet,
            inverse_normal_cell_size_current, physical_normal };
      return qc_current;
   }

   GENDIL_HOST_DEVICE
   auto NeighborSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, NeighborJacobian, PhysicalNormal>
         qc_neighbor{
            quad_index, X, inv_J_mesh_neighbor, weight, det_J_facet,
            inverse_normal_cell_size_neighbor, physical_normal };
      return qc_neighbor;
   }
};

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class Cell,
   class Integrand,
   class QuadIndex>
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContext(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext<Cell>& ec,
   const Integrand& integrand,
   const QuadIndex& quad_index)
{
   using I = std::remove_cvref_t<Integrand>;

   // Volume domains only for now: Cells<DomainName> must provide `static constexpr auto name`
   constexpr auto DomainName = I::domain_type::name;

   // Get quad data from operator context (already bound to IntegrationRule when op_ctx was built)
   const auto& mesh_quad_data =
      op_ctx.template GetMeshQuadData<DomainName>();

   // Types for X and Jacobian from the domain mesh type
   const auto& mesh =
      GetCellIntegrationDomainMesh(integrand, wf_ctx);
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian            = typename Mesh::cell_type::jacobian;

   QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc{};

   qc.quad_index = quad_index;

   ec.cell.GetValuesAndJacobian(quad_index, mesh_quad_data, qc.X, qc.J_mesh);

   qc.det_J   = ComputeInverseAndDeterminant(qc.J_mesh, qc.inv_J_mesh);
   qc.weight = GetWeight(quad_index, mesh_quad_data);

   return qc;
}

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class ElementContext,
   CellFaceView Face,
   class Integrand,
   class QuadIndex>
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContext(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const Face& face,
   const Integrand& integrand,
   const QuadIndex& quad_index)
{
   using I = std::remove_cvref_t<Integrand>;

   // Volume domains only for now: Cells<DomainName> must provide `static constexpr auto name`
   constexpr auto DomainName = I::domain_type::name;

   // Get quad data from operator context (already bound to IntegrationRule when op_ctx was built)
   const auto& mesh_quad_data =
      op_ctx.template GetMeshFacetQuadData<DomainName>().MinusSide();
   auto&& mesh_facet_quad_data = GetFacetQuadData(mesh_quad_data, face);

   // Types for X and Jacobian from the domain mesh type
   const auto& mesh =
      GetCellIntegrationDomainMesh(integrand, wf_ctx);
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian            = typename Mesh::cell_type::jacobian;

   QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc{};

   qc.quad_index = quad_index;

   // Jacobian J_mesh;
   // element_context.cell.GetValuesAndJacobian(quad_index, mesh_quad_data, qc.X, J_mesh);
   mesh::ComputePhysicalCoordinatesAndJacobian(
      element_context.cell,
      quad_index,
      mesh_facet_quad_data,
      qc.X,
      qc.J_mesh);

   qc.det_J  = ComputeInverseAndDeterminant(qc.J_mesh, qc.inv_J_mesh);
   qc.weight = GetWeight(quad_index, mesh_facet_quad_data);

   return qc;
}

// =============================================================================
// Interior facet quadrature-point context
// =============================================================================
template<
   bool NeedSecondSideInvJ,
   template<class, class, class, class, class> class Context = TwoSidedFacetQuadraturePointContext,
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceContext,
   typename Integrand,
   typename QuadIndex>
GENDIL_HOST_DEVICE
auto MakeInteriorFacetQuadraturePointContextImpl(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const FaceContext& face,
   const Integrand& integrand,
   const QuadIndex& quad_index)
{
   static_assert(
      InteriorFacetIntegrand<Integrand>,
      "MakeInteriorFacetQuadraturePointContext requires an interior facet "
      "integrand.");

   constexpr auto DomainName = Integrand::domain_type::name;

   const auto& mesh_quad_data =
      op_ctx.template GetMeshFacetQuadData<DomainName>();
   const auto& first_side = [&]() -> decltype(auto)
   {
      if constexpr (requires { face.CurrentSide(); }) return face.CurrentSide();
      else return face.MinusSide();
   }();
   auto&& first_quad_data =
      GetFacetQuadData(mesh_quad_data.MinusSide(), first_side);

   const auto& mesh =
      GetCellIntegrationDomainMesh(integrand, wf_ctx);
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian            = typename Mesh::cell_type::jacobian;

   QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> first{};
   first.quad_index = quad_index;
   mesh::ComputePhysicalCoordinatesAndJacobian(
      element_context.cell,
      quad_index,
      first_quad_data,
      first.X,
      first.J_mesh);

   first.det_J  = ComputeInverseAndDeterminant(first.J_mesh, first.inv_J_mesh);
   first.weight = GetWeight(quad_index, first_quad_data);

   auto first_reference_normal = first_side.GetReferenceNormal();
   auto first_geometry = ComputeFacetGeometry(
      first.inv_J_mesh,
      first_reference_normal,
      first.det_J);

   using PhysicalNormal = decltype(first_geometry.normalized_physical_normal);

   if constexpr (NeedSecondSideInvJ)
   {
      const auto& second_side = [&]() -> decltype(auto)
      {
         if constexpr (requires { face.NeighborSide(); }) return face.NeighborSide();
         else return face.PlusSide();
      }();
      QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> second{};
      auto&& second_quad_data =
         GetFacetQuadData(mesh_quad_data.PlusSide(), second_side);
      mesh::ComputePhysicalCoordinatesAndJacobian(
         face.second_cell,
         quad_index,
         second_quad_data,
         second.X,
         second.J_mesh);

      second.det_J  = ComputeInverseAndDeterminant(second.J_mesh, second.inv_J_mesh);
      const auto second_geometry = ComputeFacetGeometry(
         second.inv_J_mesh,
         second_side.GetReferenceNormal(),
         second.det_J);

      return Context<QuadIndex, decltype(first.X), decltype(first.inv_J_mesh),
         decltype(second.inv_J_mesh), PhysicalNormal>{
         std::move(first.quad_index), std::move(first.X),
         std::move(first.inv_J_mesh), std::move(second.inv_J_mesh),
         first.weight, first_side.Measure() * first_geometry.det_J_facet,
         first_geometry.inverse_normal_cell_size,
         second_geometry.inverse_normal_cell_size,
         std::move(first_geometry.normalized_physical_normal)};
   }
   else
   {
      return Context<QuadIndex, decltype(first.X), decltype(first.inv_J_mesh),
         Empty, PhysicalNormal>{
         std::move(first.quad_index), std::move(first.X),
         std::move(first.inv_J_mesh), Empty{},
         first.weight, first_side.Measure() * first_geometry.det_J_facet,
         first_geometry.inverse_normal_cell_size, Real{0},
         std::move(first_geometry.normalized_physical_normal)};
   }
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceContext,
   typename Integrand,
   typename QuadIndex>
GENDIL_HOST_DEVICE
auto MakeLocalInteriorFacetQuadraturePointContext(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const FaceContext& face,
   const Integrand& integrand,
   const QuadIndex& quad_index)
{
   constexpr int Sign = static_face_normal_orientation_v<FaceContext>;
   auto point = MakeInteriorFacetQuadraturePointContextImpl<
      local_interior_context_requires_neighbor_jacobian_v<Integrand, Sign>,
      LocalFacetQuadraturePointData>(
         kernel_context,
         wf_ctx,
         op_ctx,
         element_context,
         face,
         integrand,
         quad_index);
   if constexpr (Sign == -1)
      point.physical_normal = NegateValue(point.physical_normal);
   return point;
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceContext,
   typename Integrand,
   typename Channels,
   typename QuadIndex>
GENDIL_HOST_DEVICE
auto MakeGlobalInteriorFacetQuadraturePointContext(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const FaceContext& face,
   const Integrand& integrand,
   const Channels& /*channels*/,
   const QuadIndex& quad_index)
{
   return MakeInteriorFacetQuadraturePointContextImpl<
      global_interior_context_requires_plus_side_jacobian_v<
         Integrand,
         Channels>>(
            kernel_context,
            wf_ctx,
            op_ctx,
            element_context,
            face,
            integrand,
            quad_index);
}

// =============================================================================
// MakeFacetQuadraturePointContext
// =============================================================================
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceContext,
   typename Integrand,
   typename QuadIndex>
GENDIL_HOST_DEVICE
auto MakeFacetQuadraturePointContext(
   const KernelContext& kernel_context,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const ElementContext& element_context,
   const FaceContext& face,
   const Integrand& integrand,
   const QuadIndex& quad_index)
{
   if constexpr (BoundaryFacetIntegrand<Integrand>)
   {
      const auto& side = [&]() -> const auto&
      {
         if constexpr (requires { face.CurrentSide(); }) return face.CurrentSide();
         else return face.MinusSide();
      }();
      // 1. Create base context using existing factory (preserves old convention)
      auto base_qctx = MakeQuadraturePointContext(
         kernel_context,
         wf_ctx,
         op_ctx,
         element_context,
         side,
         integrand,
         quad_index);

      // 2. Extract reference normal from face using existing API
      auto reference_normal = side.GetReferenceNormal();

      // 3. Compute cleaned-up facet geometry
      auto facet_geometry = ComputeFacetGeometry(
         base_qctx.inv_J_mesh,
         reference_normal,
         base_qctx.det_J);  // Volume determinant from base context

      // 4. Construct FacetQuadraturePointContext
      using PhysicalNormal = decltype(facet_geometry.normalized_physical_normal);

      BoundaryFacetQuadraturePointContext<
         QuadIndex,
         decltype(base_qctx.X),
         decltype(base_qctx.J_mesh),
         PhysicalNormal
      > facet_qctx;

      facet_qctx.quad_index = base_qctx.quad_index;
      facet_qctx.X = base_qctx.X;
      facet_qctx.inv_J_mesh = base_qctx.inv_J_mesh;
      facet_qctx.J_mesh = base_qctx.J_mesh;
      facet_qctx.det_J = base_qctx.det_J;
      facet_qctx.weight = base_qctx.weight;
      // Use the selected face qdata, including its fixed normal coordinate and
      // any reference subface map. A volume-rule lookup has the wrong point.
      constexpr auto DomainName = Integrand::domain_type::name;
      const auto& mesh_face_qdata = op_ctx.template GetMeshFacetQuadData<DomainName>();
      const auto& selected_qdata = GetFacetQuadData(mesh_face_qdata.MinusSide(), side);
      size_t reference_dimension = 0;
      quadrature_context_detail::BoundaryReferenceCoordinates(selected_qdata, quad_index,
         facet_qctx.reference_coordinates, reference_dimension);

      // Set facet-specific fields
      facet_qctx.det_J_facet =
         side.Measure() * facet_geometry.det_J_facet;
      facet_qctx.inverse_normal_cell_size = facet_geometry.inverse_normal_cell_size;
      facet_qctx.physical_normal = facet_geometry.normalized_physical_normal;

      return facet_qctx;
   }
   else if constexpr (InteriorFacetIntegrand<Integrand>)
   {
      return MakeLocalInteriorFacetQuadraturePointContext(
         kernel_context,
         wf_ctx,
         op_ctx,
         element_context,
         face,
         integrand,
         quad_index);
   }
}

} // namespace gendil
