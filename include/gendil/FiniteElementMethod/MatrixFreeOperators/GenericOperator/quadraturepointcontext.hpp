// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/computefacetgeometry.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"

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
   Real inverse_facet_size;   // Local facet size inverse: ||J^{-T} n_ref||
   PhysicalNormal physical_normal;  // Normalized physical normal (unit length)
};

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
   Real det_J_facet;          // Current/minus-side surface Jacobian determinant
   Real inverse_facet_size;   // Current/minus-side local facet size inverse
   PhysicalNormal physical_normal;

   GENDIL_HOST_DEVICE
   auto MinusSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, MinusJacobian, PhysicalNormal>
         qc_minus{
            quad_index, X, inv_J_mesh_minus, weight, det_J_facet, inverse_facet_size, physical_normal };
      return qc_minus;
   }

   GENDIL_HOST_DEVICE
   auto PlusSide() const
   {
      OneSidedFacetQuadraturePointContext<QuadIndex, PhysicalCoordinates, PlusJacobian, PhysicalNormal>
         qc_plus{
            quad_index, X, inv_J_mesh_plus, weight, det_J_facet, inverse_facet_size, physical_normal };
      return qc_plus;
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

   constexpr auto TrialName = requirements<I>::trial_name;
   static_assert(TrialName != StaticString("Error"),
      "MakeQuadraturePointContext: trial_name == \"Error\". Integrand must contain a TrialSpace.");

   // Get quad data from operator context (already bound to IntegrationRule when op_ctx was built)
   const auto& mesh_quad_data   = op_ctx.template mesh_quad_data<DomainName>();
   const auto& trial_quad_data  = op_ctx.template finite_element_quad_data<TrialName>();

   // Types for X and Jacobian from the domain mesh type
   const auto& mesh = wf_ctx.template domain<DomainName>();
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian            = typename Mesh::cell_type::jacobian;

   QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc{};

   qc.quad_index = quad_index;

   ec.cell.GetValuesAndJacobian(quad_index, mesh_quad_data, qc.X, qc.J_mesh);

   qc.det_J   = ComputeInverseAndDeterminant(qc.J_mesh, qc.inv_J_mesh);
   qc.weight = GetWeight(quad_index, trial_quad_data);

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

   constexpr auto TrialName = requirements<I>::trial_name;
   static_assert(TrialName != StaticString("Error"),
      "MakeQuadraturePointContext: trial_name == \"Error\". Integrand must contain a TrialSpace.");

   // Get quad data from operator context (already bound to IntegrationRule when op_ctx was built)
   const auto& mesh_quad_data   = op_ctx.template mesh_facet_quad_data<DomainName>();
   const auto& trial_quad_data  = op_ctx.template finite_element_facet_quad_data<TrialName>();

   // Types for X and Jacobian from the domain mesh type
   const auto& mesh = wf_ctx.template domain<DomainName>();
   using Mesh = std::remove_cvref_t<decltype(mesh)>;
   using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
   using Jacobian            = typename Mesh::cell_type::jacobian;

   QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc{};

   qc.quad_index = quad_index;

   // Jacobian J_mesh;
   // element_context.cell.GetValuesAndJacobian(quad_index, mesh_quad_data, qc.X, J_mesh);
   mesh::ComputePhysicalCoordinatesAndJacobian( element_context.cell, face, quad_index, mesh_quad_data, qc.X, qc.J_mesh );

   qc.det_J  = ComputeInverseAndDeterminant(qc.J_mesh, qc.inv_J_mesh);
   qc.weight = GetWeight( face, quad_index, trial_quad_data );

   return qc;
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
   if constexpr ( BoundaryFacetIntegrand<Integrand> )
   {
      // 1. Create base context using existing factory (preserves old convention)
      auto base_qctx = MakeQuadraturePointContext(
         kernel_context,
         wf_ctx,
         op_ctx,
         element_context,
         face.MinusSide(),
         integrand,
         quad_index);

      // 2. Extract reference normal from face using existing API
      auto reference_normal = face.MinusSide().GetReferenceNormal();

      // 3. Compute cleaned-up facet geometry
      auto facet_geometry = ComputeFacetGeometry(
         base_qctx.inv_J_mesh,
         reference_normal,
         base_qctx.det_J);  // Volume determinant from base context

      // 4. Construct FacetQuadraturePointContext
      using PhysicalNormal = decltype(facet_geometry.normalized_physical_normal);

      OneSidedFacetQuadraturePointContext<
         QuadIndex,
         decltype(base_qctx.X),
         decltype(base_qctx.J_mesh),
         PhysicalNormal
      > facet_qctx;

      facet_qctx.quad_index = base_qctx.quad_index;
      facet_qctx.X = base_qctx.X;
      facet_qctx.inv_J_mesh = base_qctx.inv_J_mesh;
      facet_qctx.weight = base_qctx.weight;

      // Set facet-specific fields
      facet_qctx.det_J_facet = facet_geometry.det_J_facet;
      facet_qctx.inverse_facet_size = facet_geometry.inverse_facet_size;
      facet_qctx.physical_normal = facet_geometry.normalized_physical_normal;

      return facet_qctx;
   }
   else if constexpr ( InteriorFacetIntegrand<Integrand> )
   {
      // Volume domains only for now: Cells<DomainName> must provide `static constexpr auto name`
      constexpr auto DomainName = Integrand::domain_type::name;

      constexpr auto TrialName = requirements<Integrand>::trial_name;
      static_assert(TrialName != StaticString("Error"),
         "MakeQuadraturePointContext: trial_name == \"Error\". Integrand must contain a TrialSpace.");

      // Get quad data from operator context (already bound to IntegrationRule when op_ctx was built)
      const auto& mesh_quad_data   = op_ctx.template mesh_facet_quad_data<DomainName>();
      const auto& trial_quad_data  = op_ctx.template finite_element_facet_quad_data<TrialName>();

      // Types for X and Jacobian from the domain mesh type
      const auto& mesh = wf_ctx.template domain<DomainName>();
      using Mesh = std::remove_cvref_t<decltype(mesh)>;
      using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
      using Jacobian            = typename Mesh::cell_type::jacobian;

      // Minus side
      QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc_minus{};
      qc_minus.quad_index = quad_index;
      mesh::ComputePhysicalCoordinatesAndJacobian(
         element_context.cell, face.MinusSide(), quad_index, mesh_quad_data, qc_minus.X, qc_minus.J_mesh );

      qc_minus.det_J  = ComputeInverseAndDeterminant(qc_minus.J_mesh, qc_minus.inv_J_mesh);
      qc_minus.weight = GetWeight( face.MinusSide(), quad_index, trial_quad_data );

      auto reference_normal_minus = face.MinusSide().GetReferenceNormal();
      auto facet_geometry_minus = ComputeFacetGeometry(
         qc_minus.inv_J_mesh,
         reference_normal_minus,
         qc_minus.det_J);
      
      using PhysicalNormal = decltype(facet_geometry_minus.normalized_physical_normal);

      if constexpr (requires_plus_side_jacobian_v<Integrand>)
      {
         QuadraturePointContext<QuadIndex, PhysicalCoordinates, Jacobian> qc_plus{};
         mesh::ComputePhysicalCoordinatesAndJacobian(
            face.plus_cell, face.PlusSide(), quad_index, mesh_quad_data, qc_plus.X, qc_plus.J_mesh );

         qc_plus.det_J  = ComputeInverseAndDeterminant(qc_plus.J_mesh, qc_plus.inv_J_mesh);

         TwoSidedFacetQuadraturePointContext<
            QuadIndex,
            decltype(qc_minus.X),
            decltype(qc_minus.inv_J_mesh),
            decltype(qc_plus.inv_J_mesh),
            PhysicalNormal
         > facet_qctx;

         facet_qctx.quad_index = qc_minus.quad_index;
         facet_qctx.X = qc_minus.X;
         facet_qctx.inv_J_mesh_minus = qc_minus.inv_J_mesh;
         facet_qctx.inv_J_mesh_plus = qc_plus.inv_J_mesh;
         facet_qctx.weight = qc_minus.weight;
         facet_qctx.det_J_facet = facet_geometry_minus.det_J_facet;
         facet_qctx.inverse_facet_size = facet_geometry_minus.inverse_facet_size;
         facet_qctx.physical_normal = facet_geometry_minus.normalized_physical_normal;

         return facet_qctx;
      }
      else
      {
         TwoSidedFacetQuadraturePointContext<
            QuadIndex,
            decltype(qc_minus.X),
            decltype(qc_minus.inv_J_mesh),
            Empty,
            PhysicalNormal
         > facet_qctx;

         facet_qctx.quad_index = qc_minus.quad_index;
         facet_qctx.X = qc_minus.X;
         facet_qctx.inv_J_mesh_minus = qc_minus.inv_J_mesh;
         facet_qctx.inv_J_mesh_plus = Empty{};
         facet_qctx.weight = qc_minus.weight;
         facet_qctx.det_J_facet = facet_geometry_minus.det_J_facet;
         facet_qctx.inverse_facet_size = facet_geometry_minus.inverse_facet_size;
         facet_qctx.physical_normal = facet_geometry_minus.normalized_physical_normal;

         return facet_qctx;
      }
   }
}

} // namespace gendil
