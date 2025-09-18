// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

/**
 * Cross-dimensional Euler–Vlasov face operator (configuration ⟷ phase space).
 *
 * Minus (low-dim) side: Euler (vector of 5 conserved components).
 * Plus  (high-dim) side: Vlasov (scalar distribution f).
 *
 * Numerical flux:
 *   v_n = v · n_x (n_x is the spatial face normal from Euler side)
 *   f_up = (v_n > 0 ? f_plus : Maxwellian(U_minus))
 *
 * Phase-side residual:  dS_x dv * v_n * f_up  tested with high-dim face basis.
 * Euler-side residuals (moments): aggregate over velocity at quad level and
 *   write into low-dim face quadrature slots, then apply low-dim test functions:
 *     mass:     ∫ v_n f_up dv
 *     momentum: ∫ v_n v_i f_up dv  (i=0..LowDim-1)
 *     energy:   ∫ v_n (1/2 |v|^2) f_up dv
 */
template <
   // Execution
   typename KernelContext,
   // Spaces
   typename FiniteElementSpaceEuler,   // Dim = LowDim
   typename FiniteElementSpaceVlasov,  // Dim = HighDim
   // Face mesh connectivity (e.g. CartesianCrossDimConformingIntermesh<LowDim,HighDim,LocalFaceIndexLow>)
   typename FaceMesh,
   // Face integration rules
   typename FaceIntegrationRulesEuler,   // for low-dim faces
   typename FaceIntegrationRulesVlasov,  // for high-dim faces
   // Mesh face quad-data
   typename MeshFaceDofToQuadEuler,
   typename MeshFaceDofToQuadVlasov,
   // FE face quad-data (trace interpolation/testing)
   typename ElementFaceDofToQuadEuler,
   typename ElementFaceDofToQuadVlasov,
   // Euler inputs (minus): rho, momentum, rhoE (passed separately)
   typename InputRhoMinus,
   typename InputMomMinus,
   typename InputEnergyMinus,
   // Vlasov inputs (plus): f
   typename InputFPlus,
   // Euler outputs (minus): rho, momentum, rhoE
   typename OutputRhoMinus,
   typename OutputMomMinus,
   typename OutputEnergyMinus,
   // Vlasov outputs (plus): f
   typename OutputFPlus
>
GENDIL_HOST_DEVICE
void EulerVlasovCrossDimLocalFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpaceEuler  & euler_fe_space_minus,
   const FiniteElementSpaceVlasov & vlasov_fe_space_plus,
   const FaceMesh & face_mesh,
   const FaceIntegrationRulesEuler  & face_rules_euler,
   const FaceIntegrationRulesVlasov & face_rules_vlasov,
   const GlobalIndex face_index,
   const MeshFaceDofToQuadEuler  & mesh_face_qdata_euler,
   const MeshFaceDofToQuadVlasov & mesh_face_qdata_vlasov,
   const ElementFaceDofToQuadEuler  & face_qdata_euler,
   const ElementFaceDofToQuadVlasov & face_qdata_vlasov,
   // Inputs
   const InputRhoMinus    & dofs_rho_minus_in,
   const InputMomMinus    & dofs_mom_minus_in,
   const InputEnergyMinus & dofs_rhoE_minus_in,
   const InputFPlus       & dofs_f_plus_in,
   // Outputs (accumulate)
   OutputRhoMinus    & dofs_rho_minus_out,
   OutputMomMinus    & dofs_mom_minus_out,
   OutputEnergyMinus & dofs_rhoE_minus_out,
   OutputFPlus       & dofs_f_plus_out,
   // Thermo params
   const Real gamma = 1.4,   // ratio of specific heats
   const Real Rgas  = 1.0    // gas constant (non-dimensional)
)
{
   constexpr Integer LowDim  = FiniteElementSpaceEuler ::Dim;
   constexpr Integer HighDim = FiniteElementSpaceVlasov::Dim;
   static_assert(HighDim > LowDim, "Vlasov (high-dim) space must have extra velocity dims.");
   constexpr Integer dvel = HighDim - LowDim;
   static_assert(dvel == LowDim, "Configuration space and velocity space should have the same dimension (2D-2V or 3D-3V).");

   // Pull the pair of (minus, plus) cells from connectivity
   const auto face_info = face_mesh.GetGlobalFaceInfo(face_index);

   // Read Euler component DOFs on the minus side (per-cell views)
   auto u_rho  = ReadDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), dofs_rho_minus_in);
   auto u_mom  = ReadDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), dofs_mom_minus_in);
   auto u_rhoE = ReadDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), dofs_rhoE_minus_in);

   // Read Vlasov DOFs on the plus side
   auto u_f = ReadDofs(kernel_conf, vlasov_fe_space_plus, face_info.plus_side(), dofs_f_plus_in);

   // Interpolate traces to face quadrature points
   auto Brho_minus  = InterpolateValues(kernel_conf, face_info.minus_side(), face_qdata_euler,  u_rho );
   auto Bmom_minus  = InterpolateValues(kernel_conf, face_info.minus_side(), face_qdata_euler,  u_mom );
   auto BrhoE_minus = InterpolateValues(kernel_conf, face_info.minus_side(), face_qdata_euler,  u_rhoE);

   auto Bf_plus     = InterpolateValues(kernel_conf, face_info.plus_side(),  face_qdata_vlasov, u_f   );

   // Cells (for geometry / mappings)
   const auto low_cell  = euler_fe_space_minus .GetCell(face_info.minus_side().get_cell_index());
   const auto high_cell = vlasov_fe_space_plus.GetCell(face_info.plus_side().get_cell_index());

   // Choose face integration rules on each side
   const auto int_rule_low  = GetFaceIntegrationRule(face_info.minus_side(), face_rules_euler );
   const auto int_rule_high = GetFaceIntegrationRule(face_info.plus_side(),  face_rules_vlasov);

   // Phase side: container for scalar face contribution (Df = dS_x dv * v_n * f_up)
   auto Df_high = MakeQuadraturePointValuesContainer(kernel_conf, int_rule_high);

   // Euler side: we accumulate moments at the LOW face quadrature points by
   // writing into low-dim face containers from the high-dim loop via sub-index.
   auto Dmass_shared = MakeSharedQuadraturePointValuesContainer(kernel_conf, int_rule_low);
   auto Dmom_shared  = MakeSharedQuadraturePointValuesContainer<LowDim>(kernel_conf, int_rule_low);
   auto DE_shared    = MakeSharedQuadraturePointValuesContainer(kernel_conf, int_rule_low);
   QuadraturePointLoop( kernel_conf, int_rule_low, [&] ( auto const & quad_index )
   {
      WriteQuadratureLocalValues( kernel_conf, quad_index, 0.0, Dmass_shared );
      Real z[LowDim]; for(int i=0;i<LowDim;++i) z[i]=0.0;
      WriteQuadratureLocalValues( kernel_conf, quad_index, z, Dmom_shared );
      WriteQuadratureLocalValues( kernel_conf, quad_index, 0.0, DE_shared );
   } );
   kernel_conf.Synchronize();

   // Loop high-dim face quadrature, aggregate to both sides
   QuadraturePointLoop(kernel_conf, int_rule_high, [&] (auto const& qh)
   {
      // ---------- Geometry on LOW face (for normal) using sub-index ----------
      using MeshLow  = typename FiniteElementSpaceEuler ::mesh_type;
      using MeshHigh = typename FiniteElementSpaceVlasov::mesh_type;
      using XCoord   = typename MeshLow ::cell_type::physical_coordinates;
      using YCoord   = typename MeshHigh::cell_type::physical_coordinates;
      using JLow     = typename MeshLow ::cell_type::jacobian;
      using JHigh    = typename MeshHigh::cell_type::jacobian;

      auto ql = qh.template Sub<0, LowDim>();

      XCoord X;  JLow  Jl;
      mesh::ComputePhysicalCoordinatesAndJacobian(low_cell,  face_info.minus_side(), ql, mesh_face_qdata_euler,  X, Jl);

      // Normal from low side
      JLow  invJl;
      (void) ComputeInverseAndDeterminant(Jl, invJl); // detJ_low not needed for measure in this kernel
      const auto ref_n = face_info.get_reference_normal();
      const auto n_phys = ComputePhysicalNormal(invJl, ref_n);  // spatial normal in R^{LowDim}

      // ---------- Geometry on HIGH face (for total measure and velocity coords) ----------
      YCoord Y;  JHigh Jh;
      JHigh  invJh;
      mesh::ComputePhysicalCoordinatesAndJacobian(high_cell, face_info.plus_side(),  qh, mesh_face_qdata_vlasov, Y, Jh);
      const Real detJ_high = ComputeInverseAndDeterminant(Jh, invJh);
      const Real w_high    = GetWeight(face_info.plus_side(), qh, face_qdata_vlasov);
      const Real dmeas     = detJ_high * w_high; // dS_x * dv (combined, consistent with high-dim face rule)

      // ---------- Extract velocity and compute v_n ----------
      Real v[dvel];
      for (int k = 0; k < dvel; ++k) { v[k] = Y[LowDim + k]; }

      Real vn = 0.0;
      for (int i = 0; i < LowDim; ++i) { vn += v[i] * n_phys[i]; }

      // ---------- Euler trace (minus) at the low face point ql ----------
      const Real rho   = ReadQuadratureLocalValues(kernel_conf, ql, Brho_minus );
      const auto mom   = ReadQuadratureLocalValues(kernel_conf, ql, Bmom_minus );
      const Real rhoE  = ReadQuadratureLocalValues(kernel_conf, ql, BrhoE_minus);

      // Primitive vars
      const Real inv_rho = 1.0 / Max(rho, 1e-14); // safeguard
      Real u[LowDim];
      for (size_t i = 0; i < LowDim; i++)
      {
         u[i] = mom[i] * inv_rho;
      }
      Real kinetic = 0.0;
      for (size_t i = 0; i < LowDim; i++)
      {
         kinetic += u[i] * u[i];
      }
      kinetic *= 0.5;
      const Real e_int   = Max(rhoE * inv_rho - kinetic, 1e-14);
      const Real p       = (gamma - 1.0) * rho * e_int;
      const Real RT      = Max(p * inv_rho / Rgas, 1e-14);

      // Maxwellian M(U-) at this (x,v)
      //   M = rho / ( (2*pi*RT)^(dvel/2) ) * exp( -|v - u|^2 / (2 RT) )
      Real dv_minus_u_sq = 0.0;
      {
         // use LowDim components of u; if dvel > LowDim, the extra are treated as zero mean
         for (int i = 0; i < LowDim; ++i)
         {
               const Real dvux = v[i] - u[i];
               dv_minus_u_sq += dvux * dvux;
         }
      }
      const Real norm_pref = Pow(2.0 * M_PI * RT, -0.5 * Real(dvel));
      const Real Mq = rho * norm_pref * Exp( - dv_minus_u_sq / (2.0 * RT) );

      // Vlasov trace (plus)
      const Real f_plus_q = ReadQuadratureLocalValues(kernel_conf, qh, Bf_plus);

      // Upwind distribution at the interface
      const Real f_up = (vn > 0.0) ? f_plus_q : Mq;

      // ---------- Write phase-side scalar contribution ----------
      const Real Df_q = dmeas * vn * f_up;
      WriteQuadratureLocalValues(kernel_conf, qh, Df_q, Df_high);

      // ---------- Write fluid-side contributions ----------------
      cont Real mass = Df_q;
      Real mom[LowDim];
      for (int i = 0; i < LowDim; ++i) { mom[i] = Df_q * v[i]; }
      Real v2 = 0.0;
      for (int k = 0; k < dvel; ++k) v2 += v[k]*v[k];
      Real energy = Df_q * 0.5 * v2;
      WriteAddQuadratureLocalValues( kernel_conf, ql, mass, Dmass_shared );
      WriteAddQuadratureLocalValues( kernel_conf, ql, mom, Dmom_shared );
      WriteAddQuadratureLocalValues( kernel_conf, ql, energy, DE_shared );
   });

   // -------- Apply test functions and scatter to DoFs --------

   // Vlasov (plus) side: subtract (to keep antisymmetry across the interface)
   {
      auto BDf = ApplyTestFunctions(kernel_conf, face_info.plus_side(), face_qdata_vlasov, Df_high);
      WriteSubDofs(kernel_conf, vlasov_fe_space_plus, face_info.plus_side(), BDf, dofs_f_plus_out);
   }

   // Euler (minus) side: add contributions to each conserved component
   kernel_conf.Synchronize();
   {
      auto Dmass = MakeQuadraturePointValuesContainer(kernel_conf, int_rule_low);
      auto Dmom = MakeQuadraturePointValuesContainer<LowDim>(kernel_conf, int_rule_low);
      auto DE = MakeQuadraturePointValuesContainer(kernel_conf, int_rule_low);
      QuadraturePointLoop( kernel_conf, int_rule_low, [&] ( auto const & quad_index )
      {
         const auto Dmass_q = Apply( Dmass_shared, quad_index );
         const auto Dmom_q  = Apply( Dmom_shared,  quad_index );
         const auto DE_q    = Apply( DE_shared,    quad_index );
         WriteQuadratureLocalValues( kernel_conf, quad_index, Dmass_q, Dmass );
         WriteQuadratureLocalValues( kernel_conf, quad_index, Dmom_q,  Dmom );
         WriteQuadratureLocalValues( kernel_conf, quad_index, DE_q,    DE );
      });
      kernel_conf.Synchronize();
      kernel_conf.SharedAllocator.reset();

      auto Bmass = ApplyTestFunctions(kernel_conf, face_info.minus_side(), face_qdata_euler, Dmass);
      auto Bmom  = ApplyTestFunctions(kernel_conf, face_info.minus_side(), face_qdata_euler, Dmom);
      auto BE    = ApplyTestFunctions(kernel_conf, face_info.minus_side(), face_qdata_euler, DE);
      WriteAddDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), Bmass, dofs_rho_minus_out );
      WriteAddDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), Bmom,  dofs_mom_minus_out );
      WriteAddDofs(kernel_conf, euler_fe_space_minus, face_info.minus_side(), BE,    dofs_rhoE_minus_out );
   }

}

/**
 * @brief Explicit cross-dimension face operator (Euler ⟷ Vlasov).
 * Launches EulerVlasovCrossDimLocalFaceOperator over all faces in the FaceMesh.
 *
 */
template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FaceIntegrationRulesEuler,   // decltype(GetFaceIntegrationRules(IntegrationRule{}))
   typename FaceIntegrationRulesVlasov,  // decltype(GetFaceIntegrationRules(IntegrationRule{}))
   typename FiniteElementSpaceEuler,
   typename FiniteElementSpaceVlasov,
   typename FaceMesh,
   typename MeshFaceDofToQuadEuler,
   typename MeshFaceDofToQuadVlasov,
   typename ElementFaceDofToQuadEuler,
   typename ElementFaceDofToQuadVlasov,
   // Inputs
   typename InputRhoMinus,
   typename InputMomMinus,
   typename InputEnergyMinus,
   typename InputFPlus,
   // Outputs
   typename OutputRhoMinus,
   typename OutputMomMinus,
   typename OutputEnergyMinus,
   typename OutputFPlus>
void EulerVlasovExplicitCrossDimFaceOperator(
   const FiniteElementSpaceEuler  & fe_space_euler,
   const FiniteElementSpaceVlasov & fe_space_vlasov,
   const FaceMesh & face_mesh,
   const MeshFaceDofToQuadEuler  & mesh_face_quad_data_euler,
   const MeshFaceDofToQuadVlasov & mesh_face_quad_data_vlasov,
   const ElementFaceDofToQuadEuler  & element_face_quad_data_euler,
   const ElementFaceDofToQuadVlasov & element_face_quad_data_vlasov,
   const InputRhoMinus    & dofs_rho_minus_in,
   const InputMomMinus    & dofs_mom_minus_in,
   const InputEnergyMinus & dofs_rhoE_minus_in,
   const InputFPlus       & dofs_f_plus_in,
   OutputRhoMinus    & dofs_rho_minus_out,
   OutputMomMinus    & dofs_mom_minus_out,
   OutputEnergyMinus & dofs_rhoE_minus_out,
   OutputFPlus       & dofs_f_plus_out,
   const Real gamma = 1.4,
   const Real Rgas  = 1.0 )
{
   mesh::GlobalFaceIterator<KernelConfiguration>(
      face_mesh,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex face_index ) mutable
      {
         constexpr size_t required_shared_mem =
            required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         EulerVlasovCrossDimLocalFaceOperator(
            kernel_conf,
            fe_space_euler,
            fe_space_vlasov,
            face_mesh,
            FaceIntegrationRulesEuler{},
            FaceIntegrationRulesVlasov{},
            face_index,
            mesh_face_quad_data_euler,
            mesh_face_quad_data_vlasov,
            element_face_quad_data_euler,
            element_face_quad_data_vlasov,
            dofs_rho_minus_in,
            dofs_mom_minus_in,
            dofs_rhoE_minus_in,
            dofs_f_plus_in,
            dofs_rho_minus_out,
            dofs_mom_minus_out,
            dofs_rhoE_minus_out,
            dofs_f_plus_out,
            gamma,
            Rgas );
      }
   );
}


/**
 * @brief Class wrapper mirroring your AdvectionNonconformingFaceOperator, but for
 * Euler (low-dim) ↔ Vlasov (high-dim) coupling across cross-dimension faces.
 *
 * Stores the FE spaces, face mesh, and face quadrature data; exposes Apply(...)
 * that takes separate input/output views for Euler (rho, mom, rhoE) and Vlasov (f).
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpaceEuler,
   typename FiniteElementSpaceVlasov,
   typename FaceMesh,
   typename IntegrationRuleEuler,
   typename IntegrationRuleVlasov>
class EulerVlasovCrossDimFaceOperator
{
   // FE spaces
   using fe_euler_t  = FiniteElementSpaceEuler;
   using fe_vlasov_t = FiniteElementSpaceVlasov;

   // Integration rules (derive face rules per side from the same volume rule)
   using integration_rule_euler_t = IntegrationRuleEuler;
   using integration_rule_vlasov_t = IntegrationRuleVlasov;
   using face_rules_euler_t  = decltype( GetFaceIntegrationRules( integration_rule_euler_t{} ) );
   using face_rules_vlasov_t = decltype( GetFaceIntegrationRules( integration_rule_vlasov_t{} ) );

   // Mesh types
   using mesh_euler_t  = typename fe_euler_t ::mesh_type;
   using mesh_vlasov_t = typename fe_vlasov_t::mesh_type;

   // Mesh face quadrature data
   using mesh_face_q_euler_t =
      decltype( MakeMeshFaceQuadData< mesh_euler_t >( face_rules_euler_t{} ) );
   using mesh_face_q_vlasov_t =
      decltype( MakeMeshFaceQuadData< mesh_vlasov_t >( face_rules_vlasov_t{} ) );

   // FE face DoF→Quad maps
   using fe_euler_shape  = typename fe_euler_t ::finite_element_type::shape_functions;
   using fe_vlasov_shape = typename fe_vlasov_t::finite_element_type::shape_functions;

   using element_face_q_euler_t  = decltype( MakeFaceDofToQuad<fe_euler_shape >( face_rules_euler_t{}  ) );
   using element_face_q_vlasov_t = decltype( MakeFaceDofToQuad<fe_vlasov_shape>( face_rules_vlasov_t{} ) );

   // References to spaces and face set(s)
   const fe_euler_t  & fe_euler;
   const fe_vlasov_t & fe_vlasov;
   const FaceMesh    & face_meshes;

   // Stored quad data
   mesh_face_q_euler_t   mesh_face_q_euler;
   mesh_face_q_vlasov_t  mesh_face_q_vlasov;
   element_face_q_euler_t   element_face_q_euler;
   element_face_q_vlasov_t  element_face_q_vlasov;

   // Thermo
   const Real gamma;
   const Real Rgas;

public:
   EulerVlasovCrossDimFaceOperator( const fe_euler_t  & fe_euler_,
                                    const fe_vlasov_t & fe_vlasov_,
                                    const FaceMesh    & face_mesh_,
                                    const IntegrationRuleEuler & int_rule_euler,
                                    const IntegrationRuleVlasov & int_rule_vlasov,
                                    const Real gamma_ = 1.4,
                                    const Real Rgas_  = 1.0 )
   : fe_euler(fe_euler_)
   , fe_vlasov(fe_vlasov_)
   , face_meshes(face_mesh_)
   , mesh_face_q_euler{}
   , mesh_face_q_vlasov{}
   , element_face_q_euler{}
   , element_face_q_vlasov{}
   , gamma(gamma_)
   , Rgas(Rgas_)
   {
      // Nothing else to do: mesh_face_q_* and element_face_q_* default-construct from types
      // that capture face rules statically (like your advection operator does).
      (void)int_rule; // kept for API parity with your Advection* class
   }

   /**
    * @brief Apply the cross-dim face coupling on all interface faces.
    * The input/output types are the usual GenDiL element-vector StridedView’s.
    */
   template <
      typename InputRhoMinus,
      typename InputMomMinus,
      typename InputEnergyMinus,
      typename InputFPlus,
      typename OutputRhoMinus,
      typename OutputMomMinus,
      typename OutputEnergyMinus,
      typename OutputFPlus >
   void Apply( const InputRhoMinus    & dofs_rho_minus_in,
               const InputMomMinus    & dofs_mom_minus_in,
               const InputEnergyMinus & dofs_rhoE_minus_in,
               const InputFPlus       & dofs_f_plus_in,
               OutputRhoMinus    & dofs_rho_minus_out,
               OutputMomMinus    & dofs_mom_minus_out,
               OutputEnergyMinus & dofs_rhoE_minus_out,
               OutputFPlus       & dofs_f_plus_out ) const
   {
      // Restriction check parity with your style (optional):
      static_assert(
         std::is_same_v< typename fe_euler_t::restriction_type, L2Restriction >,
         "EulerVlasovCrossDimFaceOperator::Apply requires L2Restriction on Euler space" );
      static_assert(
         std::is_same_v< typename fe_vlasov_t::restriction_type, L2Restriction >,
         "EulerVlasovCrossDimFaceOperator::Apply requires L2Restriction on Vlasov space" );

      mesh::ForEachFaceMesh(
         face_meshes,
         [&] ( const auto & face_mesh ) mutable
         {
            EulerVlasovExplicitCrossDimFaceOperator<
               KernelPolicy,
               IntegrationRule,
               face_rules_euler_t,
               face_rules_vlasov_t >(
                  fe_euler,
                  fe_vlasov,
                  face_mesh,
                  mesh_face_q_euler,
                  mesh_face_q_vlasov,
                  element_face_q_euler,
                  element_face_q_vlasov,
                  dofs_rho_minus_in,
                  dofs_mom_minus_in,
                  dofs_rhoE_minus_in,
                  dofs_f_plus_in,
                  dofs_rho_minus_out,
                  dofs_mom_minus_out,
                  dofs_rhoE_minus_out,
                  dofs_f_plus_out,
                  gamma,
                  Rgas );
         }
      );
   }
};

template <
   typename KernelPolicy,
   typename FiniteElementSpaceEuler,
   typename FiniteElementSpaceVlasov,
   typename FaceMesh,
   typename IntegrationRuleEuler,
   typename IntegrationRuleVlasov >
auto MakeEulerVlasovCrossDimFaceOperator( const FiniteElementSpaceEuler  & fe_euler,
                                          const FiniteElementSpaceVlasov & fe_vlasov,
                                          const FaceMesh                 & face_mesh,
                                          const IntegrationRuleEuler     & int_rule_euler,
                                          const IntegrationRuleVlasov    & int_rule_vlasov,
                                          const Real gamma = 1.4,
                                          const Real Rgas  = 1.0 )
{
   return EulerVlasovCrossDimFaceOperator<
      KernelPolicy,
      FiniteElementSpaceEuler,
      FiniteElementSpaceVlasov,
      FaceMesh,
      IntegrationRuleEuler,
      IntegrationRuleVlasov
   >( fe_euler, fe_vlasov, face_mesh, int_rule_euler, int_rule_vlasov, gamma, Rgas );
}

} // namespace gendil
