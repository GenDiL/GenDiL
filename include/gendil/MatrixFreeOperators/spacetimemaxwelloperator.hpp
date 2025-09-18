// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"

namespace gendil {

//----------------------------------------------------------------------
// Space–Time Maxwell Fused DG Operator (no BC), generic NCOMP from FE
//----------------------------------------------------------------------

template <
    typename IntegrationRule,
    typename FaceIntegrationRulesTuple,
    typename KernelContext,
    typename FiniteElementSpace,
    typename MeshQuadData,
    typename MeshFaceDofToQuad,
    typename ElementQuadData,
    typename ElementFaceDofToQuad,
    typename Eps,
    typename Mu,
    typename DofsInView,
    typename DofsOutView
>
GENDIL_HOST_DEVICE
void MaxwellFusedOperator(
    const KernelContext &kernel_conf,
    const FiniteElementSpace &fe_space,
    const GlobalIndex element_index,
    const MeshQuadData &mesh_quad_data,
    const MeshFaceDofToQuad &mesh_face_quad_data,
    const ElementQuadData &element_quad_data,
    const ElementFaceDofToQuad &element_face_quad_data,
    Eps & eps,
    Mu & mu,
    const DofsInView &dofs_in,
    DofsOutView &dofs_out)
{
    using Mesh = typename FiniteElementSpace::mesh_type;
    using PhysicalCoordinates = typename Mesh::cell_type::physical_coordinates;
    using Jacobian = typename Mesh::cell_type::jacobian;

    // Determine dimensions
    constexpr Integer Dim = FiniteElementSpace::Dim;
    constexpr Integer SpaceDim = Dim - 1;
    constexpr Integer NCOMP    = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
    static_assert(NCOMP == 2 * SpaceDim, "Maxwell DG requires vector_dim == 2*SpaceDim (E and H components)");

    // Access the cell for Jacobian
    const auto cell = fe_space.GetCell(element_index);

    // Read local vector DoFs
    auto U = ReadDofs(kernel_conf, fe_space, element_index, dofs_in);

    // Prepare quad containers
    auto R_U   = MakeVectorQuadraturePointValuesContainer<NCOMP>(kernel_conf, IntegrationRule{});

    // Precompute spatial gradient at quad
    auto GradU = InterpolateGradient(kernel_conf, element_quad_data, U);

    // Volume contributions: time derivative + curl
    QuadraturePointLoop<IntegrationRule>(kernel_conf, [&](auto const &qi) {
        // Physical coords & Jacobian
        PhysicalCoordinates X;
        Jacobian J;
        cell.GetValuesAndJacobian(qi, mesh_quad_data, X, J);
        Jacobian invJ;
        Real detJ = ComputeInverseAndDeterminant(J, invJ);
        Real w = detJ * GetWeight(qi, element_quad_data);

        // Material
        Real eps_q = eps(X);
        Real mu_q  = mu(X);

        // Read gradients
        Real GU_q[NCOMP][Dim];
        ReadQuadratureLocalValues(kernel_conf, qi, GradU, GU_q);

        // Residual
        Real DU_q[NCOMP];
        // time derivative
        for(int c = 0; c < SpaceDim; ++c)
            DU_q[c]          = w * eps_q * GU_q[c][SpaceDim];
        for(int c = SpaceDim; c < NCOMP; ++c)
            DU_q[c]          = w * mu_q  * GU_q[c][SpaceDim];

        // split electric/magnetic gradients
        Real GE[SpaceDim][SpaceDim], GH[SpaceDim][SpaceDim];
        for(int c = 0; c < SpaceDim; ++c)
            for(int d = 0; d < SpaceDim; ++d) {
                GE[c][d] = GU_q[c][d];
                GH[c][d] = GU_q[SpaceDim + c][d];
            }
        // curls
        auto curlH = Curl(GH);
        auto curlE = Curl(GE);
        for(int c = 0; c < SpaceDim; ++c) {
            DU_q[c]           += w * curlH[c];
            DU_q[SpaceDim+c]  -= w * curlE[c];
        }

        WriteQuadratureLocalValues(kernel_conf, qi, DU_q, R_U);
    });

    // Apply test functions
    auto B_RU = ApplyTestFunctions(kernel_conf, element_quad_data, R_U);

    // Face contributions
    FaceLoop(fe_space, element_index,
        // interior
        [&](auto const &fi) {
            using FT = std::remove_reference_t<decltype(fi)>;
            constexpr int fidx = FT::local_face_index;
            constexpr int nidx = FT::neighbor_local_face_index;
            using FR = std::tuple_element_t<fidx, FaceIntegrationRulesTuple>;

            auto U_m = InterpolateValues(kernel_conf, std::get<fidx>(element_face_quad_data), U);
            auto U_p = ReadDofs(kernel_conf, fe_space, fi, dofs_in);
            auto U_pI= InterpolateValues(kernel_conf, std::get<nidx>(element_face_quad_data), U_p);

            QuadraturePointLoop<FR>(kernel_conf, [&](auto const &qf) {
                // ---- geometry & weights
                PhysicalCoordinates Xf;
                Jacobian Jf, invJf;
                cell.GetValuesAndJacobian(qf, std::get<fidx>(mesh_face_quad_data), Xf, Jf);
                ComputeInverseAndDeterminant(Jf, invJf);
                Real wf    = GetWeight(qf, std::get<fidx>(element_face_quad_data));

                // reference->physical space–time normal (not unit)
                auto n_ref  = GetReferenceNormal(fi);
                auto n_phys = ComputePhysicalNormal(invJf, n_ref); // length ≈ detJf

                // normalize the space–time normal used in the flux
                Real n_norm = 0;
                for (int d = 0; d < Dim; ++d) n_norm += n_phys[d]*n_phys[d];
                n_norm = std::sqrt(n_norm) + Real(1e-30); // guard
                Real nt_hat = n_phys[SpaceDim] / n_norm;
                std::array<Real,SpaceDim> ns_hat{};
                for (int d = 0; d < SpaceDim; ++d) ns_hat[d] = n_phys[d] / n_norm;

                // area-weighted quadrature factor
                Real wfJ = wf * n_norm; // (equivalent to wf * detJf)

                // traces
                Real Um[NCOMP], Up[NCOMP];
                ReadQuadratureLocalValues(kernel_conf, qf, U_m,  Um);
                ReadQuadratureLocalValues(kernel_conf, qf, U_pI, Up);

                // local wavespeed for dissipation (max on the two sides if coefficients differ)
                Real eps_m = eps(Xf), mu_m = mu(Xf);
                Real c_m = Real(1) / std::sqrt(eps_m * mu_m);
                // If eps/mu are discontinuous across the face and you can evaluate side-specific
                // values, do that and take c_hat = max(c_m, c_p). Otherwise:
                Real c_hat = c_m;

                // physical flux on each side, using the SAME normalized normal
                auto SpaceTimeFlux = [&](Real ntv,
                                        const std::array<Real,SpaceDim>& ns,
                                        const Real (&U)[NCOMP],
                                        Real (&F)[NCOMP]) {
                    // split
                    std::array<Real,SpaceDim> E{}, H{};
                    for (int i = 0; i < SpaceDim; ++i) { E[i] = U[i]; H[i] = U[SpaceDim+i]; }
                    auto nsxH = Cross(ns, H);
                    auto nsxE = Cross(ns, E);
                    for (int i = 0; i < SpaceDim; ++i) {
                        F[i]              = ntv * E[i] + nsxH[i];
                        F[SpaceDim + i]   = ntv * H[i] - nsxE[i];
                    }
                };

                Real Fm[NCOMP], Fp[NCOMP], Fq[NCOMP];
                SpaceTimeFlux(nt_hat, ns_hat, Um, Fm);
                SpaceTimeFlux(nt_hat, ns_hat, Up, Fp);

                // Rusanov/LF dissipation: s = |nt| + c*|ns|
                Real ns_mag = 0; for (int d=0; d<SpaceDim; ++d) ns_mag += ns_hat[d]*ns_hat[d];
                ns_mag = Sqrt(ns_mag);
                Real s = Abs(nt_hat) + c_hat * ns_mag;

                for (int i = 0; i < NCOMP; ++i) {
                    Real jump = Up[i] - Um[i];
                    Fq[i] = ( Real(0.5)*(Fm[i] + Fp[i]) - Real(0.5)*s*jump ) * wfJ;
                }

                WriteQuadratureLocalValues(kernel_conf, qf, Fq, U_m);
            });

            ApplyAddTestFunctions(kernel_conf, std::get<fidx>(element_face_quad_data), U_m, B_RU);
        },
        // boundary
        [&](auto const &fi) {
            // no-flux/BC
        }
    );

    // write back
    WriteDofs(kernel_conf, fe_space, element_index, B_RU, dofs_out);
}

//----------------------------------------------------------------------
// Explicit apply wrapper
//----------------------------------------------------------------------

template <
    typename KernelPolicy,
    typename IntegrationRule,
    typename FaceIntegrationRulesTuple,
    typename FiniteElementSpace,
    typename MeshQuadData,
    typename MeshFaceDofToQuad,
    typename ElementQuadData,
    typename ElementFaceDofToQuad,
    typename Eps,
    typename Mu,
    typename DofsInView,
    typename DofsOutView
>
void MaxwellExplicitOperator(
    const FiniteElementSpace &fe_space,
    const MeshQuadData &mesh_quad_data,
    const MeshFaceDofToQuad &mesh_face_quad_data,
    const ElementQuadData &element_quad_data,
    const ElementFaceDofToQuad &element_face_quad_data,
    Eps & eps,
    Mu & mu,
    const DofsInView &dofs_in,
    DofsOutView &dofs_out)
{
    mesh::CellIterator<KernelPolicy>(
        fe_space,
        [=] GENDIL_HOST_DEVICE(GlobalIndex idx) mutable {
            constexpr size_t shmem = required_shared_memory_v<KernelPolicy, IntegrationRule>;
            GENDIL_SHARED Real _smem[shmem];
            KernelContext<KernelPolicy, shmem> kctx(_smem);
            MaxwellFusedOperator< IntegrationRule, FaceIntegrationRulesTuple >
                (kctx, fe_space, idx,
                mesh_quad_data, mesh_face_quad_data,
                element_quad_data, element_face_quad_data,
                eps, mu,
                dofs_in, dofs_out);
        }
    );
}

//----------------------------------------------------------------------
// Operator class and factory
//----------------------------------------------------------------------

template <
    typename KernelPolicy,
    typename FiniteElementSpace,
    typename IntegrationRule,
    typename Eps,
    typename Mu
>
class MaxwellOperator
    : public MatrixFreeBilinearFiniteElementOperator<FiniteElementSpace, IntegrationRule>
{
    using base = MatrixFreeBilinearFiniteElementOperator<FiniteElementSpace, IntegrationRule>;
    Eps eps;
    Mu mu;

public:
    MaxwellOperator(const FiniteElementSpace &fes, const IntegrationRule &ir, Eps & eps, Mu & mu)
        : base(fes, ir),
        eps( eps ),
        mu( mu )
    { }

    template < typename Vector > // TODO replace with FiniteElementVector concept
    void operator()(const Vector &in, Vector &out) const {
        auto vin = MakeReadOnlyEVectorView<KernelPolicy>(this->finite_element_space, in);
        auto vout = MakeWriteOnlyEVectorView<KernelPolicy>(this->finite_element_space, out);
        MaxwellExplicitOperator<
            KernelPolicy,
            typename base::integration_rule,
            typename base::face_integration_rules
        >(
            this->finite_element_space,
            this->mesh_quad_data,
            this->mesh_face_quad_data,
            this->element_quad_data,
            this->element_face_quad_data,
            eps, mu,
            vin, vout
        );
    }

    #ifdef GENDIL_USE_MFEM
    void Mult(const mfem::Vector &in, mfem::Vector &out) const override {
        operator()(in, out);
    }
    #endif
};

// Factory
 template <
    typename KernelPolicy,
    typename FiniteElementSpace,
    typename IntegrationRule,
    typename Eps,
    typename Mu
>
auto MakeMaxwellOperator(const FiniteElementSpace &fes, const IntegrationRule &ir, Eps && eps, Mu && mu)
{
    return MaxwellOperator<KernelPolicy, FiniteElementSpace, IntegrationRule, Eps, Mu>(fes, ir, eps, mu);
}

} // namespace gendil
