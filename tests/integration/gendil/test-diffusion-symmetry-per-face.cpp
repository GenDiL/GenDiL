// Compare symmetry contributions on periodic meshes elongated along each axis.
// Periodic wrap and self-neighbor interfaces also contribute to each case.

#include <gendil/gendil.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace gendil;

// Vary the axis with two cells while retaining periodicity in every direction.
template<int FaceDir>  // 0=x, 1=y, 2=z
int TestSingleFaceDirection() {
    constexpr Integer Dim = 3;

    // Use two cells in the specified direction and one in the other directions.
    Integer nx = (FaceDir == 0) ? 2 : 1;
    Integer ny = (FaceDir == 1) ? 2 : 1;
    Integer nz = (FaceDir == 2) ? 2 : 1;

    const Real h = 0.5;
    CartesianMesh<Dim> mesh({nx, ny, nz}, {h, h, h}, {0.0, 0.0, 0.0}, true);

    constexpr Integer order = 1;
    FiniteElementOrders<order, order, order> orders;
    auto finite_element = MakeLobattoFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

    constexpr Integer num_quad_1d = order + 3;
    IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
    auto int_rules = MakeIntegrationRule(num_quads);

    const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

    Vector u_h(num_dofs);
    u_h = 0.0;
    u_h[0] = 1.0;

    Vector v_h_legacy(num_dofs), v_h_generic(num_dofs), v_h_no_symm(num_dofs);
    v_h_legacy = 0.0;
    v_h_generic = 0.0;
    v_h_no_symm = 0.0;

    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;

    using KernelPolicy = SerialKernelConfiguration;

    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    diffusion_operator(u_h, v_h_legacy);

    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);
    diffusion_no_symm(u_h, v_h_no_symm);

    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinates>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeIntegrationDomain<"mesh1">(mesh));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    const char* dir_name = (FaceDir == 0) ? "X" : (FaceDir == 1) ? "Y" : "Z";
    std::cout << "\n=== Testing periodic mesh elongated along " << dir_name << " (mesh: "
              << nx << "×" << ny << "×" << nz << ") ===\n";

    // Compare all entries, including zeros, and reject nonfinite output.
    const Real* reference = v_h_legacy_symm.ReadHostData();
    const Real* actual = v_h_generic.ReadHostData();
    Real max_error = 0.0;
    Real reference_scale = 0.0;
    for (Integer i = 0; i < num_dofs; ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            std::cerr << "Nonfinite symmetry contribution at DoF " << i << "\n";
            return 1;
        }
        max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
        reference_scale = std::max(reference_scale, std::abs(reference[i]));
    }
    if (!(reference_scale > 1.0e-12)) {
        std::cerr << "The fixture does not exercise a nonzero symmetry contribution.\n";
        return 1;
    }
    if (!(max_error <= 1.0e-11 * std::max(Real{1.0}, reference_scale))) {
        std::cerr << "Generic and legacy symmetry contributions differ: "
                  << max_error << "\n";
        return 1;
    }
    return 0;
}

int main() {
    std::cout << "Testing periodic two-cell meshes along each axis\n";
    std::cout << "Each case also includes wrap and self-neighbor interfaces.\n";

    int failures = 0;
    failures += TestSingleFaceDirection<0>();  // X-faces
    failures += TestSingleFaceDirection<1>();  // Y-faces
    failures += TestSingleFaceDirection<2>();  // Z-faces

    if (failures > 0) {
        std::cout << "\n" << failures << " periodic mesh case(s) failed.\n";
    } else {
        std::cout << "\nAll symmetry contributions match on the periodic meshes.\n";
    }

    return failures;
}
