// Minimal periodic 1D test: 2 elements, including the wrap interface.
// Unit vector input to isolate the stencil

#include <gendil/gendil.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace gendil;

int main() {
    // 1D mesh: 2 elements, periodic
    constexpr Integer Dim = 1;
    const Integer n = 2;
    const Real h = 1.0 / n;
    CartesianMesh<Dim> mesh({n}, {h}, {0.0}, true);

    constexpr Integer order = 1;
    FiniteElementOrders<order> orders;
    auto finite_element = MakeLobattoFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

    constexpr Integer num_quad_1d = order + 3;
    IntegrationRuleNumPoints<num_quad_1d> num_quads;
    auto int_rules = MakeIntegrationRule(num_quads);

    const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
    std::cout << "1D mesh: " << n << " elements, " << num_dofs << " DOFs\n";

    Vector u_h(num_dofs);
    u_h = 0.0;
    u_h[0] = 1.0;  // Unit vector at first DOF

    Vector v_h_legacy(num_dofs), v_h_generic(num_dofs), v_h_no_symm(num_dofs);
    v_h_legacy = 0.0;
    v_h_generic = 0.0;
    v_h_no_symm = 0.0;

    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;

    using KernelPolicy = SerialKernelConfiguration;

    // Legacy full
    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    diffusion_operator(u_h, v_h_legacy);

    // Legacy without symmetry (sigma=0)
    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);
    diffusion_no_symm(u_h, v_h_no_symm);

    // Generic symmetry only
    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinates>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeIntegrationDomain<"mesh1">(mesh));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    // Compute legacy symmetry = full - (volume+consistency)
    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    std::cout << "\nInput: u[0]=1.0, all others=0\n";
    std::cout << "\nDOF | Legacy_Symm | Generic_Symm | Ratio\n";
    std::cout << "----+-------------+--------------+-------\n";
    for (Integer i = 0; i < num_dofs; i++) {
        Real ratio = (std::abs(v_h_legacy_symm[i]) > 1e-14) ? (v_h_generic[i] / v_h_legacy_symm[i]) : 0.0;
        printf("%3d | %11.6f | %12.6f | %6.3f\n", (int)i, v_h_legacy_symm[i], v_h_generic[i], ratio);
    }

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
