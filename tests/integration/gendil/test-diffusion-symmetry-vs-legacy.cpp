// Test symmetry term: generic vs legacy
// This will reveal the exact difference

#include <gendil/gendil.hpp>
#include <array>
#include <cmath>
#include <iostream>

using namespace std;
using namespace gendil;

int main() {
    const Integer n = 2;
    const Real h = 1.0 / n;
    CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0}, true);

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

    // Use unit vector: set only one DOF to 1.0
    const Integer test_dof = 0;
    u_h[test_dof] = 1.0;

    std::cout << "Testing with unit vector: u[" << test_dof << "] = 1.0, all others = 0.0\n";
    std::cout << "Total DOFs: " << num_dofs << "\n\n";

    Vector v_h_legacy(num_dofs);
    Vector v_h_generic(num_dofs);
    v_h_legacy = 0.0;
    v_h_generic = 0.0;

    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, 3> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;  // No penalty

    using KernelPolicy = SerialKernelConfiguration;

    // Legacy: full operator, but we'll only look at symmetry contribution
    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    diffusion_operator(u_h, v_h_legacy);

    // Generic: ONLY symmetry term
    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeIntegrationDomain<"mesh1">(fe_space));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    // Also test with volume + consistency (sigma=0)
    Vector v_h_no_symm(num_dofs);
    v_h_no_symm = 0.0;
    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);
    diffusion_no_symm(u_h, v_h_no_symm);

    // Legacy symmetry contribution = full - (volume+consistency)
    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    std::cout << "Non-zero DOF contributions:\n";
    std::cout << "Index | Legacy_Symm | Generic_Symm | Ratio | Diff\n";
    std::cout << "------+--------------+--------------+-------+--------------\n";
    int count = 0;
    for (int i = 0; i < num_dofs && count < 30; i++) {
        if (std::abs(v_h_legacy_symm[i]) > 1e-14 || std::abs(v_h_generic[i]) > 1e-14) {
            Real ratio = (std::abs(v_h_legacy_symm[i]) > 1e-14) ? (v_h_generic[i] / v_h_legacy_symm[i]) : 0.0;
            Real diff = v_h_generic[i] - v_h_legacy_symm[i];
            printf("%5d | %12.6e | %12.6e | %6.3f | %12.6e\n", i, v_h_legacy_symm[i], v_h_generic[i], ratio, diff);
            count++;
        }
    }
    if (count >= 30) std::cout << "... (showing first 30 non-zero entries)\n";

    // Compute norms
    Real norm_legacy = 0, norm_generic = 0;
    for (Integer i = 0; i < num_dofs; i++) {
        norm_legacy += v_h_legacy_symm[i] * v_h_legacy_symm[i];
        norm_generic += v_h_generic[i] * v_h_generic[i];
    }
    norm_legacy = std::sqrt(norm_legacy);
    norm_generic = std::sqrt(norm_generic);

    std::cout << "\n||legacy_symm|| = " << norm_legacy << "\n";
    std::cout << "||generic_symm|| = " << norm_generic << "\n";
    std::cout << "Ratio: " << (norm_generic / norm_legacy) << "\n";

    return 0;
}
