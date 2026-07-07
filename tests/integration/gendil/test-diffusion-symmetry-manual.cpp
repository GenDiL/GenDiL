// Manually compute the symmetry term to compare with both implementations
// This helps identify if the bug is in integrand evaluation or test function application

#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

int main() {
    // Simplest case: 2 elements in x-direction only
    constexpr Integer Dim = 3;
    const Integer nx = 2, ny = 1, nz = 1;
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

    std::cout << "2×1×1 mesh: 2 elements, " << num_dofs << " DOFs\n";
    std::cout << "Order " << order << ": " << ((order+1)*(order+1)*(order+1))
              << " DOFs per element\n\n";

    Vector u_h(num_dofs);
    u_h = 0.0;
    u_h[0] = 1.0;  // Element 0, local DOF 0

    std::cout << "Input: u[0]=1, all others=0\n\n";

    // Element 0 has one interior face (face 3 = +x direction)
    // This face connects element 0 to element 1
    // Face normal in reference space: [1, 0, 0]

    // For the symmetry term: sigma * jump(u) * average(mu * dot(grad(v), n))
    // With sigma=1, mu=1:
    //   jump(u) = u_elem0 - u_elem1
    //   average(mu * n_x) = 0.5 * (1 * 1 + 1 * 1) = 1.0 (on Cartesian mesh, n_x = 1)
    //
    // At quadrature points on the face:
    //   u_elem0 varies (from basis functions)
    //   u_elem1 = 0 (since u[8:15]=0)
    //
    // So the contribution is: jump(u) * 1.0 * test_gradient_x

    std::cout << "Expected behavior on face 3 of element 0 (connecting to element 1):\n";
    std::cout << "  - jump(u) = u_minus - u_plus = u_elem0 - u_elem1\n";
    std::cout << "  - Since u_elem1 DOFs are all 0, jump(u) = u_elem0 values\n";
    std::cout << "  - Normal = [1, 0, 0] (pointing in +x)\n";
    std::cout << "  - Contribution: jump(u) * average(mu * grad_x(v) * n_x)\n";
    std::cout << "  - On Cartesian mesh: average(mu * n_x) = 1.0\n";
    std::cout << "  - So: contribution = jump(u) * 1.0 * grad_x(v)\n\n";

    std::cout << "The test gradients in x-direction should produce contributions\n";
    std::cout << "to the DOFs based on which basis functions have non-zero x-gradients\n";
    std::cout << "on this face.\n\n";

    // Run both implementations
    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;

    using KernelPolicy = SerialKernelConfiguration;

    Vector v_h_legacy(num_dofs), v_h_generic(num_dofs), v_h_no_symm(num_dofs);
    v_h_legacy = 0.0;
    v_h_generic = 0.0;
    v_h_no_symm = 0.0;

    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    diffusion_operator(u_h, v_h_legacy);

    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);
    diffusion_no_symm(u_h, v_h_no_symm);

    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeIntegrationDomain<"mesh1">(fe_space));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    std::cout << "Results (non-zero DOFs only):\n";
    std::cout << "DOF | Element | Legacy_Symm | Generic_Symm | Ratio\n";
    std::cout << "----+---------+-------------+--------------+-------\n";
    for (Integer i = 0; i < num_dofs; i++) {
        if (std::abs(v_h_legacy_symm[i]) > 1e-14 || std::abs(v_h_generic[i]) > 1e-14) {
            int elem = i / 8;
            Real ratio = (std::abs(v_h_legacy_symm[i]) > 1e-14)
                         ? (v_h_generic[i] / v_h_legacy_symm[i]) : 0.0;
            printf("%3d | %7d | %11.6f | %12.6f | %6.3f\n",
                   (int)i, elem, v_h_legacy_symm[i], v_h_generic[i], ratio);
        }
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "  - DOFs 0-7 belong to element 0 (where u[0]=1)\n";
    std::cout << "  - DOFs 8-15 belong to element 1 (where all u=0)\n";
    std::cout << "  - If element 1 shows exact -1.0 ratio, there's a sign flip\n";
    std::cout << "  - The sign flip could be in:\n";
    std::cout << "    1. How jump(u) is computed\n";
    std::cout << "    2. How the normal vector is applied\n";
    std::cout << "    3. How gradient test functions are applied\n";

    return 0;
}
