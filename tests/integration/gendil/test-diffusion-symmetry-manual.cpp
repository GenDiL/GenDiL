// Print a comparison of symmetry contributions from two implementations.
// This diagnostic does not compute an independent numerical reference.

#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

int main() {
    // Periodic two-cell mesh with wrap interfaces in all directions.
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

    // On element 0's positive-x interface, canonical minus is element 1.
    // For a test basis supported on element 0, the average contributes one
    // half of its gradient; the neighbor's test trace is zero. With sigma=1,
    // mu=1, the row contribution is 0.5*(u_elem0-u_elem1)*grad_x(v_elem0).
    // Here the Q1 basis for u[0] vanishes on that positive-x face. Nonzero
    // output can instead come from the periodic wrap interfaces.
    std::cout << "On element 0's positive-x interface with element 1:\n";
    std::cout << "  - DSL minus is element 1; plus is element 0.\n";
    std::cout << "  - jump(u) = u_elem1 - u_elem0; Normal{} = [-1, 0, 0].\n";
    std::cout << "  - The element-0 test row receives 0.5*(u_elem0-u_elem1)*grad_x(v).\n";
    std::cout << "  - This input vanishes on that interface; periodic wrap faces also contribute.\n\n";

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
    std::cout << "  - Ratios compare complete periodic-mesh outputs, not individual faces.\n";
    std::cout << "  - This print-only diagnostic does not determine correctness.\n";

    return 0;
}
