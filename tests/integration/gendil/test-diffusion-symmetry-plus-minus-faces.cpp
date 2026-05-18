// Test if the bug is related to "minus" faces (0,1,2) vs "plus" faces (3,4,5)
// Face indices: 0=-x, 1=-y, 2=-z, 3=+x, 4=+y, 5=+z

#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

int main() {
    // 3D mesh with 2 elements in each direction
    // This creates faces in all 6 directions
    constexpr Integer Dim = 3;
    const Integer n = 2;
    const Real h = 0.5;
    CartesianMesh<Dim> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0}, true);

    constexpr Integer order = 1;
    FiniteElementOrders<order, order, order> orders;
    auto finite_element = MakeLobattoFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

    constexpr Integer num_quad_1d = order + 3;
    IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
    auto int_rules = MakeIntegrationRule(num_quads);

    const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

    std::cout << "Testing with 2×2×2 mesh (" << (n*n*n) << " elements, "
              << num_dofs << " DOFs)\n";
    std::cout << "Setting u[i]=1 for each DOF i and checking which elements show -1.0 ratio\n\n";

    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;

    using KernelPolicy = SerialKernelConfiguration;

    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);

    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeDomain<"mesh1">(mesh));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);

    // Test with u[0]=1 (element 0, which is at grid position (0,0,0))
    Vector u_h(num_dofs);
    u_h = 0.0;
    u_h[0] = 1.0;

    Vector v_h_legacy(num_dofs), v_h_generic(num_dofs), v_h_no_symm(num_dofs);
    v_h_legacy = 0.0;
    v_h_generic = 0.0;
    v_h_no_symm = 0.0;

    diffusion_operator(u_h, v_h_legacy);
    diffusion_no_symm(u_h, v_h_no_symm);
    generic_operator(u_h, v_h_generic);

    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    std::cout << "Input: u[0]=1 (element 0 at grid position (0,0,0))\n\n";
    std::cout << "Element 0 neighbors:\n";
    std::cout << "  Face 0 (-x): wraps to element 1 (periodic)\n";
    std::cout << "  Face 1 (-y): wraps to element 2 (periodic)\n";
    std::cout << "  Face 2 (-z): wraps to element 4 (periodic)\n";
    std::cout << "  Face 3 (+x): connects to element 1\n";
    std::cout << "  Face 4 (+y): connects to element 2\n";
    std::cout << "  Face 5 (+z): connects to element 4\n\n";

    // Elements in 2×2×2 grid (row-major: x varies fastest)
    // Element 0: (0,0,0), Element 1: (1,0,0), Element 2: (0,1,0), Element 3: (1,1,0)
    // Element 4: (0,0,1), Element 5: (1,0,1), Element 6: (0,1,1), Element 7: (1,1,1)

    std::cout << "Checking which neighbor elements have DOFs with -1.0 ratio:\n";
    const int dofs_per_elem = (order+1) * (order+1) * (order+1);

    for (int elem = 1; elem <= 7; elem++) {
        bool has_minus_one = false;
        for (int local_dof = 0; local_dof < dofs_per_elem; local_dof++) {
            int dof = elem * dofs_per_elem + local_dof;
            Real ratio = (std::abs(v_h_legacy_symm[dof]) > 1e-14)
                         ? (v_h_generic[dof] / v_h_legacy_symm[dof]) : 0.0;
            if (std::abs(ratio + 1.0) < 0.01) {
                has_minus_one = true;
                break;
            }
        }

        const char* grid_pos = (elem == 1) ? "(1,0,0)" :
                               (elem == 2) ? "(0,1,0)" :
                               (elem == 3) ? "(1,1,0)" :
                               (elem == 4) ? "(0,0,1)" :
                               (elem == 5) ? "(1,0,1)" :
                               (elem == 6) ? "(0,1,1)" : "(1,1,1)";

        std::cout << "  Element " << elem << " " << grid_pos << ": "
                  << (has_minus_one ? "HAS -1.0 ratio" : "OK") << "\n";
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "  - Element 1 connects via face 3 (+x) = OK\n";
    std::cout << "  - Element 2 connects via face 4 (+y) = OK\n";
    std::cout << "  - Element 4 connects via face 5 (+z) = ?\n";
    std::cout << "\nIf only element 4 has -1.0, the bug affects +z faces specifically.\n";
    std::cout << "If elements 1,2,4 all have -1.0, the bug affects all positive faces.\n";

    return 0;
}
