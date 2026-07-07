// Detailed analysis of symmetry term on a single face
// Print all intermediate values to understand the sign discrepancy

#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

int main() {
    // Simplest case: 2×1×1 mesh (only one interior face)
    constexpr Integer Dim = 3;
    const Integer nx = 2, ny = 1, nz = 1;
    const Real h = 1.0 / nx;
    CartesianMesh<Dim> mesh({nx, ny, nz}, {h, h, h}, {0.0, 0.0, 0.0}, true);

    constexpr Integer order = 1;
    FiniteElementOrders<order, order, order> orders;
    auto finite_element = MakeLobattoFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

    constexpr Integer num_quad_1d = order + 3;
    IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
    auto int_rules = MakeIntegrationRule(num_quads);

    const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
    std::cout << "Mesh: " << nx << "×" << ny << "×" << nz << " = " << (nx*ny*nz) << " elements\n";
    std::cout << "DOFs per element: " << ((order+1)*(order+1)*(order+1)) << "\n";
    std::cout << "Total DOFs: " << num_dofs << "\n\n";

    // Unit vector: u[0] = 1, all others = 0
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

    // Legacy full
    auto diffusion_operator = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma, kappa);
    diffusion_operator(u_h, v_h_legacy);

    // Legacy without symmetry
    auto diffusion_no_symm = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, 0.0, kappa);
    diffusion_no_symm(u_h, v_h_no_symm);

    // Generic symmetry only
    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeIntegrationDomain<"mesh1">(fe_space));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    // Compute legacy symmetry = full - (volume+consistency)
    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    std::cout << "Input: u[0]=1.0, all others=0\n\n";
    std::cout << "DOF | Legacy_Symm | Generic_Symm | Ratio\n";
    std::cout << "----+-------------+--------------+-------\n";
    for (Integer i = 0; i < num_dofs; i++) {
        if (std::abs(v_h_legacy_symm[i]) > 1e-14 || std::abs(v_h_generic[i]) > 1e-14) {
            Real ratio = (std::abs(v_h_legacy_symm[i]) > 1e-14) ? (v_h_generic[i] / v_h_legacy_symm[i]) : 0.0;
            printf("%3d | %12.6e | %12.6e | %6.3f\n", (int)i, v_h_legacy_symm[i], v_h_generic[i], ratio);
        }
    }

    // Analyze which element each DOF belongs to
    std::cout << "\nElement ownership (order=" << order << ", so " << ((order+1)*(order+1)*(order+1)) << " DOFs per element):\n";
    std::cout << "Element 0: DOFs 0-7\n";
    std::cout << "Element 1: DOFs 8-15\n";
    std::cout << "\nIf element 1's DOFs have ratio -1.0, the sign is flipped for the neighbor element.\n";

    return 0;
}
