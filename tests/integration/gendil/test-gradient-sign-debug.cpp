// Debug test to check if gradient basis function signs differ between faces
#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

int main() {
    // 1×1×2 mesh: one interior face in z-direction
    constexpr Integer Dim = 3;
    const Integer nx = 1, ny = 1, nz = 2;
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
    u_h[0] = 1.0;  // DOF 0 of element 0

    Vector v_h_generic(num_dofs);
    v_h_generic = 0.0;

    auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real { return 1.0; };
    const Real sigma = 1.0;
    const Real kappa = 0.0;

    using KernelPolicy = SerialKernelConfiguration;

    TrialSpace<"displacement"> u;
    TestSpace<"displacement"> v;
    InteriorFacets<"mesh1"> interior_facets;
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeDomain<"mesh1">(mesh));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    std::cout << "1×1×2 mesh symmetry term with u[0]=1:\n";
    std::cout << "Element 0 face 5 (+z): normal = [0, 0, +1]\n";
    std::cout << "Element 1 face 2 (-z): normal = [0, 0, -1]\n\n";

    std::cout << "Element 0 DOFs (0-7):\n";
    for (int i = 0; i < 8; i++) {
        printf("  v[%d] = %12.6e\n", i, v_h_generic[i]);
    }

    std::cout << "\nElement 1 DOFs (8-15):\n";
    for (int i = 8; i < 16; i++) {
        printf("  v[%d] = %12.6e\n", i, v_h_generic[i]);
    }

    std::cout << "\nIf element 1 DOFs have opposite sign from expected, there's a sign flip bug.\n";

    return 0;
}
