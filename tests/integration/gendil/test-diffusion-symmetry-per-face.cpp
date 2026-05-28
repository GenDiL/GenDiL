// Test symmetry term on each face individually
// This will tell us if the bug is direction-specific (x, y, or z faces)

#include <gendil/gendil.hpp>
#include <iostream>

using namespace gendil;

// Test with mesh that isolates a single face direction
template<int FaceDir>  // 0=x, 1=y, 2=z
int TestSingleFaceDirection() {
    constexpr Integer Dim = 3;

    // Create mesh with only one face in the specified direction
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
    auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

    auto symmetry_wf = integrate(interior_facets, sigma * jump(u) * average(mu * dot(grad(v), Normal{})));
    auto wf_context = MakeWeakFormContext(MakeTrialField<"displacement">(fe_space), MakeDomain<"mesh1">(mesh));
    auto generic_operator = MakeGenericOperator<KernelPolicy>(symmetry_wf, wf_context, int_rules);
    generic_operator(u_h, v_h_generic);

    Vector v_h_legacy_symm(num_dofs);
    v_h_legacy_symm = 0.0;
    for (Integer i = 0; i < num_dofs; i++) {
        v_h_legacy_symm[i] = v_h_legacy[i] - v_h_no_symm[i];
    }

    const char* dir_name = (FaceDir == 0) ? "X" : (FaceDir == 1) ? "Y" : "Z";
    std::cout << "\n=== Testing " << dir_name << "-direction face (mesh: "
              << nx << "×" << ny << "×" << nz << ") ===\n";

    bool has_exact_minus_one = false;
    for (Integer i = 0; i < num_dofs; i++) {
        if (std::abs(v_h_legacy_symm[i]) > 1e-14 || std::abs(v_h_generic[i]) > 1e-14) {
            Real ratio = (std::abs(v_h_legacy_symm[i]) > 1e-14) ? (v_h_generic[i] / v_h_legacy_symm[i]) : 0.0;
            printf("DOF %3d | Legacy: %12.6e | Generic: %12.6e | Ratio: %7.3f%s\n",
                   (int)i, v_h_legacy_symm[i], v_h_generic[i], ratio,
                   (std::abs(ratio + 1.0) < 0.01) ? " <-- EXACT -1.0" : "");
            if (std::abs(ratio + 1.0) < 0.01) {  // Close to -1.0
                has_exact_minus_one = true;
            }
        }
    }

    std::cout << "Result: " << (has_exact_minus_one ? "HAS -1.0 ratio (BUG)" : "No -1.0 ratio (OK)") << "\n";
    return has_exact_minus_one ? 1 : 0;
}

int main() {
    std::cout << "Testing each face direction independently\n";
    std::cout << "This will show if the bug is specific to X, Y, or Z faces\n";

    int failures = 0;
    failures += TestSingleFaceDirection<0>();  // X-faces
    failures += TestSingleFaceDirection<1>();  // Y-faces
    failures += TestSingleFaceDirection<2>();  // Z-faces

    if (failures > 0) {
        std::cout << "\n" << failures << " direction(s) show the -1.0 bug.\n";
    } else {
        std::cout << "\nNo -1.0 ratios found in any direction.\n";
    }

    return failures;
}
