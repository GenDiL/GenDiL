// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

template< Integer order, Integer num_quad_1d = order + 2 >
void compare_solvers(const Integer n)
{
    constexpr Integer Dim = 1;

    const Real h = 1.0 / n;
    Cartesian1DMesh mesh(h, n);

    FiniteElementOrders<order> orders;
    auto fe       = MakeLegendreFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, fe);

    IntegrationRuleNumPoints<num_quad_1d> nq;
    auto int_rules = MakeIntegrationRule(nq);

#if defined(GENDIL_USE_DEVICE)
    using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
    using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, Dim>;
#else
    using KernelPolicy = SerialKernelConfiguration;
#endif

    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    auto dot = [](const Vector &u, const Vector &v) { return Dot(u, v); };

    auto rhs_mass_lambda = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X) {
        return sin(M_PI * X[0]);
    };

    auto rhs_diffusion_lambda = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X) {
        return M_PI * M_PI * sin(M_PI * X[0]);
    };

    Vector b_mass = MakeLinearForm(fe_space, int_rules, rhs_mass_lambda);
    Vector b_diffusion = MakeLinearForm(fe_space, int_rules, rhs_diffusion_lambda);

    auto u_exact = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X) {
        return sin(M_PI * X[0]);
    };

    auto sigma_mass = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X) { return 1.0; };
    auto mass_op = MakeMassFiniteElementOperator<KernelPolicy>(fe_space, int_rules, sigma_mass);

    const Real sigma_diff = -1.0;
    const Real kappa = (order + 1) * (order + 1);
    auto velocity = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X) { return 1.0; };
    auto diff_op = MakeDiffusionOperator<KernelPolicy>(fe_space, int_rules, velocity, sigma_diff, kappa);

    const Integer max_iters = 500;
    const Real tol = 1e-10;
    const Integer restart = 50;

    auto run_solver = [&](const char * name, auto && op, const Vector & b_rhs)
    {
        std::cout << "\n[" << name << "]\n";

        // Conjugate Gradient
        {
            Vector x(ndofs), tmp(ndofs), z(ndofs), r(ndofs), pvec(ndofs);
            x = 0.0; tmp = 0.0; z = 0.0; r = 0.0; pvec = 0.0;
            auto [ok, iters, relres] = ConjugateGradient(
                op, b_rhs, dot, max_iters, tol,
                x, tmp, z, r, pvec
            );
            auto err = L2Error<KernelPolicy>(fe_space, int_rules, u_exact, x);
            std::cout << "  CG       : err = " << err << ", iter = " << iters << ", res = " << relres << "\n";
        }

        // GMRES
        {
            Vector x(ndofs); x = 0.0;
            std::vector<Vector> V_array(restart + 1);
            for (size_t i = 0; i < restart + 1; i++)
            {
                V_array[i] = Vector(ndofs);
                V_array[i] = 0.0;
            }
            Vector w(ndofs); w = 0.0;
            std::vector<Real> H((restart+1)*restart, 0.0);
            std::vector<Real> cs(restart, 0.0), sn(restart, 0.0);
            std::vector<Real> e1_rhs(restart+1, 0.0), y(restart, 0.0);

            auto [ok, iters, relres] = GMRES_no_alloc(
                op, b_rhs, dot, max_iters, restart, tol,
                x, V_array, w, H, cs, sn, e1_rhs, y
            );
            auto err = L2Error<KernelPolicy>(fe_space, int_rules, u_exact, x);
            std::cout << "  GMRES    : err = " << err << ", iter = " << iters << ", res = " << relres << "\n";
        }

        // BiCGSTAB
        {
            Vector x(ndofs); x = 0.0;
            Vector r(ndofs), r_hat(ndofs), p(ndofs), v(ndofs), s(ndofs), t(ndofs);
            auto [ok, iters, relres] = BiCGSTAB_no_alloc(
                op, b_rhs, dot, max_iters, tol,
                x, r, r_hat, p, v, s, t
            );
            auto err = L2Error<KernelPolicy>(fe_space, int_rules, u_exact, x);
            std::cout << "  BiCGSTAB : err = " << err << ", iter = " << iters << ", res = " << relres << "\n";
        }
    };

    run_solver("Mass Operator", mass_op, b_mass);
    run_solver("Diffusion Operator", diff_op, b_diffusion);
}

int main(int argc, char ** argv)
{
    constexpr Integer order = 3;
    Integer n_elems = 32; // Number of elements in the mesh
    compare_solvers<order>(n_elems);

    return 0;
}
