// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// 1D Advection DG convergence.
//
// Toggle PDE form (advective vs conservative) here:
static constexpr bool ADVECTIVE_FORM = true; // true: f=+beta·grad u ; false: f=-beta·grad u
// -----------------------------------------------------------------------------

#include <gendil/gendil.hpp>
#include <cmath>
#include <iostream>
using namespace std;
using namespace gendil;

// ===== Manufactured solution u(x) = sin(pi x) in 1D (kept generic in Dim) =====
template<int Dim>
struct Manufactured {
    static Real u_exact(const array<Real,Dim>& X) {
        Real prod = 1.0;
        for (int i=0;i<Dim;++i) prod *= sin(M_PI * X[i]);
        return prod;
    }
    static void grad_exact(const array<Real,Dim>& X, array<Real,Dim>& grad) {
        for (int i=0;i<Dim;++i) {
            Real g = M_PI * cos(M_PI * X[i]);
            for (int j=0;j<Dim;++j) if (j!=i) g *= sin(M_PI * X[j]);
            grad[i] = g;
        }
    }
};

// ===== Single (n) solve & error for order p =====
template <
    Integer order,
    Integer num_quad = (order + 2),
    Integer num_quad_rhs = (order + 4),
    Integer num_quad_err = (order + 4) >
void test_advection_1D(const Integer n)
{
    constexpr int Dim = 1;

    // 1) Mesh [0,1] with n elements
    const Real h = 1.0 / n;
    Cartesian1DMesh mesh(h, n);

    // 2) FE space (Legendre p)
    FiniteElementOrders<order> orders;
    auto fe       = MakeLegendreFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, fe);

    // 3) Quadrature (slightly higher for RHS & error)
    IntegrationRuleNumPoints<num_quad> nq;  auto int_rules = MakeIntegrationRule(nq);
    IntegrationRuleNumPoints<num_quad_rhs> nq_rhs;  auto int_rules_rhs = MakeIntegrationRule(nq_rhs);
    IntegrationRuleNumPoints<num_quad_err> nq_err;  auto int_rules_err = MakeIntegrationRule(nq_err);

    // 4) Advection operator (your pattern)
    const Real a = 1.2345; // constant to the right
    auto adv  = [=] GENDIL_HOST_DEVICE (const std::array<Real,Dim>& /*X*/, Real (&v)[Dim]) { v[0] = a; };
    auto zero = []  GENDIL_HOST_DEVICE (const std::array<Real,Dim>& /*X*/) { return 0.0; };
    auto one = []  GENDIL_HOST_DEVICE (const std::array<Real,Dim>& /*X*/) { return 1.0; };

#if defined(GENDIL_USE_DEVICE)
    using ThreadLayout = ThreadBlockLayout<num_quad_rhs>;
    using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, 1>;
#else
    using KernelPolicy = SerialKernelConfiguration;
#endif

    auto A = MakeAdvectionOperator<KernelPolicy>(fe_space, int_rules, adv, zero);
    auto M = MakeMassFiniteElementOperator<KernelPolicy>(fe_space, int_rules, one);
    auto Minv = MakeMassInverseFiniteElementOperator< KernelPolicy >( fe_space, int_rules, one );
    auto MA = [&](const Vector& x, Vector& y) {  A(x, y); Minv(y, y); };

    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    auto dot = [](const Vector& u, const Vector& v){ return Dot(u, v); };
    Vector tmp_dot(ndofs);
    auto dot_M = [&](const Vector& u, const Vector& v) {
        M(u, tmp_dot);
        return Dot(tmp_dot, v);
    };

    // 5) Continuous MMS RHS: f = SIGN * beta · grad(u_exact)
    const Real SIGN = ADVECTIVE_FORM ? +1.0 : -1.0;
    auto rhs_lambda = [=] GENDIL_HOST_DEVICE (const array<Real,Dim>& X) {
        array<Real,Dim> g{}; Manufactured<Dim>::grad_exact(X, g);
        return SIGN * a * g[0];
    };
    Vector b = MakeLinearForm(fe_space, int_rules_rhs, rhs_lambda);
    Vector Mb(ndofs);
    Minv(b, Mb);

    // 6) Solve A x = b with GMRES (robust for transport)
    Vector x(ndofs); x = 0.0;
    const Integer max_iters = std::max<Integer>(20000, 3*n);
    const Integer restart   = 200;
    const Real    tol       = 1e-10;

    vector<Vector> V_array(restart + 1);
    for (size_t i = 0; i < restart + 1; i++)
    {
        V_array[i] = Vector(ndofs);
        V_array[i] = 0.0;
    }
    Vector         w(ndofs); w = 0.0;
    vector<Real>   H((restart+1)*restart),
                   cs2(restart), sn2(restart),
                   e1_rhs2(restart+1), y2(restart);
    // auto ret = GMRES_no_alloc(
    //     A, b, dot, max_iters, restart, tol,
    //     x, V_array, w,
    //     H, cs2, sn2, e1_rhs2, y2
    // );
    auto ret = GMRES_no_alloc(
        MA, Mb, dot_M, max_iters, restart, tol,
        x, V_array, w,
        H, cs2, sn2, e1_rhs2, y2
    );
    bool ok = ret.success;
    Integer iters = ret.iterations;
    Real relres = ret.relative_error;

    // Residual check
    Vector Ax(ndofs), res(ndofs);
    A(x, Ax);
    res = b; Axpy(-1.0, Ax, res);
    const Real nres = std::sqrt(dot(res,res));
    const Real nb   = std::sqrt(dot(b,b));
    if (!ok || !std::isfinite(relres) || !std::isfinite(nres) || nres > 1e-6*(nb+1e-32)) {
        std::cerr << "WARNING: GMRES ok="<<ok
                  << " iters="<<iters
                  << " relres="<<relres
                  << " ||b-Ax||="<<nres
                  << "  ["
                  << (ADVECTIVE_FORM ? "advective f=+beta·grad u" : "conservative f=-beta·grad u")
                  << "]\n";
    }

    // 7) L2 error vs continuous exact u
    auto err_L2 = L2Error<KernelPolicy>(fe_space, int_rules_err, Manufactured<Dim>::u_exact, x);

    // 8) TikZ-friendly point
    cout << "       (" << ndofs << ", " << err_L2 << ")\n";
}

// ===== Sweep resolution for a given p =====
template < Integer order,
           Integer num_quad_rhs = (order + 4),
           Integer num_quad_err = (order + 4) >
void test_range()
{
    const Integer max_dofs = 1e5; 
    constexpr Integer dim = 1;
    Integer n = 1;

    cout << "       \\addplot coordinates {\n";
    while (true) {
        Integer ndofs_est = Pow<dim>(order + 1) * Pow<dim>(n); // 1D: (p+1)*n
        if (ndofs_est > max_dofs) break;

        test_advection_1D<order, num_quad_rhs, num_quad_err>(n);
        n *= 2;
    }
    cout << "       };\n";
    cout << "       \\addlegendentry{p=" << order
         << ", q_rhs=" << (order+4)
         << ", q_err=" << (order+4)
         << ", GMRES, "
         << (ADVECTIVE_FORM ? "adv" : "cons")
         << "}\n\n";
}

int main()
{
    constexpr Integer max_p = 4;

    cout << "\n1D Advection Convergence Study (Upwind DG, GMRES-only, continuous MMS)\n"
         << "  Manufactured: ∏ sin(π x_i), β = (1.2345)\n"
         << "  Form: " << (ADVECTIVE_FORM ? "advective f=+β·∇u" : "conservative f=-β·∇u") << "\n\n"
         << "  \\begin{tikzpicture}[scale=0.9]\n"
         << "    \\begin{axis}[\n"
         << "       title={L2-Error vs DoFs (1D Advection)},\n"
         << "       xlabel={Number of DoFs},\n"
         << "       ylabel={L2 Error},\n"
         << "       xmode=log, ymode=log,\n"
         << "       grid=major,\n"
         << "       legend pos=outer north east\n"
         << "    ]\n";

    ConstexprLoop<max_p>([](auto p){
        test_range<p>();
    });

    cout << "    \\end{axis}\n"
         << "  \\end{tikzpicture}\n\n";
    return 0;
}
