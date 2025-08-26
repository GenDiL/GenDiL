// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;
using namespace gendil;

// Utility: compute the Euclidean norm of the difference between two GenDiL::Vector
static Real difference_norm(const Vector &a, const Vector &b)
{
    Vector tmp = a;
    Axpy(-1.0, b, tmp);
    return Sqrt(Dot(tmp, tmp));
}

// Utility: compare two scalars in relative/absolute sense
static bool almost_equal(Real x, Real y, Real tol = 1e-8)
{
    return std::fabs(x - y) <= tol * std::max(Real(1.0), std::max(std::fabs(x), std::fabs(y)));
}

// Test 1: Identity operator. GMRES should “happy-breakdown” (iters==0 or 1).
bool test_identity_operator()
{
    cout << "Test 1: Identity operator... ";
    const size_t n = 10;
    Vector b(n), x(n);
    x = 0.0;
    for(size_t i=0; i<n; i++){ b[i] = Real(i+1); }

    struct IdentityOp {
        void operator()(const Vector &v, Vector &w) const {
            w = v;
        }
    } A_id;

    const Integer max_iters = 10;
    const Integer restart   = 5;
    const Real    tol       = 1e-12;

    auto dot = [](const Vector &u, const Vector &v){ return Dot(u,v); };

    vector<Vector> V_array(restart+1, Vector(n));
    Vector         w(n);
    vector<Real>   H((restart+1)*restart),
                   cs(restart), sn(restart),
                   e1_rhs(restart+1), y(restart);

    auto [converged, iters, relres] = GMRES_no_alloc(
        A_id, b, dot, max_iters, restart, tol,
        x, V_array, w,
        H, cs, sn, e1_rhs, y
    );

    bool pass = true;
    if (!converged)                pass = false;
    if (!(iters == 0 || iters == 1)) pass = false;
    if (relres > 1e-12)            pass = false;
    if (difference_norm(x, b) > 1e-12) pass = false;

    cout << (pass ? "PASS\n" : "FAIL\n");
    return pass;
}

// Test 2: Hand‐worked 2×2 matrix A, b.  Accept iters ∈ {0,1,2}, relres ≤ 1e-8, solution error ≤ 1e-8.
bool test_small_2x2()
{
    cout << "Test 2: Hand-worked 2×2 matrix... ";
    struct A2x2 {
        void operator()(const Vector &v, Vector &w) const {
            w[0] = 2.0*v[0] + 1.0*v[1];
            w[1] = 1.0*v[0] + 3.0*v[1];
        }
    } A_mat;

    Vector b(2), x(2);
    x = 0.0;
    b[0] = 1.0; b[1] = 2.0;
    Vector x_expected(2);
    x_expected[0] = 0.2;  // 1/5
    x_expected[1] = 0.6;  // 3/5

    const Integer max_iters = 200;
    const Integer restart   = 2;
    const Real    tol       = 1e-12;

    auto dot = [](const Vector &u, const Vector &v){ return Dot(u,v); };

    vector<Vector> V_array(restart+1, Vector(2));
    Vector         w(2);
    vector<Real>   H((restart+1)*restart),
                   cs(restart), sn(restart),
                   e1_rhs(restart+1), y(restart);

    auto [converged, iters, relres] = GMRES_no_alloc(
        A_mat, b, dot, max_iters, restart, tol,
        x, V_array, w,
        H, cs, sn, e1_rhs, y
    );

    bool pass = true;
    if (!converged) pass = false;
    if (difference_norm(x, x_expected) > 1e-8)
    {
        cout << "x=(" << x[0] << ", " << x[1] << ")" << endl;
        pass = false;
    }

    cout << (pass ? "PASS\n" : "FAIL\n");
    return pass;
}

// Helper: multiply an n×n SerialRecursiveArray<Real,n,n> by Vector u → v
template <Integer n>
void Mult(const SerialRecursiveArray<Real,n,n> &A, const Vector &u, Vector &v)
{
    for(int i=0;i<(int)n;i++){
        Real sum = 0.0;
        for(int j=0;j<(int)n;j++){
            sum += A(i,j) * u[j];
        }
        v[i] = sum;
    }
}

// Test 3: Random SPD vs. CG.  Use restart=n, gm_tol=1e-10, require relres ≤ 1e-8.
bool test_random_spd_vs_cg()
{
    constexpr Integer n = 20;
    using DenseMatrix = SerialRecursiveArray<Real,n,n>;
    cout << "Test 3: Random SPD vs CG... ";
    DenseMatrix D{};
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int i=0;i<(int)n;i++){
        for(int j=0;j<(int)n;j++){
            D(i,j) = dist(rng);
        }
    }
    DenseMatrix A{};
    const Real alpha = 1e-1;
    for(int i=0;i<(int)n;i++){
        for(int j=0;j<(int)n;j++){
            Real sum = 0.0;
            for(int k=0;k<(int)n;k++){
                sum += D(k,i)*D(k,j);
            }
            A(i,j) = sum + (i==j ? alpha : 0.0);
        }
    }
    Vector b(n);
    for(int i=0;i<(int)n;i++){
        b[i] = dist(rng);
    }

    // Solve with CG
    Vector x_cg(n), tmp(n), z(n), residual(n), p(n);
    x_cg = 0.0;
    const Integer cg_max = 1000;
    const Real cg_tol = 1e-12;
    auto dot = [&](const Vector &u, const Vector &v){ return Dot(u,v); };

    struct DenseOp {
        const DenseMatrix &M;
        DenseOp(const DenseMatrix &_M) : M(_M) { }
        void operator()(const Vector &v, Vector &w) const {
            Mult(M, v, w);
        }
    } A_op(A);

    auto [cg_conv, cg_iters, cg_relres] = ConjugateGradient(
        A_op, b, dot, cg_max, cg_tol,
        x_cg, tmp, z, residual, p
    );
    if (!cg_conv) {
        cout << "FAIL (CG did not converge)\n";
        return false;
    }

    // Solve with GMRES: restart = n, gm_tol = 1e-10
    Vector x_gmres(n);
    x_gmres = 0.0;
    const Integer gm_max = 10000;
    const Integer restart = n;    // no‐restart
    const Real    gm_tol = cg_tol;

    vector<Vector> V_array(restart+1, Vector(n));
    Vector         w2(n);
    vector<Real>   H((restart+1)*restart),
                   cs2(restart), sn2(restart),
                   e1_rhs2(restart+1), y2(restart);

    auto [gm_conv, gm_iters, gm_relres] = GMRES_no_alloc(
        A_op, b, dot, gm_max, restart, gm_tol,
        x_gmres, V_array, w2,
        H, cs2, sn2, e1_rhs2, y2
    );
    if (!gm_conv) {
        cout << "FAIL (GMRES did not converge)\n";
        return false;
    }

    bool pass = true;
    if (difference_norm(x_cg, x_gmres) > 1e-6) {
        pass = false;
    }
    if (gm_relres > 1e-8) {
        pass = false;
    }
    cout << (pass ? "PASS\n" : "FAIL\n");
    return pass;
}

// Test 4: Poisson (1D) vs. CG (small mesh).  restart=ndofs, gm_tol=1e-10, relres ≤ 1e-8, L²‐error within 1e-8.
bool test_poisson_vs_cg()
{
    cout << "Test 4: Poisson (1D) GMRES vs CG... ";
    const Integer n = 4;
    const Integer order = 1, num_quad_1d = order + 2;

    const Real h = 1.0 / n;
    Cartesian1DMesh mesh(h, n);
    FiniteElementOrders<order> orders;
    auto fe       = MakeLegendreFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, fe);
    IntegrationRuleNumPoints<num_quad_1d> nq;
    auto int_rules = MakeIntegrationRule(nq);

    constexpr Integer Dim = 1;
    auto coeff = [] GENDIL_HOST_DEVICE (const array<Real,Dim> &X){ return 1.0; };
    const double sigma = -1.0;
    const double kappa = (order+1)*(order+1);
    using KernelPolicy = SerialKernelConfiguration;
    auto poisson_op = MakeDiffusionOperator<KernelPolicy>(
        fe_space, int_rules, coeff, sigma, kappa
    );

    auto u_exact = [] GENDIL_HOST_DEVICE (array<Real,Dim> const &X)
    {
        return std::sin(M_PI * X[0]);
    };
    auto rhs_lambda = [] GENDIL_HOST_DEVICE(array<Real,Dim> const &X)
    {
        return (Real)( M_PI*M_PI * std::sin(M_PI * X[0]) );
    };
    const auto b = MakeLinearForm(fe_space, int_rules, rhs_lambda);

    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    Vector x0(ndofs), tmp(ndofs), z(ndofs), residual(ndofs), p(ndofs);
    x0 = 0.0;
    const Integer cg_max = 2000;
    const Real cg_tol = 1e-10;
    auto dot = [&](const Vector &u, const Vector &v){ return Dot(u,v); };
    auto [cg_conv, cg_iters, cg_relres] = ConjugateGradient(
        poisson_op, b, dot, cg_max, cg_tol,
        x0, tmp, z, residual, p
    );
    if (!cg_conv) {
        cout << "FAIL (CG did not converge on Poisson)\n";
        return false;
    }
    Real err_cg = L2Error<KernelPolicy>(fe_space, int_rules, u_exact, x0);

    // GMRES: restart=ndofs, gm_tol=1e-10
    Vector x1(ndofs);
    x1 = 0.0;
    const Integer gm_max = 2000;
    const Integer restart = ndofs;
    const Real    gm_tol = 1e-10;

    vector<Vector> V_array(restart+1, Vector(ndofs));
    Vector         w(ndofs);
    vector<Real>   H((restart+1)*restart),
                   cs(restart), sn(restart),
                   e1_rhs(restart+1), y(restart);

    auto [gm_conv, gm_iters, gm_relres] = GMRES_no_alloc(
        poisson_op, b, dot, gm_max, restart, gm_tol,
        x1, V_array, w,
        H, cs, sn, e1_rhs, y
    );
    if (!gm_conv) {
        cout << "FAIL (GMRES did not converge on Poisson)\n";
        return false;
    }
    Real err_gm = L2Error<KernelPolicy>(fe_space, int_rules, u_exact, x1);

    bool pass = true;
    if (difference_norm(x0, x1) > 1e-6) pass = false;
    if (gm_relres > 1e-8)               pass = false;
    if (!almost_equal(err_cg, err_gm, 1e-8)) pass = false;

    cout << (pass ? "PASS\n" : "FAIL\n");
    return pass;
}

// Test 5: Happy breakdown (3×3).  Expect iters=0, x = e0 exactly.
bool test_happy_breakdown()
{
    cout << "Test 5: Happy breakdown test... ";
    struct A3x3 {
        void operator()(const Vector &v, Vector &w) const {
            w[0] = 5.0 * v[0];
            w[1] = 2.0 * v[1] + 1.0 * v[2];
            w[2] = 1.0 * v[1] + 3.0 * v[2];
        }
    } Aop;

    Vector v0(3), b(3);
    v0 = 0.0; v0[0] = 1.0;  // e0
    Aop(v0, b);            // b = [5, 0, 0]

    Vector x(3);
    x = 0.0;
    const Integer max_iters = 5;
    const Integer restart   = 3;
    const Real    tol       = 1e-12;
    auto dot = [](const Vector &u, const Vector &v){ return Dot(u,v); };

    vector<Vector> V_array(restart+1, Vector(3));
    Vector         w2(3);
    vector<Real>   H((restart+1)*restart),
                   cs2(restart), sn2(restart),
                   e1_rhs2(restart+1), y2(restart);

    auto [converged, iters, relres] = GMRES_no_alloc(
        Aop, b, dot, max_iters, restart, tol,
        x, V_array, w2,
        H, cs2, sn2, e1_rhs2, y2
    );

    bool pass = true;
    if (!converged)      pass = false;
    if (iters != 0)      pass = false;
    if (relres > 1e-12)  pass = false;
    if (std::fabs(x[0] - 1.0) > 1e-12) pass = false;
    if (std::fabs(x[1]) > 1e-12)       pass = false;
    if (std::fabs(x[2]) > 1e-12)       pass = false;

    cout << (pass ? "PASS\n" : "FAIL\n");
    return pass;
}


int main()
{
    bool ok = true;
    ok &= test_identity_operator();
    ok &= test_small_2x2();
    ok &= test_random_spd_vs_cg();
    ok &= test_poisson_vs_cg();
    ok &= test_happy_breakdown();

    if (ok) {
        cout << "\nAll GMRES unit tests PASSED.\n";
        return 0;
    } else {
        cout << "\nSOME GMRES unit tests FAILED.\n";
        return 1;
    }
}
