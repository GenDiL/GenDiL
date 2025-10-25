// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <chrono>
#include <cmath>

using namespace std;
using namespace gendil;

// exact solution for prod sin(pi x_i)
template<int Dim>
struct Manufactured {
    static Real u_exact(const array<Real,Dim>& X) {
        Real prod = Real(1.0);
        for(int i=0;i<Dim;i++)
            prod *= Real(sin(M_PI * X[i]));
        return prod;
    }
    // not needed for mass problem
};

template < Integer order, Integer num_quad_1d = order + 2 >
void test_mass_1D( const Integer n )
{
    // 1) build a 1D mesh
    const Real h = Real(1.0/n);
    Cartesian1DMesh mesh(h,n);

    // 2) finite element space
    FiniteElementOrders<order> orders;
    auto fe = MakeLobattoFiniteElement(orders);
    auto fe_space = MakeFiniteElementSpace(mesh, fe);

    // 3) integration rule
    IntegrationRuleNumPoints<num_quad_1d> nq;
    auto int_rules = MakeIntegrationRule(nq);

    constexpr Integer Dim = GetDim( fe_space );
#if defined(GENDIL_USE_DEVICE)
    using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
    constexpr size_t NumShared = Dim;
    using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout,NumShared>;
#else
    using KernelPolicy = SerialKernelConfiguration;
#endif

    // 4) Create mass operator
    auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
    {
        return Real(1.0);
    };
    auto mass_op = MakeMassFiniteElementOperator<KernelPolicy>(fe_space, int_rules, sigma);

    // 5) Build RHS vector b_i = ∫ φ_i(x) u_exact(x) dx
    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    auto rhs_lambda = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) {
        return Manufactured<Dim>::u_exact(X);
    };
    const Vector b = MakeLinearForm(fe_space, int_rules, rhs_lambda);

    // 1) Solve M u = b via CG
    Vector u_h(ndofs);
    u_h = Real(0.0);
    const Integer max_iters = 2000;
    const Real tol = Real(1e-12);
    auto dot = []( const Vector & U, const Vector & V ) { return Dot(U,V); };
    // scratch vectors
    Vector tmp(ndofs), z(ndofs), residual(ndofs), p(ndofs);
    ConjugateGradient(mass_op, b, dot, max_iters, tol, u_h, tmp, z, residual, p);

    // 7) Compute L2 error: ∥u_h − u_exact∥_{L2}
    constexpr Integer num_quad_error = num_quad_1d+2;
    IntegrationRuleNumPoints<num_quad_error> nq_error;
    auto error_int_rules = MakeIntegrationRule(nq_error);
    auto err_L2 = L2Error<KernelPolicy>(
        fe_space, error_int_rules,
        Manufactured<Dim>::u_exact,
        u_h
    );

    // 8) Print for TikZ
    cout << "       (" << ndofs << ", " << double(err_L2) << ")\n";
}

template < Integer order, Integer num_quad_1d = order + 2 >
void test_range()
{
    const Integer max_dofs = 1e7;
    Integer n = 1;
    cout << "       \\addplot coordinates {\n";
    while ( true )
    {
        Integer ndofs = Pow<1>(order+1) * Pow<1>(n);
        if ( ndofs > max_dofs ) break;

        test_mass_1D<order,num_quad_1d>(n);
        n *= 2;
    }
    cout << "       };\n";
    cout << "       \\addlegendentry{p="<<order
         <<", q="<< num_quad_1d <<"}\n\n";
}

int main()
{
    constexpr Integer max_p = 4;

    cout << "\n1D Mass-Matrix Convergence Study\n"
         << "  Manufactured: ∏ sin(π x_i)\n\n"
         << "  \\begin{tikzpicture}[scale=0.9]\n"
         << "    \\begin{axis}[\n"
         << "       title={L2-Error vs DoFs (1D Mass)},\n"
         << "       xlabel={Number of DoFs},\n"
         << "       ylabel={L2 Error},\n"
         << "       xmode=log, ymode=log,\n"
         << "       grid=major,\n"
         << "       legend pos=outer north east\n"
         << "    ]\n";

    ConstexprLoop<max_p>([](auto p){
        constexpr Integer num_quad = p + 2;
        test_range<p,num_quad>();
    });

    cout << "    \\end{axis}\n"
         << "  \\end{tikzpicture}\n\n";
    return 0;
}
