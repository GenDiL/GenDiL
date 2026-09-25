// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>
#include <chrono>
#include <cmath>

using namespace std;
using namespace gendil;

// exact solution and RHS for prod sin(pi x_i)
template<int Dim>
struct Manufactured {
    static Real u_exact(const array<Real,Dim>& X) {
        Real prod = 1.0;
        for(int i=0;i<Dim;i++)
            prod *= sin(M_PI * X[i]);
        return prod;
    }
    static void grad_exact(const array<Real,Dim>& X, array<Real,Dim>& grad) {
        // ∂_i u = π cos(π x_i) ∏_{j≠i} sin(π x_j)
        for(int i=0;i<Dim;i++){
            Real term = M_PI * cos(M_PI * X[i]);
            for(int j=0;j<Dim;j++) if(j!=i)
                term *= sin(M_PI * X[j]);
            grad[i] = term;
        }
    }
    static Real rhs(const array<Real,Dim>& X) {
        // f = d π^2 ∏ sin(π x_i)
        Real prod = 1.0;
        for(int i=0;i<Dim;i++)
            prod *= sin(M_PI * X[i]);
        return Dim * M_PI * M_PI * prod;
    }
};

template < Integer order, Integer num_quad_1d = order + 2 >
void test_poisson_6D( const Integer n )
{
    // 1) build a 6D mesh as cartesian product of two 3D cubes
    const Real h = 1.0/n;
    Cartesian3DMesh mesh1(h,n,n,n), mesh2(h,n,n,n);
    auto mesh = MakeCartesianProductMesh( mesh1, mesh2 );

    // 2) finite element space
    FiniteElementOrders<order,order,order,order,order,order> orders;
    auto fe = MakeLegendreFiniteElement( orders );
    auto fe_space = MakeFiniteElementSpace( mesh, fe );

    // 3) integration rule
    IntegrationRuleNumPoints<
      num_quad_1d,num_quad_1d,num_quad_1d,
      num_quad_1d,num_quad_1d,num_quad_1d
    > nq;
    auto int_rules = MakeIntegrationRule( nq );

    constexpr Integer Dim = GetDim( fe_space );
    // 4) Create diffusion operator (homogeneous Dirichlet)
#if defined(GENDIL_USE_DEVICE)
    using ThreadLayout = ThreadBlockLayout<
      num_quad_1d,num_quad_1d,num_quad_1d,
      num_quad_1d,num_quad_1d,num_quad_1d
    >;
    constexpr size_t NumShared = Dim;
    using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout,NumShared>;
#else
    using KernelPolicy = SerialKernelConfiguration;
#endif

    auto coeff = [] GENDIL_HOST_DEVICE ( const array<Real,Dim>& X )
    {
        return 1.0;
    };

    const double sigma = -1.0;
    const double kappa = (order+1)*(order+1);
    auto poisson_op = MakeDiffusionOperator<KernelPolicy>( fe_space, int_rules, coeff, sigma, kappa );

    // 5) Build RHS vector b_i = ∫Ω φ_i f
    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    auto rhs_lambda = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ){
        return Manufactured<Dim>::rhs(X);
    };
    Vector b = MakeLinearForm( fe_space, int_rules, rhs_lambda );

    // 6) Solve A x = b via CG
    Vector x( ndofs );
    x = 0.0;
    const Integer max_iters = 2000;
    const Real tol = 1e-10;
    auto dot = []( const Vector & u, const Vector & v )
    {
        return Dot( u,v );
    };
    // scratch memory inputs
    Vector tmp(ndofs), z(ndofs), residual(ndofs), p(ndofs);

    ConjugateGradient( poisson_op, b, dot, max_iters, tol, x, tmp, z, residual, p );

    // 7) Compute L2 error: ∥u_h−u_exact∥_{L2}
    auto err_L2 = L2Error<KernelPolicy>( fe_space, int_rules, Manufactured<6>::u_exact, x );

    // 8) Print for TikZ
    cout << "       (" << ndofs << ", " << err_L2 << ")\n";
}

template < Integer order, Integer num_quad_1d = order + 2 >
void test_range()
{
    const Integer max_dofs = 1e7; 
    constexpr Integer dim = 6;
    Integer n = 1;
    cout << "       \\addplot coordinates {\n";
    while( true )
    {
        // stop if we exceed max dofs
        Integer ndofs = Pow<dim>(order+1) * Pow<dim>(n);
        if(ndofs > max_dofs) break;

        test_poisson_6D<order,num_quad_1d>( n );
        n *= 2;
    }
    cout << "       };\n";
    cout << "       \\addlegendentry{p="<<order<<", q="<< num_quad_1d <<"}\n\n";
}

int main()
{
    constexpr Integer max_p = 2;

    cout << "\n6D Poisson Convergence Study\n"
         << "  Manufactured: ∏ sin(π x_i)\n\n"
         << "  \\begin{tikzpicture}[scale=0.9]\n"
         << "    \\begin{axis}[\n"
         << "       title={L2‐Error vs DoFs (6D Poisson)},\n"
         << "       xlabel={Number of DoFs},\n"
         << "       ylabel={L2 Error},\n"
         << "       xmode=log, ymode=log,\n"
         << "       grid=major,\n"
         << "       legend pos=outer north east\n"
         << "    ]\n";

    ConstexprLoop<max_p>( []( auto p )
    {
        constexpr Integer num_quad = p + 2;
        test_range<p,num_quad>();
    } );

    cout << "    \\end{axis}\n"
         << "  \\end{tikzpicture}\n\n";
    return 0;
}
