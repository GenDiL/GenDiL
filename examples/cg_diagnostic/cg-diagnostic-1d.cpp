// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <iomanip>
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
void run_cg_diagnostic_1d( const Integer n )
{
    // 1) build a 1D mesh
    const Real h = 1.0/n;
    Cartesian1DMesh mesh(h,n);

    // 2) finite element space
    FiniteElementOrders<order> orders;
    auto fe = MakeLegendreFiniteElement( orders );
    auto fe_space = MakeFiniteElementSpace( mesh, fe );

    // 3) integration rule
    IntegrationRuleNumPoints<num_quad_1d> nq;
    auto int_rules = MakeIntegrationRule( nq );

    constexpr int Dim = 1;
    // 4) Create diffusion operator (homogeneous Dirichlet)
#if defined(GENDIL_USE_DEVICE)
    using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
    constexpr size_t NumShared = Dim;
    using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout,NumShared>;
#else
    using KernelPolicy = SerialKernelConfiguration;
#endif

    auto coeff = [] GENDIL_HOST_DEVICE ( const array<Real,Dim>& X )
    {
        return 1.0;
    };

    const Real sigma = Real(-1.0);
    const Real kappa = Real((order+1)*(order+1));
    auto poisson_op = MakeDiffusionOperator<KernelPolicy>( fe_space, int_rules, coeff, sigma, kappa );

    // 5) Build RHS vector b_i = ∫Ω φ_i f
    const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
    auto rhs_lambda = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ){
        return Manufactured<Dim>::rhs(X);
    };
    const Vector b = MakeLinearForm( fe_space, int_rules, rhs_lambda );

    // 6) Solve A x = b via diagnostic CG
    Vector x( ndofs );
    x = 0.0;
    const Integer max_iters = 800;
    const Real tol = 1e-10;
    auto dot = []( const Vector & u, const Vector & v )
    {
        return Dot( u,v );
    };
    // scratch memory inputs
    Vector tmp(ndofs), z(ndofs), residual(ndofs), p(ndofs);
    tmp = 0.0;
    z = 0.0;
    residual = 0.0;
    p = 0.0;

    // Configure diagnostics
    CGDiagnosticOptions opts;
    opts.recompute_true_residual_each_iter = true;
    opts.store_basis = false;  // Disabled: O(iterations^2) cost
    opts.estimate_ritz = true;
    opts.store_snapshots = false;

    auto diag = ConjugateGradientDiagnose(
        poisson_op, b, dot, max_iters, tol, x, tmp, z, residual, p, opts );

    // 7) Print diagnostics
    cout << "\n% CG Diagnostic Results for 1D Poisson (Real=" << GENDIL_REAL_TYPE_STR << ")\n";
    cout << "% sizeof(Real) = " << sizeof(Real) << " bytes\n";
    cout << "% n=" << n << ", p=" << order << ", q=" << num_quad_1d << ", ndofs=" << ndofs << "\n";
    cout << "% Converged: " << (diag.success ? "yes" : "no")
         << ", Iterations: " << diag.iterations
         << ", Final relative error: " << to_printable(diag.relative_error) << "\n\n";

    // Output 1: Residual convergence (recursive vs true)
    cout << "% Plot 1: Residual Convergence\n";
    cout << "\\addplot[blue, mark=*, mark size=1pt] coordinates {\n";
    for (const auto& step : diag.steps) {
        cout << "  (" << step.iter << ", " << to_printable(step.relative_recursive_residual) << ")\n";
    }
    cout << "};\n";
    cout << "\\addlegendentry{Recursive residual}\n\n";

    cout << "\\addplot[red, mark=square*, mark size=1pt] coordinates {\n";
    for (const auto& step : diag.steps) {
        cout << "  (" << step.iter << ", " << to_printable(step.relative_true_residual) << ")\n";
    }
    cout << "};\n";
    cout << "\\addlegendentry{True residual}\n\n";

    // Output 2: Residual gap
    cout << "% Plot 2: Residual Gap (|r_recursive - r_true|)\n";
    cout << "\\addplot[orange, mark=triangle*, mark size=1pt] coordinates {\n";
    for (const auto& step : diag.steps) {
        cout << "  (" << step.iter << ", " << to_printable(step.residual_gap_norm) << ")\n";
    }
    cout << "};\n";
    cout << "\\addlegendentry{Residual gap}\n\n";

    // Output 3: Condition number estimates
    cout << "% Plot 3: Estimated Condition Number\n";
    cout << "\\addplot[purple, mark=diamond*, mark size=1pt] coordinates {\n";
    for (const auto& step : diag.steps) {
        if (step.estimated_condition > 0) {
            cout << "  (" << step.iter << ", " << to_printable(step.estimated_condition) << ")\n";
        }
    }
    cout << "};\n";
    cout << "\\addlegendentry{Condition number (Ritz)}\n\n";

    // Output 4: Loss of orthogonality (skipped - requires store_basis=true)

    // Output 5: Min/Max Ritz values
    cout << "% Plot 5: Ritz Value Bounds\n";
    cout << "\\addplot[blue!50, mark=none, dashed] coordinates {\n";
    for (const auto& step : diag.steps) {
        if (step.min_ritz > 0) {
            cout << "  (" << step.iter << ", " << to_printable(step.min_ritz) << ")\n";
        }
    }
    cout << "};\n";
    cout << "\\addlegendentry{Min Ritz value}\n\n";

    cout << "\\addplot[red!50, mark=none, dashed] coordinates {\n";
    for (const auto& step : diag.steps) {
        if (step.max_ritz > 0) {
            cout << "  (" << step.iter << ", " << to_printable(step.max_ritz) << ")\n";
        }
    }
    cout << "};\n";
    cout << "\\addlegendentry{Max Ritz value}\n\n";
}

int main()
{
    constexpr Integer order = 2;
    constexpr Integer num_quad = order + 2;
    constexpr Integer n = 400;

    cout << "\\documentclass{standalone}\n";
    cout << "\\usepackage{pgfplots}\n";
    cout << "\\pgfplotsset{compat=1.18}\n";
    cout << "\\begin{document}\n\n";

    // Plot 1 & 2: Residual convergence and gap
    cout << "\\begin{tikzpicture}\n";
    cout << "\\begin{axis}[\n";
    cout << "  title={CG Residual Convergence (Real=" << GENDIL_REAL_TYPE_STR << ")},\n";
    cout << "  xlabel={Iteration},\n";
    cout << "  ylabel={Relative Residual},\n";
    cout << "  ymode=log,\n";
    cout << "  grid=major,\n";
    cout << "  legend pos=south west,\n";
    cout << "  width=12cm, height=8cm\n";
    cout << "]\n";

    run_cg_diagnostic_1d<order, num_quad>(n);

    cout << "\\end{axis}\n";
    cout << "\\end{tikzpicture}\n\n";

    cout << "\\end{document}\n";

    return 0;
}
