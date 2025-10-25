// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

template < Integer order, Integer num_quad_1d = order + 2 >
void test_mass_5D(
    const Integer n1,
    const Integer n2,
    const Integer n3,
    const Integer n4,
    const Integer n5 )
{

   //////////////
   // Mesh
   const Real h_space = Real(1.0);
   Cartesian3DMesh mesh1( h_space, n1, n2, n3 );
   Cartesian2DMesh mesh2( h_space, n4, n5 );
   auto mesh = MakeCartesianProductMesh( mesh1, mesh2 );

   ///////////////////////
   // Finite Element Space
   // orders
   FiniteElementOrders< order, order, order, order, order > orders;

   // Finite element
   auto finite_element = MakeLegendreFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d, num_quad_1d, num_quad_1d > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      return Real(1.0);
   };

#if defined(GENDIL_USE_DEVICE)
//    using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
//    using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   const Integer num_iter = 5;
   double throughput( 0.0 );
   {
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      dofs_in = Real(1.0);
      mass_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         mass_operator( dofs_out, dofs_in );
         mass_operator( dofs_in, dofs_out );
      }
      GENDIL_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      const std::chrono::duration<double> elapsed_seconds = end - start;
      const Integer total_iter = 2 * num_iter;
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      throughput = num_dofs / time_per_iter;
      std::cout << "          (" << num_dofs << ", " << throughput << ")\n";
   }
}

template < Integer order, Integer num_quad_1d = order + 2 >
void test_range()
{
   const Integer max_dofs = 1e8;
   constexpr Integer dim = 5;
   Integer n[dim] = { 1, 1, 1, 1, 1 };
   Integer num_dofs = Pow< dim >( order + 1 ) * n[0] * n[1] * n[2] * n[3] * n[4];
   Integer i = 0;

   std::cout << "       \\addplot\n";
   std::cout << "       coordinates {\n";
   while ( num_dofs < max_dofs )
   {
      test_mass_5D< order, num_quad_1d >( n[0], n[1], n[2], n[3], n[4] );
      n[i] *= 2;
      i = (i+1)%dim;
      num_dofs *= 2;
   }
   std::cout << "       };\n";
   std::cout << "       \\addlegendentry{p=" << order << ", q=" << num_quad_1d << "}\n";
}

int main(int argc, char *argv[])
{
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer max_order = 2;
   constexpr Integer num_quad_1d = 2;
#else
   constexpr Integer max_order = 6;
   constexpr Integer num_quad_1d = 2;
#endif
   std::cout << "\n5d mesh mass operator. \n";
   std::cout << "Orders: 0-" << max_order << "\n";
   std::cout << "Quad pts: order + " << num_quad_1d << "\n\n";

   std::cout << " \\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "    \\begin{axis}[\n";
   std::cout << "       title={5D Mass Operator Application Performance},\n";
   std::cout << "       xlabel={Number of dofs},\n";
   std::cout << "       ylabel={Throughput [DoF/s]},\n";
   std::cout << "       legend pos=outer north east,\n";
   std::cout << "       grid=major,\n";
   std::cout << "       xmode = log,\n";
   std::cout << "       ymode = log,\n";
   std::cout << "       cycle list name=color list,\n";
   std::cout << "       ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ){
      test_range<p,p+num_quad_1d>();
   });
   std::cout << "    \\end{axis}\n";
   std::cout << " \\end{tikzpicture}\n";
   return 0;
}
