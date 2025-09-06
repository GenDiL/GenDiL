// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

template < Integer order, Integer num_quad_1d = order + 2 >
void test_advection_3D( const Integer nx, const Integer ny, const Integer nz )
{

   //////////////
   // Mesh
   const Real h_space = 1.0;
   Cartesian3DMesh mesh( h_space, nx, ny, nz );

   ///////////////////////
   // Finite Element Space
   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLegendreFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule
   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto advection_field = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      v[0] = 1.0;
      v[1] = 1.0;
      v[2] = 1.0;
   };

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = Dim;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto adv_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, advection_field );

   const Integer num_iter = 5;
   double throughput( 0.0 );
   {
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      dofs_in = 1.0;
      adv_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         adv_operator( dofs_out, dofs_in );
         adv_operator( dofs_in, dofs_out );
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
   constexpr Integer dim = 3;
   Integer n[dim] = { 1, 1, 1 };
   Integer num_dofs = Pow< dim >( order + 1 ) * n[0] * n[1] * n[2];
   Integer i = 0;

   std::cout << "       \\addplot\n";
   std::cout << "       coordinates {\n";
   while ( num_dofs < max_dofs )
   {
      test_advection_3D< order, num_quad_1d >( n[0], n[1], n[2] );
      n[i] *= 2;
      i = (i+1)%dim;
      num_dofs *= 2;
   }
   std::cout << "       };\n";
   std::cout << "       \\addlegendentry{p=" << order << ", q=" << num_quad_1d << "}\n";
}

int main(int argc, char *argv[])
{
   constexpr Integer max_order = 6;
   constexpr Integer num_quad_1d = 2;
   std::cout << "\n3d mesh advection operator. \n";
   std::cout << "Orders: 0-" << max_order << "\n";
   std::cout << "Quad pts: order + " << num_quad_1d << "\n\n";

   std::cout << " \\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "    \\begin{axis}[\n";
   std::cout << "       title={3D Advection Operator Application Performance},\n";
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
