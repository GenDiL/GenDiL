// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

template < Integer order, Integer num_quad_1d = order + 2 >
void benchmark_space_time_maxwell(
    const Integer n1,
    const Integer n2,
    const Integer n3,
    const Integer n4 )
{

   //////////////
   // Mesh
   const Real h_space = 1.0;
   const Real h_time = 1.0;
   Cartesian3DMesh space_mesh( h_space, n1, n2, n3 );
   Cartesian1DMesh time_mesh( h_time, n4 );
   auto mesh = MakeCartesianProductMesh( space_mesh, time_mesh );

   ///////////////////////
   // Finite Element Space
   // orders
   FiniteElementOrders< order, order, order, order > orders;

   // Finite element
   auto fe = MakeLegendreFiniteElement( orders );
   auto vector_fe = MakeVectorFiniteElement(fe, fe, fe, fe, fe, fe);

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   ///////////////////
   // Integration Rule
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d, num_quad_1d > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto eps_field = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X ) -> Real
   {
      return 1.0;
   };
   auto mu_field = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X ) -> Real
   {
      return 1.0;
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto maxwell_operator = MakeMaxwellOperator< KernelPolicy >( fe_space, int_rules, eps_field, mu_field );

   const Integer num_iter = 5;
   double throughput( 0.0 );
   {
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      dofs_in = 1.0;
      maxwell_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         maxwell_operator( dofs_out, dofs_in );
         maxwell_operator( dofs_in, dofs_out );
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
void benchmark_range()
{
   const Integer max_dofs = 1e7;
   constexpr Integer dim = 4;
   Integer n[dim] = { 1, 1, 1, 1 };
   Integer num_dofs = 2 * dim * Pow< dim >( order + 1 ) * n[0] * n[1] * n[2] * n[3];
   Integer i = 0;

   std::cout << "       \\addplot\n";
   std::cout << "       coordinates {\n";
   while ( num_dofs < max_dofs )
   {
      benchmark_space_time_maxwell< order, num_quad_1d >( n[0], n[1], n[2], n[3] );
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
   constexpr Integer max_order = 1;
   constexpr Integer num_quad_1d = 1;
#else
   constexpr Integer max_order = 6;
   constexpr Integer num_quad_1d = 2;
#endif
   std::cout << "\n4d mesh space-time Maxwell operator. \n";
   std::cout << "Orders: 0-" << max_order << "\n";
   std::cout << "Quad pts: order + " << num_quad_1d << "\n\n";

   std::cout << " \\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "    \\begin{axis}[\n";
   std::cout << "       title={4D Space-time Maxwell Operator Application Performance},\n";
   std::cout << "       xlabel={Number of dofs},\n";
   std::cout << "       ylabel={Throughput [DoF/s]},\n";
   std::cout << "       legend pos=outer north east,\n";
   std::cout << "       grid=major,\n";
   std::cout << "       xmode = log,\n";
   std::cout << "       ymode = log,\n";
   std::cout << "       cycle list name=color list,\n";
   std::cout << "       ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ){
      benchmark_range<p,p+num_quad_1d>();
   });
   std::cout << "    \\end{axis}\n";
   std::cout << " \\end{tikzpicture}\n";
   return 0;
}
