// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <chrono>

using namespace std;
using namespace gendil;

template < typename OperatorFactory, Integer order, Integer num_quad_1d = order + 2 >
void test_speed_of_light_6D( const Integer n1, const Integer n2, const Integer n3,
                             const Integer n4, const Integer n5, const Integer n6,
                             OperatorFactory make_operator )
{
   const Real h_space = 1.0;
   Cartesian3DMesh mesh1( h_space, n1, n2, n3 );
   Cartesian3DMesh mesh2( h_space, n4, n5, n6 );
   auto mesh = MakeCartesianProductMesh( mesh1, mesh2 );

   FiniteElementOrders<order, order, order, order, order, order> orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d,
                            num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 6;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d, num_quad_1d,
                                          num_quad_1d, num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto op = make_operator.template operator()<KernelPolicy>( fe_space, int_rules );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   Vector dofs_in( num_dofs );
   Vector dofs_out( num_dofs );
   dofs_in = 1.0;

   const Integer num_iter = 5;
   GENDIL_DEVICE_SYNC;
   op( dofs_in, dofs_out );
   GENDIL_DEVICE_SYNC;
   auto start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < num_iter; ++iter ) {
      op( dofs_out, dofs_in );
      op( dofs_in, dofs_out );
   }
   GENDIL_DEVICE_SYNC;
   auto end = std::chrono::steady_clock::now();
   double time = std::chrono::duration<double>(end - start).count() / (2 * num_iter);
   double throughput = num_dofs / time;
   std::cout << "          (" << num_dofs << ", " << throughput << ")\n";
}

struct VolumeOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename IntRule>
   auto operator()( const FESpace& fe_space, const IntRule& int_rule ) const {
      return MakeSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

struct FaceReadOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename IntRule>
   auto operator()( const FESpace& fe_space, const IntRule& int_rule ) const {
      return MakeFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

struct FaceWriteOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename IntRule>
   auto operator()( const FESpace& fe_space, const IntRule& int_rule ) const {
      return MakeWriteFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

template < typename Factory, Integer order, Integer num_quad_1d = order + 2 >
void test_sol_range_6D(const std::string& label, Factory factory)
{
   constexpr Integer dim = 6;
   const Integer max_dofs = 1e6;
   Integer n[dim] = {1, 1, 1, 1, 1, 1};
   Integer num_dofs = Pow<dim>(order + 1) * n[0]*n[1]*n[2]*n[3]*n[4]*n[5];
   Integer i = 0;

   std::cout << "       \\addplot coordinates {\n";
   while ( num_dofs < max_dofs ) {
      test_speed_of_light_6D<Factory, order, num_quad_1d>(
         n[0], n[1], n[2], n[3], n[4], n[5], factory );
      n[i] *= 2;
      i = (i + 1) % dim;
      num_dofs *= 2;
   }
   std::cout << "       };\\addlegendentry{" << label << "}\n";
}

int main(int argc, char *argv[])
{
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer max_order = 2;
#else
   constexpr Integer max_order = 4;
#endif
   constexpr Integer num_quad_1d = 2;

   std::cout << "\n6D Speed-of-Light operator benchmarks\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={6D Volume Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range_6D<VolumeOperatorFactory, p, p+num_quad_1d>("vol p=" + std::to_string(p), VolumeOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={6D Face Read Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range_6D<FaceReadOperatorFactory, p, p+num_quad_1d>("read p=" + std::to_string(p), FaceReadOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={6D Face Write Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range_6D<FaceWriteOperatorFactory, p, p+num_quad_1d>("write p=" + std::to_string(p), FaceWriteOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   return 0;
}
