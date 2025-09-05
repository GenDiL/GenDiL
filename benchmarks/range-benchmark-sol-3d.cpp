// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>
#include <chrono>

using namespace std;
using namespace gendil;

template < typename OperatorFactory, Integer order, Integer num_quad_1d = order + 2 >
void test_speed_of_light_3D( const Integer nx, const Integer ny, const Integer nz, OperatorFactory make_operator )
{
   const Real h_space = 1.0;
   Cartesian3DMesh mesh( h_space, nx, ny, nz );

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 3;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d, num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto op = make_operator.template operator()<KernelPolicy>( fe_space, face_meshes, int_rules );

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
   template <typename KernelPolicy, typename FESpace, typename FaceMeshes, typename IntRule>
   auto operator()( const FESpace& fe_space, const FaceMeshes& face_meshes, const IntRule& int_rule ) const {
      return MakeSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

struct FaceReadOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename FaceMeshes, typename IntRule>
   auto operator()( const FESpace& fe_space, const FaceMeshes& face_meshes, const IntRule& int_rule ) const {
      return MakeFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

struct FaceWriteOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename FaceMeshes, typename IntRule>
   auto operator()( const FESpace& fe_space, const FaceMeshes& face_meshes, const IntRule& int_rule ) const {
      return MakeWriteFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rule );
   }
};

struct GlobalFaceOperatorFactory {
   template <typename KernelPolicy, typename FESpace, typename FaceMeshes, typename IntRule>
   auto operator()( const FESpace& fe_space, const FaceMeshes& face_meshes, const IntRule& int_rule ) const {
      return MakeGlobalFaceSpeedOfLightOperator<KernelPolicy>( fe_space, face_meshes, int_rule );
   }
};

template < typename Factory, Integer order, Integer num_quad_1d = order + 2 >
void test_sol_range(const std::string& label, Factory factory)
{
   constexpr Integer dim = 3;
   const Integer max_dofs = 1e7;
   Integer n[dim] = {1, 1, 1};
   Integer num_dofs = Pow<dim>(order + 1) * n[0] * n[1] * n[2];
   Integer i = 0;

   std::cout << "       \\addplot coordinates {\n";
   while ( num_dofs < max_dofs ) {
      test_speed_of_light_3D<Factory, order, num_quad_1d>( n[0], n[1], n[2], factory );
      n[i] *= 2;
      i = (i + 1) % dim;
      num_dofs *= 2;
   }
   std::cout << "       };\\addlegendentry{" << label << "}\n";
}

int main(int argc, char *argv[])
{
   constexpr Integer max_order = 6;
   constexpr Integer num_quad_1d = 2;

   std::cout << "\n3D Speed-of-Light operator benchmarks\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={Volume Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range<VolumeOperatorFactory, p, p+num_quad_1d>("vol p=" + std::to_string(p), VolumeOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={Face Read Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range<FaceReadOperatorFactory, p, p+num_quad_1d>("read p=" + std::to_string(p), FaceReadOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={Face Write Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=outer north east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range<FaceWriteOperatorFactory, p, p+num_quad_1d>("write p=" + std::to_string(p), FaceWriteOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   std::cout << "\n\\begin{tikzpicture}[scale=0.9]\n";
   std::cout << "  \\begin{axis}[\n";
   std::cout << "    title={Global Face Speed-of-Light Operator},\n";
   std::cout << "    xlabel={Number of DoFs},\n";
   std::cout << "    ylabel={Throughput [DoF/s]},\n";
   std::cout << "    legend pos=south east,\n";
   std::cout << "    grid=major, xmode=log, ymode=log, cycle list name=color list\n";
   std::cout << "  ]\n";
   ConstexprLoop<max_order+1>( [=]( auto p ) {
      test_sol_range<GlobalFaceOperatorFactory, p, p+num_quad_1d>("write p=" + std::to_string(p), GlobalFaceOperatorFactory{});
   });
   std::cout << "  \\end{axis}\n\\end{tikzpicture}\n";

   return 0;
}
