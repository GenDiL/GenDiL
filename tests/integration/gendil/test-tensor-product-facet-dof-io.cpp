// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

struct FullSharedSerialKernelConfiguration : SerialKernelConfiguration
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
   using face_write_dofs_policy = FullSharedFaceWriteDofsPolicy;
};

template < typename Operator >
Vector Apply( const Operator & op, const Vector & input )
{
   Vector output( input.Size() );
   output = 0.0;
   op( input, output );
   return output;
}

bool CheckClose(
   const char * label,
   const Vector & lhs,
   const Vector & rhs )
{
   lhs.ReadHostData();
   rhs.ReadHostData();
   Real error2 = 0.0;
   Real norm2 = 0.0;
   for ( Integer i = 0; i < lhs.Size(); ++i )
   {
      const Real difference = lhs[ i ] - rhs[ i ];
      error2 += difference * difference;
      norm2 += rhs[ i ] * rhs[ i ];
   }
   const Real relative_error =
      std::sqrt( error2 ) / std::max( Real{ 1 }, std::sqrt( norm2 ) );
   if ( relative_error > 1e-12 )
   {
      std::cerr << label << " relative error = " << relative_error << "\n";
      return false;
   }
   return true;
}

struct TensorFixture
{
   Cartesian1DMesh mesh0{ 0.5, 2 };
   Cartesian1DMesh mesh1{ 0.5, 2 };
   std::array< int, 4 > h1_indices{ 0, 1, 1, 2 };
   HostDevicePointer< const int > h1_pointer{};

   TensorFixture()
   {
      h1_pointer.host_pointer = h1_indices.data();
   }

   auto MakeSpace() const
   {
      const auto factor0 = MakeFiniteElementSpace(
         mesh0, GLFiniteElement< 1 >{}, L2Restriction{} );
      const auto factor1 = MakeFiniteElementSpace(
         mesh1,
         GLFiniteElement< 1 >{},
         H1Restriction{ h1_pointer, 3 } );
      const auto mesh = MakeCartesianProductMesh( mesh0, mesh1 );
      const auto restriction =
         MakeTensorProductRestriction( factor0, factor1 );
      return MakeFiniteElementSpace(
         mesh, GLFiniteElement< 1, 1 >{}, restriction );
   }
};

bool TestMeshLocalGenericOperator()
{
   const TensorFixture fixture;
   const auto space = fixture.MakeSpace();
   using Space = std::remove_cvref_t< decltype( space ) >;
   const auto & mesh =
      static_cast< const typename Space::mesh_type & >( space );

   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   const auto form = integrate(
      InteriorFacets< "mesh" >{}, jump( u ) * jump( v ) );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "u" >( space ),
      MakeIntegrationDomain< "mesh" >( mesh ) );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );

   const auto direct = MakeGenericOperator< SerialKernelConfiguration >(
      form, context, integration_rule );
   const auto shared =
      MakeGenericOperator< FullSharedSerialKernelConfiguration >(
         form, context, integration_rule );

   Vector input( space.GetNumberOfFiniteElementDofs() );
   input.WriteHostData();
   for ( Integer i = 0; i < input.Size(); ++i )
   {
      input[ i ] = 1.0 + 0.125 * static_cast< Real >( i );
   }
   return CheckClose(
      "mesh-local tensor-restriction facet operator",
      Apply( direct, input ),
      Apply( shared, input ) );
}

bool TestPartitionGlobalGenericOperator()
{
   const TensorFixture fixture;
   const auto product_space = fixture.MakeSpace();
   using ProductSpace = std::remove_cvref_t< decltype( product_space ) >;
   const auto & product_mesh =
      static_cast< const typename ProductSpace::mesh_type & >( product_space );
   const auto interior_faces =
      MakeCartesianInteriorFaceConnectivity< 2 >( { 2, 2 } );
   const auto partition = MakePartition(
      MakeCellPart( product_mesh ),
      MakeInteriorFacePart< 0, 0 >( interior_faces ) );
   const auto mixed_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ product_space.finite_element },
      std::tuple{ product_space.restriction } );

   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   const auto form = integrate(
      InteriorFacets< "partition" >{}, jump( u ) * jump( v ) );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "u" >( mixed_space ),
      MakeIntegrationDomain< "partition" >( partition ) );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );

   const auto direct = MakeGenericOperator< SerialKernelConfiguration >(
      form, context, integration_rule );
   const auto shared =
      MakeGenericOperator< FullSharedSerialKernelConfiguration >(
         form, context, integration_rule );

   Vector input( mixed_space.GetNumberOfFiniteElementDofs() );
   input.WriteHostData();
   for ( Integer i = 0; i < input.Size(); ++i )
   {
      input[ i ] = 2.0 + 0.25 * static_cast< Real >( i );
   }
   return CheckClose(
      "partition/global tensor-restriction facet operator",
      Apply( direct, input ),
      Apply( shared, input ) );
}

} // namespace

int main()
{
   return TestMeshLocalGenericOperator() &&
         TestPartitionGlobalGenericOperator()
      ? 0
      : 1;
}
