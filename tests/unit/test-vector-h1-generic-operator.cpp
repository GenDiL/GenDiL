// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

void FillInputCase( Vector & x, const Integer case_id )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] = 0.0;
   }

   if ( case_id == 0 )
   {
      for ( GlobalIndex i = 0; i < x.Size(); ++i )
      {
         data[i] = 0.5 + 0.25 * static_cast< Real >( i );
      }
   }
   else if ( case_id == 1 )
   {
      data[0] = 1.0;
      data[1] = 2.0;
      data[2] = 3.0;
   }
   else
   {
      data[3] = -1.0;
      data[4] = 0.5;
      data[5] = 4.0;
   }
}

void ApplyTwoCellVectorH1P1MassReference(
   const Vector & x,
   Vector & y,
   const Real h )
{
   // Two uniform 1D p1 H1 elements have local scalar mass
   // (h / 6) * [[2, 1], [1, 2]]. Assembling the shared middle
   // node gives the component-major diag(M_scalar, M_scalar) action.
   const Real scale = h / 6.0;
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = 0.0;
   }

   for ( GlobalIndex component = 0; component < 2; ++component )
   {
      const GlobalIndex offset = 3 * component;
      y_data[offset + 0] =
         scale * ( 2.0 * x_data[offset + 0] + x_data[offset + 1] );
      y_data[offset + 1] =
         scale * ( x_data[offset + 0] +
                   4.0 * x_data[offset + 1] +
                   x_data[offset + 2] );
      y_data[offset + 2] =
         scale * ( x_data[offset + 1] + 2.0 * x_data[offset + 2] );
   }
}

bool CheckVectorNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < actual.Size(); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

bool TestVectorH1GenericOperatorCellMass()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > cells;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( cells, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;
   auto mass_op =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      fe_space.GetNumberOfFiniteElementDofs() == 6,
      "Vector H1 GenericOperator test expected component-major true DoFs [c0:0..2, c1:3..5]." ) && success;

   for ( Integer case_id = 0; case_id < 3; ++case_id )
   {
      Vector x( fe_space.GetNumberOfFiniteElementDofs() );
      Vector y( fe_space.GetNumberOfFiniteElementDofs() );
      Vector y_expected( fe_space.GetNumberOfFiniteElementDofs() );

      FillInputCase( x, case_id );
      y = 0.0;
      mass_op( x, y );
      ApplyTwoCellVectorH1P1MassReference( x, y_expected, h );

      success = CheckVectorNear(
         y,
         y_expected,
         "Vector H1 GenericOperator action disagrees with the manual p1 mass reference." ) && success;

      const Real * y_data = y.ReadHostData();
      const Real * expected_data = y_expected.ReadHostData();

      if ( case_id == 1 )
      {
         success = Check(
            Near( y_data[1], expected_data[1] ),
            "Vector H1 GenericOperator did not accumulate the component 0 shared middle node." ) && success;
         success = Check(
            Near( y_data[3], 0.0 ) &&
            Near( y_data[4], 0.0 ) &&
            Near( y_data[5], 0.0 ),
            "Vector H1 GenericOperator aliased component 0 input into component 1 output." ) && success;
      }
      else if ( case_id == 2 )
      {
         success = Check(
            Near( y_data[4], expected_data[4] ),
            "Vector H1 GenericOperator did not accumulate the component 1 shared middle node." ) && success;
         success = Check(
            Near( y_data[0], 0.0 ) &&
            Near( y_data[1], 0.0 ) &&
            Near( y_data[2], 0.0 ),
            "Vector H1 GenericOperator aliased component 1 input into component 0 output." ) && success;
      }
   }

   return success;
}

} // namespace

int main()
{
   return TestVectorH1GenericOperatorCellMass() ? 0 : 1;
}
