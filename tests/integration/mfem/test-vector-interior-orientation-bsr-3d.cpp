// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gendil;

namespace
{

void rotate_3d_vertices( int * v, int ref_face, int rot )
{
   std::vector< int > face_1, face_2;

   switch ( ref_face / 2 )
   {
      case 0:
         face_1 = { v[0], v[1], v[2], v[3] };
         face_2 = { v[4], v[5], v[6], v[7] };
         break;
      case 1:
         face_1 = { v[1], v[5], v[6], v[2] };
         face_2 = { v[0], v[4], v[7], v[3] };
         break;
      case 2:
         face_1 = { v[4], v[5], v[1], v[0] };
         face_2 = { v[7], v[6], v[2], v[3] };
         break;
   }

   if ( ref_face % 2 == 0 )
   {
      std::reverse( face_1.begin(), face_1.end() );
      std::reverse( face_2.begin(), face_2.end() );
      std::swap( face_1, face_2 );
   }

   std::rotate( face_1.begin(), face_1.begin() + rot, face_1.end() );
   std::rotate( face_2.begin(), face_2.begin() + rot, face_2.end() );

   for ( int i = 0; i < 4; ++i )
   {
      v[i] = face_1[i];
      v[i + 4] = face_2[i];
   }
}

mfem::Mesh MakeOrientationMesh( int face_perm_1, int face_perm_2 )
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;

   mfem::Mesh mesh( dim, nv, nel );

   double x[dim];

   x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex( x );

   x[0] = 0.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex( x );

   x[0] = 0.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex( x );

   x[0] = 0.0; x[1] = 1.0; x[2] = 1.0;
   mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 1.0; x[2] = 1.0;
   mesh.AddVertex( x );

   x[0] = 3.0; x[1] = 1.0; x[2] = 2.0;
   mesh.AddVertex( x );

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices( el, face_perm_1 / 4, face_perm_1 % 4 );
   mesh.AddHex( el );

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices( el, face_perm_2 / 4, face_perm_2 % 4 );
   mesh.AddHex( el );

   mesh.FinalizeHexMesh( true );
   mesh.GenerateBoundaryElements();
   mesh.Finalize();

   return mesh;
}

template < typename VectorType >
Real AbsoluteL2Error( const VectorType & a, const VectorType & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Real err_sq = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
   }
   return std::sqrt( err_sq );
}

template < typename VectorType >
Real RelativeL2Error( const VectorType & a, const VectorType & b )
{
   const Real abs_err = AbsoluteL2Error( a, b );

   Real norm_b_sq = 0.0;
   for ( Integer i = 0; i < b.Size(); ++i )
   {
      norm_b_sq += b[i] * b[i];
   }

   const Real norm_b = std::sqrt( norm_b_sq );
   if ( norm_b == 0.0 )
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

template < typename VectorType >
Real L2Norm( const VectorType & x )
{
   Real norm_sq = 0.0;
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      norm_sq += x[i] * x[i];
   }
   return std::sqrt( norm_sq );
}

template <
   typename FiniteElementSpace,
   typename VectorType,
   size_t Component >
void FillComponentInput(
   const FiniteElementSpace & fe_space,
   VectorType & x,
   std::integral_constant< size_t, Component > component )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   using ComponentDofShape =
      component_dof_shape_t< ShapeFunctions, Component >;

   Real * x_data = x.WriteHostData();
   const GlobalIndex num_elements = fe_space.GetNumberOfFiniteElements();

   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< ComponentDofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex fe_index =
            GlobalDofIndex( fe_space, component, element, indices );
         const LocalIndex local_index =
            FlattenLocalDof( fe_space, component, indices );

         x_data[fe_index] =
            10000.0 * static_cast< Real >( Component + 1 ) +
            100.0 * static_cast< Real >( element + 1 ) +
            0.125 * static_cast< Real >( local_index + 1 ) +
            static_cast< Real >( ( 5 * fe_index + 3 ) % 17 );
      });
   }
}

template < typename FiniteElementSpace, typename VectorType >
void FillDeterministicVectorInput(
   const FiniteElementSpace & fe_space,
   VectorType & x )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   x = -1.0;

   constexpr Integer vdim = ShapeFunctions::vector_dim;
   ConstexprLoop< vdim >( [&] ( auto component )
   {
      FillComponentInput( fe_space, x, component );
   });
}

template <
   Integer order,
   Integer num_quad_1d >
int TestVectorInteriorOrientation( int fp1, int fp2 )
{
   constexpr Integer mesh_order = 1;

   mfem::Mesh mfem_mesh = MakeOrientationMesh( fp1, fp2 );
   HexMesh< mesh_order > mesh = MakeHexMesh< mesh_order >( mfem_mesh );

   // Keep the unstructured-mesh coverage isotropic. Anisotropic tensor shapes
   // are stress-tested at the layout level, where no mesh validity is implied.
   FiniteElementOrders< order, order, order > component0_orders;
   FiniteElementOrders< order, order, order > component1_orders;
   auto component0_fe = MakeLobattoFiniteElement( component0_orders );
   auto component1_fe = MakeLobattoFiniteElement( component1_orders );
   auto vector_finite_element =
      MakeVectorFiniteElement( component0_fe, component1_fe );
   auto vector_fe_space = MakeFiniteElementSpace( mesh, vector_finite_element );

   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout< num_quad_1d, num_quad_1d >;
   constexpr size_t NumSharedDimensions = 3;
   using KernelPolicy =
      ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   InteriorFacets< "mesh1" > interior_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;

   auto tau = MakeCoefficient< "penalty" >(
      [] GENDIL_HOST_DEVICE () -> Real
      {
         return 3.25;
      });

   auto weak_form =
      integrate(
         interior_facets,
         tau * dot( jump( u ), jump( v ) ) );

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField< "u" >( vector_fe_space ),
      MakeIntegrationDomain< "mesh1" >( vector_fe_space ) );

   auto matrix_free_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );

   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );
   const Integer ndofs = vector_fe_space.GetNumberOfFiniteElementDofs();
   Vector x( ndofs );
   Vector y_matrix_free( ndofs );
   Vector y_bsr( ndofs );

   FillDeterministicVectorInput( vector_fe_space, x );
   y_matrix_free = 0.0;
   y_bsr = 0.0;

   matrix_free_operator( x, y_matrix_free );
   sgbsr_matrix( x, y_bsr );

   const Real rel_error = RelativeL2Error( y_bsr, y_matrix_free );

   std::cout << "VECTOR_ORIENTATION_TEST<" << fp1 << ", " << fp2 << ">"
             << " dofs=" << ndofs
             << " |matrix-free|=" << L2Norm( y_matrix_free )
             << " |SGBSR|=" << L2Norm( y_bsr )
             << " rel=" << rel_error
             << std::endl;

   const Real tolerance = 1.0e-11;
   if ( rel_error > tolerance )
   {
      std::cerr
         << "FAILED: vector unstructured interior orientation test<"
         << fp1 << ", " << fp2 << "> exceeded tolerance "
         << tolerance << ".\n";
      return 1;
   }

   return 0;
}

} // namespace

int main()
{
#if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
#elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
#else
   const char device_config[] = "cpu";
#endif

   mfem::Device device( device_config );
   device.Print();

   constexpr Integer order = 2;
   constexpr Integer num_quad_1d = order + 4;

   std::cout
      << "Running vector orientation coverage over all 576 possible "
      << "24x24 face-orientation pairs.\n";

   int failed = 0;
   for ( int fp1 = 0; fp1 < 24; ++fp1 )
   {
      for ( int fp2 = 0; fp2 < 24; ++fp2 )
      {
         failed += TestVectorInteriorOrientation< order, num_quad_1d >(
            fp1,
            fp2 );
      }
   }

   if ( failed == 0 )
   {
      std::cout
         << "SUCCESS: all vector unstructured orientation tests passed.\n";
      return 0;
   }

   std::cout << "FAILED: " << failed
             << " vector unstructured orientation tests failed.\n";
   return 1;
}
