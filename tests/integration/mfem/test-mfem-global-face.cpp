// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace gendil;

namespace
{

#if defined(GENDIL_USE_DEVICE)
template <Integer NumQuad1D>
using GlobalFaceKernelPolicy =
   DeviceKernelConfiguration<ThreadBlockLayout<NumQuad1D>, 1, 2>;
#else
template <Integer>
using GlobalFaceKernelPolicy = SerialKernelConfiguration;
#endif

Real TestTolerance( Real scale = Real(4.0e6) )
{
   return scale * std::numeric_limits< Real >::epsilon();
}

bool Check( bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << "FAILED: " << message << "\n";
   }
   return condition;
}

bool CheckNear( Real a, Real b, const char * message )
{
   const Real error = std::abs( a - b );
   if ( error > TestTolerance() )
   {
      std::cerr << "FAILED: " << message << " got " << a << " expected "
                << b << " error " << error << "\n";
      return false;
   }
   return true;
}

template < Integer Dim >
bool CheckPointNear(
   const Point< Dim > & a,
   const Point< Dim > & b,
   const char * message )
{
   bool ok = true;
   for ( Integer d = 0; d < Dim; ++d )
   {
      const Real error = std::abs( a[ d ] - b[ d ] );
      if ( error > TestTolerance() )
      {
         std::cerr << "FAILED: " << message << " dim " << d
                   << " got " << a[ d ] << " expected " << b[ d ]
                   << " error " << error << "\n";
         ok = false;
      }
   }
   return ok;
}

template < Integer Dim >
bool PointsDiffer(
   const Point< Dim > & a,
   const Point< Dim > & b )
{
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( std::abs( a[d] - b[d] ) > TestTolerance() )
      {
         return true;
      }
   }
   return false;
}

template < Integer Dim >
Point< Dim > ToPoint( const mfem::IntegrationPoint & ip )
{
   Point< Dim > p;
   if constexpr ( Dim >= 1 ) { p[ 0 ] = ip.x; }
   if constexpr ( Dim >= 2 ) { p[ 1 ] = ip.y; }
   if constexpr ( Dim >= 3 ) { p[ 2 ] = ip.z; }
   return p;
}

template < Integer Dim >
mfem::IntegrationPoint ToMFEMFacePoint( const std::array< Real, Dim - 1 > & x )
{
   mfem::IntegrationPoint ip;
   if constexpr ( Dim >= 2 ) { ip.x = x[ 0 ]; }
   if constexpr ( Dim >= 3 ) { ip.y = x[ 1 ]; }
   return ip;
}

template < Integer Dim >
Point< Dim > MakeCanonicalFacePoint(
   Integer face,
   const std::array< Real, Dim - 1 > & tangential )
{
   Point< Dim > point{};
   const Integer normal_axis = HyperCube< Dim >::GetNormalDimensionIndex( face );
   Integer tangential_axis = 0;
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d == normal_axis )
      {
         point[d] = face < Dim ? Real(0.0) : Real(1.0);
      }
      else
      {
         point[d] = tangential[ tangential_axis++ ];
      }
   }
   return point;
}

template < Integer Dim >
Point< Dim > ApplyOrientationContinuous(
   const Point< Dim > & reference,
   const Permutation< Dim > & orientation )
{
   Point< Dim > native;
   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex entry = orientation( native_axis );
      const Integer reference_axis = entry > 0 ? entry - 1 : -entry - 1;
      native[ native_axis ] =
         entry > 0 ? reference[ reference_axis ]
                   : Real(1.0) - reference[ reference_axis ];
   }
   return native;
}

template < Integer Dim >
Point< Dim > StaticOppositeFacePoint(
   Point< Dim > minus_point,
   Integer canonical_minus_face )
{
   const Integer axis =
      HyperCube< Dim >::GetNormalDimensionIndex( canonical_minus_face );
   minus_point[ axis ] =
      canonical_minus_face < Dim ? Real(1.0) : Real(0.0);
   return minus_point;
}

template < typename Tuple >
GlobalIndex CountFaces( const Tuple & tuple )
{
   GlobalIndex count = 0;
   std::apply(
      [&] ( const auto & ... family )
      {
         ( ( count += family.GetNumberOfFaces() ), ... );
      },
      tuple );
   return count;
}

template < typename Geometry, Integer FaceIndex >
bool CheckInteriorMetadataFamily(
   const mfem::Mesh & mesh,
   const MFEMConformingInteriorConnectivityBundle< Geometry > & bundle,
   std::integral_constant< Integer, FaceIndex > )
{
   const auto & family = std::get< FaceIndex >( bundle.connectivity );
   const auto & metadata = std::get< FaceIndex >( bundle.metadata );
   bool ok = Check(
      static_cast< size_t >( family.GetNumberOfFaces() ) == metadata.size(),
      "interior metadata size does not match family record count" );

   int previous_source = -1;
   for ( GlobalIndex k = 0; k < family.GetNumberOfFaces(); ++k )
   {
      const auto record = family.records.host_pointer[ k ];
      const auto meta = metadata[ static_cast< size_t >( k ) ];
      const auto face_info = mesh.GetFaceInformation( meta.source_face_id );
      ok = Check( meta.source_face_id > previous_source,
                  "interior metadata source ids are not strictly sorted" ) && ok;
      ok = Check( record.minus_cell ==
                    static_cast< GlobalIndex >( face_info.element[0].index ),
                  "interior metadata does not describe record minus cell" ) && ok;
      ok = Check( record.plus_cell ==
                    static_cast< GlobalIndex >( face_info.element[1].index ),
                  "interior metadata does not describe record plus cell" ) && ok;
      previous_source = meta.source_face_id;
   }
   return ok;
}

template < typename Geometry >
bool CheckInteriorMetadataAlignment(
   const mfem::Mesh & mesh,
   const MFEMConformingInteriorConnectivityBundle< Geometry > & bundle )
{
   bool ok = true;
   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto face )
      {
         constexpr Integer FaceIndex =
            static_cast< Integer >( decltype( face )::value );
         ok = CheckInteriorMetadataFamily(
                 mesh,
                 bundle,
                 std::integral_constant< Integer, FaceIndex >{} ) && ok;
      } );
   return ok;
}

mfem::Mesh MakeTwoQuadMesh( int face_perm_1, int face_perm_2 )
{
   constexpr int dim = 2;
   constexpr int nv = 6;
   constexpr int nel = 2;
   mfem::Mesh mesh( dim, nv, nel );
   double x[ dim ];
   x[0] = 0.0; x[1] = 0.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 0.0; mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 0.0; mesh.AddVertex( x );
   x[0] = 0.0; x[1] = 1.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 1.0; mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 1.0; mesh.AddVertex( x );

   int el[4] = {0, 1, 4, 3};
   std::rotate( &el[0], &el[face_perm_1], &el[4] );
   mesh.AddQuad( el );

   el[0] = 1; el[1] = 2; el[2] = 5; el[3] = 4;
   std::rotate( &el[0], &el[face_perm_2], &el[4] );
   mesh.AddQuad( el );

   mesh.FinalizeQuadMesh( true );
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

void Rotate3DVertices( int * v, int ref_face, int rot )
{
   std::vector< int > face_1, face_2;
   switch ( ref_face / 2 )
   {
      case 0:
         face_1 = {v[0], v[1], v[2], v[3]};
         face_2 = {v[4], v[5], v[6], v[7]};
         break;
      case 1:
         face_1 = {v[1], v[5], v[6], v[2]};
         face_2 = {v[0], v[4], v[7], v[3]};
         break;
      default:
         face_1 = {v[4], v[5], v[1], v[0]};
         face_2 = {v[7], v[6], v[2], v[3]};
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
      v[ i ] = face_1[ i ];
      v[ i + 4 ] = face_2[ i ];
   }
}

mfem::Mesh MakeTwoHexMesh( int face_perm_1, int face_perm_2 )
{
   constexpr int dim = 3;
   constexpr int nv = 12;
   constexpr int nel = 2;
   mfem::Mesh mesh( dim, nv, nel );
   double x[ dim ];

   x[0] = 0.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 0.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex( x );
   x[0] = 0.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex( x );
   x[0] = 2.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex( x );
   x[0] = 0.0; x[1] = 1.0; x[2] = 1.0; mesh.AddVertex( x );
   x[0] = 1.0; x[1] = 1.0; x[2] = 1.0; mesh.AddVertex( x );
   x[0] = 3.0; x[1] = 1.0; x[2] = 2.0; mesh.AddVertex( x );

   int el[8] = {0, 1, 4, 3, 6, 7, 10, 9};
   Rotate3DVertices( el, face_perm_1 / 4, face_perm_1 % 4 );
   mesh.AddHex( el );

   el[0] = 1; el[1] = 2; el[2] = 5; el[3] = 4;
   el[4] = 7; el[5] = 8; el[6] = 11; el[7] = 10;
   Rotate3DVertices( el, face_perm_2 / 4, face_perm_2 % 4 );
   mesh.AddHex( el );

   mesh.FinalizeHexMesh( true );
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

mfem::IntegrationPoint MFEMFacePointFromMinusPoint(
   mfem::IntegrationPointTransformation & loc1,
   const Point<2> & minus_point )
{
   mfem::IntegrationPoint ip0, ip1, e0, e1, face_ip;
   ip0.x = 0.0;
   ip1.x = 1.0;
   loc1.Transform( ip0, e0 );
   loc1.Transform( ip1, e1 );
   const Point<2> a = ToPoint<2>( e0 );
   const Point<2> b = ToPoint<2>( e1 );
   const Real dx = b[0] - a[0];
   const Real dy = b[1] - a[1];
   const Real denom = dx * dx + dy * dy;
   GENDIL_VERIFY( denom > 0.0, "Degenerate MFEM face map in test reference." );
   face_ip.x =
      ( ( minus_point[0] - a[0] ) * dx +
        ( minus_point[1] - a[1] ) * dy ) / denom;
   return face_ip;
}

mfem::IntegrationPoint MFEMFacePointFromMinusPoint(
   mfem::IntegrationPointTransformation & loc1,
   const Point<3> & minus_point )
{
   mfem::IntegrationPoint ip00, ip10, ip01, e00, e10, e01, face_ip;
   ip00.x = 0.0;
   ip00.y = 0.0;
   ip10.x = 1.0;
   ip10.y = 0.0;
   ip01.x = 0.0;
   ip01.y = 1.0;
   loc1.Transform( ip00, e00 );
   loc1.Transform( ip10, e10 );
   loc1.Transform( ip01, e01 );

   const Point<3> a = ToPoint<3>( e00 );
   const Point<3> b = ToPoint<3>( e10 );
   const Point<3> c = ToPoint<3>( e01 );
   Real d0[3], d1[3], rhs[3];
   for ( Integer d = 0; d < 3; ++d )
   {
      d0[d] = b[d] - a[d];
      d1[d] = c[d] - a[d];
      rhs[d] = minus_point[d] - a[d];
   }

   Real g00 = 0.0, g01 = 0.0, g11 = 0.0, r0 = 0.0, r1 = 0.0;
   for ( Integer d = 0; d < 3; ++d )
   {
      g00 += d0[d] * d0[d];
      g01 += d0[d] * d1[d];
      g11 += d1[d] * d1[d];
      r0 += rhs[d] * d0[d];
      r1 += rhs[d] * d1[d];
   }

   const Real det = g00 * g11 - g01 * g01;
   GENDIL_VERIFY( det > 0.0, "Degenerate MFEM square face map in test reference." );
   face_ip.x = ( r0 * g11 - r1 * g01 ) / det;
   face_ip.y = ( g00 * r1 - g01 * r0 ) / det;
   return face_ip;
}

template < Integer Dim >
bool CheckCoordinateAgreement(
   mfem::Mesh & mesh,
   int source_face_id,
   Integer g0,
   const Permutation< Dim > & plus_orientation,
   const std::vector< std::array< Real, Dim - 1 > > & face_points )
{
   auto * ft = mesh.GetFaceElementTransformations( source_face_id );
   bool ok = Check( ft != nullptr, "missing MFEM face transformations" );
   for ( const auto & x : face_points )
   {
      const Point< Dim > minus_native =
         MakeCanonicalFacePoint< Dim >( g0, x );
      mfem::IntegrationPoint face_ip =
         MFEMFacePointFromMinusPoint( ft->Loc1, minus_native );
      mfem::IntegrationPoint eip1, eip2;
      ft->Loc1.Transform( face_ip, eip1 );
      ft->Loc2.Transform( face_ip, eip2 );

      ok = CheckPointNear(
              ToPoint< Dim >( eip1 ),
              minus_native,
              "MFEM Loc1 inverse recovered canonical minus point" ) && ok;
      const Point< Dim > plus_native_mfem = ToPoint< Dim >( eip2 );
      const auto plus_static = StaticOppositeFacePoint( minus_native, g0 );
      const auto plus_native_gendil =
         ApplyOrientationContinuous( plus_static, plus_orientation );
      ok = CheckPointNear(
              plus_native_gendil,
              plus_native_mfem,
              "full reference-coordinate agreement with MFEM Loc2" ) && ok;
   }
   return ok;
}

template < Integer Dim >
bool CheckIdentityOrientationWouldDisagree(
   mfem::Mesh & mesh,
   int source_face_id,
   Integer g0,
   const std::vector< std::array< Real, Dim - 1 > > & face_points )
{
   auto * ft = mesh.GetFaceElementTransformations( source_face_id );
   bool ok = Check( ft != nullptr, "missing MFEM face transformations" );
   bool found_difference = false;
   for ( const auto & x : face_points )
   {
      const Point< Dim > minus_native =
         MakeCanonicalFacePoint< Dim >( g0, x );
      mfem::IntegrationPoint face_ip =
         MFEMFacePointFromMinusPoint( ft->Loc1, minus_native );
      mfem::IntegrationPoint eip2;
      ft->Loc2.Transform( face_ip, eip2 );

      const auto plus_without_orientation =
         StaticOppositeFacePoint( minus_native, g0 );
      found_difference = found_difference ||
         PointsDiffer( plus_without_orientation, ToPoint< Dim >( eip2 ) );
   }
   ok = Check(
      found_difference,
      "identity plus-orientation negative control matched MFEM Loc2" ) && ok;
   return ok;
}

Real SegmentFaceWeight(
   mfem::FaceElementTransformations & ft,
   Real t )
{
   mfem::IntegrationPoint face_ip;
   face_ip.x = t;
   ft.SetAllIntPoints( &face_ip );
   return ft.Face->Weight();
}

Real MFEMFaceCoordinateFromMinusPoint(
   mfem::IntegrationPointTransformation & loc1,
   const Point<2> & minus_point )
{
   mfem::IntegrationPoint ip0, ip1, e0, e1;
   ip0.x = 0.0;
   ip1.x = 1.0;
   loc1.Transform( ip0, e0 );
   loc1.Transform( ip1, e1 );
   const Point<2> a = ToPoint<2>( e0 );
   const Point<2> b = ToPoint<2>( e1 );
   const Real dx = b[0] - a[0];
   const Real dy = b[1] - a[1];
   const Real denom = dx * dx + dy * dy;
   GENDIL_VERIFY( denom > 0.0, "Degenerate MFEM face map in test reference." );
   return ( ( minus_point[0] - a[0] ) * dx +
            ( minus_point[1] - a[1] ) * dy ) / denom;
}

template < typename IR, size_t... I >
Real TensorWeightImpl(
   const typename IR::index_type & q,
   std::index_sequence< I... > )
{
   return ( Real(1.0) * ... *
      std::tuple_element_t<
         I,
         typename IR::points::points_1d_tuple >::GetWeight( q[ I ] ) );
}

template < typename IR >
Real TensorWeight( const typename IR::index_type & q )
{
   return TensorWeightImpl< IR >(
      q,
      std::make_index_sequence< IR::space_dim >{} );
}

Real LobattoP2BasisValue(
   const std::array< GlobalIndex, 2 > & dof,
   const Point<2> & p )
{
   using Shape1D = GaussLobattoLegendreShapeFunctions<2>;
   return Shape1D::ComputeValue( dof[0], Point<1>{ p[0] } ) *
          Shape1D::ComputeValue( dof[1], Point<1>{ p[1] } );
}

template < typename FESpace >
Real ElementValueP2(
   const FESpace & fe_space,
   const Vector & x,
   GlobalIndex element,
   const Point<2> & p )
{
   Real value = 0.0;
   for ( GlobalIndex i = 0; i < 3; ++i )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         const std::array< GlobalIndex, 2 > dof{i, j};
         const auto flat = FlattenLocalDof( fe_space, dof );
         const auto gdof = GlobalDofIndex( fe_space, element, flat );
         value += x[ gdof ] * LobattoP2BasisValue( dof, p );
      }
   }
   return value;
}

template < typename FESpace >
void AddElementResidualP2(
   const FESpace & fe_space,
   Vector & y,
   GlobalIndex element,
   const Point<2> & p,
   Real scale )
{
   auto * data = y.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < 3; ++i )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         const std::array< GlobalIndex, 2 > dof{i, j};
         const auto flat = FlattenLocalDof( fe_space, dof );
         const auto gdof = GlobalDofIndex( fe_space, element, flat );
         data[ gdof ] += scale * LobattoP2BasisValue( dof, p );
      }
   }
}

template < typename FESpace, Integer FaceIndex >
void AssembleInteriorReferenceFamily(
   mfem::Mesh & mfem_mesh,
   const FESpace & fe_space,
   const MFEMConformingInteriorConnectivityBundle< HyperCube<2> > & bundle,
   const Vector & x,
   Vector & y,
   bool ignore_plus_orientation,
   std::integral_constant< Integer, FaceIndex > )
{
   auto volume_ir = MakeIntegrationRule( IntegrationRuleNumPoints<5, 5>{} );
   auto face_rules = GetFaceIntegrationRules( volume_ir );
   auto face_ir = std::get< FaceIndex >( face_rules );
   using FaceIR = decltype( face_ir );

   const auto & family = std::get< FaceIndex >( bundle.connectivity );
   const auto & metadata = std::get< FaceIndex >( bundle.metadata );
   for ( GlobalIndex face = 0; face < family.GetNumberOfFaces(); ++face )
   {
      const auto record = family.records.host_pointer[ face ];
      auto * ft = mfem_mesh.GetFaceElementTransformations(
         metadata[ static_cast< size_t >( face ) ].source_face_id );

      QuadraturePointLoop< FaceIR >(
         [&] ( const auto & q )
         {
            const auto q_minus = FaceIR::GetPoint( q );
            const Real t = MFEMFaceCoordinateFromMinusPoint( ft->Loc1, q_minus );
            mfem::IntegrationPoint face_ip;
            face_ip.x = t;
            mfem::IntegrationPoint eip1, eip2;
            ft->Loc1.Transform( face_ip, eip1 );
            ft->Loc2.Transform( face_ip, eip2 );

            const Point<2> p_minus = ToPoint<2>( eip1 );
            const Point<2> p_plus =
               ignore_plus_orientation
                  ? StaticOppositeFacePoint( q_minus, FaceIndex )
                  : ToPoint<2>( eip2 );
            const Real u_minus =
               ElementValueP2( fe_space, x, record.minus_cell, p_minus );
            const Real u_plus =
               ElementValueP2( fe_space, x, record.plus_cell, p_plus );
            const Real jump = u_minus - u_plus;
            const Real weight =
               TensorWeight< FaceIR >( q ) * SegmentFaceWeight( *ft, t );

            AddElementResidualP2(
               fe_space,
               y,
               record.minus_cell,
               p_minus,
               weight * jump );
            AddElementResidualP2(
               fe_space,
               y,
               record.plus_cell,
               p_plus,
               -weight * jump );
         } );
   }
}

template < typename FESpace >
Vector AssembleInteriorJumpReference(
   mfem::Mesh & mfem_mesh,
   const FESpace & fe_space,
   const MFEMConformingInteriorConnectivityBundle< HyperCube<2> > & bundle,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;
   ConstexprLoop< HyperCube<2>::num_faces >(
      [&] ( auto face )
      {
         constexpr Integer FaceIndex =
            static_cast< Integer >( decltype( face )::value );
         AssembleInteriorReferenceFamily(
            mfem_mesh,
            fe_space,
            bundle,
            x,
            y,
            false,
            std::integral_constant< Integer, FaceIndex >{} );
      } );
   return y;
}

template < typename FESpace >
Vector AssembleInteriorJumpReferenceIgnoringPlusOrientation(
   mfem::Mesh & mfem_mesh,
   const FESpace & fe_space,
   const MFEMConformingInteriorConnectivityBundle< HyperCube<2> > & bundle,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;
   ConstexprLoop< HyperCube<2>::num_faces >(
      [&] ( auto face )
      {
         constexpr Integer FaceIndex =
            static_cast< Integer >( decltype( face )::value );
         AssembleInteriorReferenceFamily(
            mfem_mesh,
            fe_space,
            bundle,
            x,
            y,
            true,
            std::integral_constant< Integer, FaceIndex >{} );
      } );
   return y;
}

template < typename FESpace, Integer FaceIndex >
void AssembleBoundaryReferenceFamily(
   mfem::Mesh & mfem_mesh,
   const FESpace & fe_space,
   const MFEMBoundaryConnectivityBundle< HyperCube<2> > & bundle,
   const Vector & x,
   Vector & y,
   std::integral_constant< Integer, FaceIndex > )
{
   auto volume_ir = MakeIntegrationRule( IntegrationRuleNumPoints<5, 5>{} );
   auto face_rules = GetFaceIntegrationRules( volume_ir );
   auto face_ir = std::get< FaceIndex >( face_rules );
   using FaceIR = decltype( face_ir );

   const auto & family = std::get< FaceIndex >( bundle.connectivity );
   const auto & metadata = std::get< FaceIndex >( bundle.metadata );
   for ( GlobalIndex face = 0; face < family.GetNumberOfFaces(); ++face )
   {
      const auto record = family.records.host_pointer[ face ];
      auto * ft = mfem_mesh.GetFaceElementTransformations(
         metadata[ static_cast< size_t >( face ) ].source_face_id );

      QuadraturePointLoop< FaceIR >(
         [&] ( const auto & q )
         {
            const auto q_minus = FaceIR::GetPoint( q );
            const Real t = MFEMFaceCoordinateFromMinusPoint( ft->Loc1, q_minus );
            mfem::IntegrationPoint face_ip;
            face_ip.x = t;
            mfem::IntegrationPoint eip1;
            ft->Loc1.Transform( face_ip, eip1 );

            const Point<2> p = ToPoint<2>( eip1 );
            const Real u = ElementValueP2( fe_space, x, record.cell, p );
            const Real weight =
               TensorWeight< FaceIR >( q ) * SegmentFaceWeight( *ft, t );
            AddElementResidualP2( fe_space, y, record.cell, p, weight * u );
         } );
   }
}

template < typename FESpace >
Vector AssembleBoundaryMassReference(
   mfem::Mesh & mfem_mesh,
   const FESpace & fe_space,
   const MFEMBoundaryConnectivityBundle< HyperCube<2> > & bundle,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;
   ConstexprLoop< HyperCube<2>::num_faces >(
      [&] ( auto face )
      {
         constexpr Integer FaceIndex =
            static_cast< Integer >( decltype( face )::value );
         AssembleBoundaryReferenceFamily(
            mfem_mesh,
            fe_space,
            bundle,
            x,
            y,
            std::integral_constant< Integer, FaceIndex >{} );
      } );
   return y;
}

bool CheckVectorNear( const Vector & a, const Vector & b, const char * label )
{
   bool ok = Check( a.Size() == b.Size(), "vector size mismatch" );
   Real err2 = 0.0;
   Real ref2 = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real diff = a[i] - b[i];
      err2 += diff * diff;
      ref2 += b[i] * b[i];
   }
   const Real rel = std::sqrt( err2 ) /
      ( std::sqrt( ref2 ) + std::numeric_limits< Real >::epsilon() );
   std::cout << label << " relative L2 error = " << rel << "\n";
   ok = Check( rel < TestTolerance( Real(5.0e7) ), label ) && ok;
   return ok;
}

bool CheckVectorNearMFEM(
   const Vector & a,
   const mfem::Vector & b,
   const char * label )
{
   bool ok = Check(
      a.Size() == static_cast< size_t >( b.Size() ),
      "vector size mismatch" );
   const Real * b_data = b.HostRead();
   Real err2 = 0.0;
   Real ref2 = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real b_i = b_data[ i ];
      const Real diff = a[i] - b_i;
      err2 += diff * diff;
      ref2 += b_i * b_i;
   }
   const Real rel = std::sqrt( err2 ) /
      ( std::sqrt( ref2 ) + std::numeric_limits< Real >::epsilon() );
   std::cout << label << " relative L2 error = " << rel << "\n";
   ok = Check( rel < TestTolerance( Real(5.0e7) ), label ) && ok;
   return ok;
}

bool CheckVectorFar( const Vector & a, const Vector & b, const char * label )
{
   bool ok = Check( a.Size() == b.Size(), "vector size mismatch" );
   Real err2 = 0.0;
   Real ref2 = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real diff = a[i] - b[i];
      err2 += diff * diff;
      ref2 += b[i] * b[i];
   }
   const Real rel = std::sqrt( err2 ) /
      ( std::sqrt( ref2 ) + std::numeric_limits< Real >::epsilon() );
   std::cout << label << " relative L2 error = " << rel << "\n";
   ok = Check( rel > TestTolerance( Real(5.0e7) ), label ) && ok;
   return ok;
}

void FillNonsymmetricDofs( const auto & fe_space, Vector & x )
{
   auto * data = x.WriteHostData();
   for ( GlobalIndex element = 0; element < fe_space.GetNumberOfFiniteElements();
         ++element )
   {
      for ( GlobalIndex i = 0; i < 3; ++i )
      {
         for ( GlobalIndex j = 0; j < 3; ++j )
         {
            const std::array< GlobalIndex, 2 > dof{i, j};
            const auto gdof =
               GlobalDofIndex( fe_space, element, FlattenLocalDof( fe_space, dof ) );
            data[ gdof ] =
               Real(0.37) + Real(1.9) * element + Real(0.23) * i +
               Real(0.41) * j + Real(0.07) * i * j;
         }
      }
   }
}

template < typename Geometry, typename Partition >
bool CheckConformingGlobalPartitionShape( const Partition & partition )
{
   bool ok = true;
   ok = Check(
      partition.GetNumberOfCellParts() == 1,
      "file-backed global partition does not have exactly one cell part" ) && ok;
   ok = Check(
      partition.GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
      "file-backed global partition does not expose conforming and NC face families" ) && ok;
   ok = Check(
      partition.GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
      "file-backed global partition does not expose one boundary family per face" ) && ok;

   GlobalIndex conforming_faces = 0;
   GlobalIndex nonconforming_faces = 0;
   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto face )
      {
         constexpr size_t FaceIndex =
            static_cast< size_t >( decltype( face )::value );
         conforming_faces +=
            std::get< FaceIndex >( partition.InteriorFaceParts() )
               .face_mesh.GetNumberOfFaces();
         nonconforming_faces +=
            std::get< FaceIndex + Geometry::num_faces >(
               partition.InteriorFaceParts() )
                  .face_mesh.GetNumberOfFaces();
      } );

   GlobalIndex boundary_faces = 0;
   std::apply(
      [&] ( const auto & ... part )
      {
         ( ( boundary_faces += part.face_mesh.GetNumberOfFaces() ), ... );
      },
      partition.BoundaryFaceParts() );

   ok = Check(
      conforming_faces > 0,
      "file-backed global partition has no conforming interior records" ) && ok;
   ok = Check(
      nonconforming_faces == 0,
      "conforming file-backed global partition has nonempty NC records" ) && ok;
   ok = Check(
      boundary_faces > 0,
      "file-backed global partition has no boundary records" ) && ok;
   return ok;
}

mfem::Vector ApplyMFEMSIPDGReference(
   mfem::Mesh & mesh,
   Vector & x,
   Real sigma,
   Real kappa )
{
   constexpr int order = 2;
   constexpr int dim = 2;
   constexpr int num_quad_1d = 5;
   constexpr int mfem_rule_order = 2 * ( num_quad_1d - 1 );

   mfem::L2_FECollection fec(
      order,
      dim,
      mfem::BasisType::GaussLobatto );
   mfem::FiniteElementSpace fes( &mesh, &fec );
   GENDIL_VERIFY(
      static_cast< size_t >( fes.GetNDofs() ) == x.Size(),
      "MFEM SIPDG reference dof count does not match GenDiL dof count." );

   mfem::ConstantCoefficient diffusivity( 1.0 );
   const mfem::IntegrationRule & volume_ir =
      mfem::IntRules.Get( mfem::Geometry::SQUARE, mfem_rule_order );
   const mfem::IntegrationRule & face_ir =
      mfem::IntRules.Get( mfem::Geometry::SEGMENT, mfem_rule_order );

   mfem::BilinearForm blf( &fes );
   auto * volume_integrator =
      new mfem::DiffusionIntegrator( diffusivity );
   volume_integrator->SetIntegrationRule( volume_ir );
   blf.AddDomainIntegrator( volume_integrator );

   auto * interior_face_integrator =
      new mfem::DGDiffusionIntegrator( diffusivity, sigma, kappa );
   interior_face_integrator->SetIntegrationRule( face_ir );
   blf.AddInteriorFaceIntegrator( interior_face_integrator );

   auto * boundary_face_integrator =
      new mfem::DGDiffusionIntegrator( diffusivity, sigma, kappa );
   boundary_face_integrator->SetIntegrationRule( face_ir );
   blf.AddBdrFaceIntegrator( boundary_face_integrator );

   blf.Assemble();
   blf.Finalize();

   auto x_mfem = x.ToMFEMVector();
   mfem::Vector y_mfem( fes.GetNDofs() );
   blf.SpMat().Mult( x_mfem, y_mfem );
   return y_mfem;
}

[[maybe_unused]] bool TestQuadConnectivity()
{
   mfem::Mesh mesh = MakeTwoQuadMesh( 0, 1 );
   auto bundle =
      MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >( mesh );

   bool ok = Check(
      CountFaces( bundle.connectivity ) ==
         mesh.GetFaceIndices( mfem::FaceType::Interior ).Size(),
      "quad source/output interior face total mismatch" );
   ok = CheckInteriorMetadataAlignment( mesh, bundle ) && ok;

   auto snapshot = []
   {
      mfem::Mesh local_mesh = MakeTwoQuadMesh( 0, 1 );
      return MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >(
         local_mesh );
   }();
   ok = Check(
      CountFaces( snapshot.connectivity ) == 1,
      "quad connectivity snapshot did not survive source mesh lifetime" ) && ok;

   bool found_nonidentity = false;
   std::apply(
      [&] ( const auto & ... family )
      {
         ( [&]
           {
              for ( GlobalIndex i = 0; i < family.GetNumberOfFaces(); ++i )
              {
                 found_nonidentity =
                    found_nonidentity ||
                    family.records.host_pointer[i].plus_orientation !=
                       MakeReferencePermutation<2>();
              }
           }(), ... );
      },
      bundle.connectivity );
   ok = Check( found_nonidentity, "two-quad fixture did not produce a nonidentity plus orientation" ) && ok;

   const int source_face_id =
      std::get<2>( bundle.metadata )[0].source_face_id;
   const auto record =
      std::get<2>( bundle.connectivity ).records.host_pointer[0];
   ok = CheckCoordinateAgreement<2>(
           mesh,
           source_face_id,
           2,
           record.plus_orientation,
           {{{0.0}}, {{1.0}}, {{0.23}}} ) && ok;
   ok = CheckIdentityOrientationWouldDisagree<2>(
           mesh,
           source_face_id,
           2,
           {{{0.23}}} ) && ok;
   return ok;
}

[[maybe_unused]] bool TestHexConnectivity()
{
   mfem::Mesh mesh = MakeTwoHexMesh( 0, 1 );
   auto bundle =
      MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<3> >( mesh );

   const auto info = mesh.GetFaceInformation( 2 );
   bool ok = true;
   ok = Check( info.element[0].local_face_id == 2,
               "3D fixture side-0 local face mismatch" ) && ok;
   ok = Check( info.element[1].local_face_id == 3,
               "3D fixture side-1 local face mismatch" ) && ok;
   ok = Check( info.element[1].orientation == 3,
               "3D fixture MFEM orientation mismatch" ) && ok;

   const auto & family = std::get<3>( bundle.connectivity );
   const auto & metadata = std::get<3>( bundle.metadata );
   ok = Check( family.GetNumberOfFaces() == 1,
               "3D fixture expected one face in g0=3 family" ) && ok;
   ok = Check( metadata[0].source_face_id == 2,
               "3D fixture source face mismatch" ) && ok;

   const auto record = family.records.host_pointer[0];
   const Permutation<3> expected{ { 2, -1, 3 } };
   ok = Check( record.plus_orientation == expected,
               "3D fixture plus orientation mismatch" ) && ok;
   ok = Check(
      NativeFaceFromReferenceFace(
         HyperCube< 3 >::GetOppositeFaceIndex( 3 ),
         record.plus_orientation ) == 4,
      "3D fixture native plus face recovery mismatch" ) && ok;

   ok = CheckCoordinateAgreement<3>(
           mesh,
           2,
           3,
           record.plus_orientation,
           {{{0.0, 0.0}},
            {{1.0, 0.0}},
           {{0.0, 1.0}},
            {{1.0, 1.0}},
            {{0.23, 0.67}}} ) && ok;
   ok = CheckIdentityOrientationWouldDisagree<3>(
           mesh,
           2,
           3,
           {{{0.23, 0.67}}} ) && ok;
   return ok;
}

template < typename Geometry >
bool SmokeInteriorBoundary( mfem::Mesh & mesh )
{
   auto conforming =
      MakeMFEMConformingGlobalInteriorFaceConnectivity< Geometry >( mesh );
   auto boundary = MakeMFEMGlobalBoundaryFaceConnectivity< Geometry >(
      mesh,
      MFEMBoundaryMetadataOptions{ true } );
   bool ok = Check(
      CountFaces( conforming.connectivity ) ==
         mesh.GetFaceIndices( mfem::FaceType::Interior ).Size(),
      "smoke interior source/output count mismatch" );
   ok = Check(
      CountFaces( boundary.connectivity ) ==
         mesh.GetFaceIndices( mfem::FaceType::Boundary ).Size(),
      "smoke boundary source/output count mismatch" ) && ok;
   return ok;
}

[[maybe_unused]] bool TestSmoke()
{
   mfem::Mesh line = mfem::Mesh::MakeCartesian1D( 2, 1.0 );
   mfem::Mesh quad =
      mfem::Mesh::MakeCartesian2D(
         2,
         1,
         mfem::Element::QUADRILATERAL,
         false,
         2.0,
         1.0 );
   mfem::Mesh hex =
      mfem::Mesh::MakeCartesian3D(
         2,
         1,
         1,
         mfem::Element::HEXAHEDRON,
         2.0,
         1.0,
         1.0 );
   bool ok = SmokeInteriorBoundary< HyperCube<1> >( line );
   ok = SmokeInteriorBoundary< HyperCube<2> >( quad ) && ok;
   ok = SmokeInteriorBoundary< HyperCube<3> >( hex ) && ok;
   return ok;
}

[[maybe_unused]] bool TestGenericInteriorOperator()
{
   mfem::Mesh mfem_mesh = MakeTwoQuadMesh( 0, 1 );
   constexpr Integer mesh_order = 1;
   QuadMesh< mesh_order > mesh = MakeQuadMesh< mesh_order >( mfem_mesh );
   auto connectivity =
      MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >(
         mfem_mesh );

   FiniteElementOrders<2, 2> orders;
   auto fe = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );
   auto partition =
      MakePartition(
         MakeCellPart( mesh ),
         MakeInteriorFacePart< 0, 0 >( connectivity.connectivity ) );
   auto global_space =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe },
         DGDirectSumNumbering{} );

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto form = integrate( InteriorFacets<"mesh">{}, jump( u ) * jump( v ) );
   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">( global_space ),
      MakeIntegrationDomain<"mesh">( global_space ) );
   constexpr Integer num_quad_1d = 5;
   auto ir =
      MakeIntegrationRule(
         IntegrationRuleNumPoints< num_quad_1d, num_quad_1d >{} );
   auto op =
      MakeGenericOperator< GlobalFaceKernelPolicy< num_quad_1d > >(
         form,
         ctx,
         ir );

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillNonsymmetricDofs( fe_space, x );
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   op( x, y );
   const Vector y_ref = AssembleInteriorJumpReference(
      mfem_mesh,
      fe_space,
      connectivity,
      x );
   const Vector y_wrong_orientation =
      AssembleInteriorJumpReferenceIgnoringPlusOrientation(
         mfem_mesh,
         fe_space,
         connectivity,
         x );
   bool ok = CheckVectorNear( y, y_ref, "generic interior jump operator" );
   ok = CheckVectorFar(
           y_wrong_orientation,
           y_ref,
           "generic interior ignored-plus-orientation negative control" ) && ok;
   return ok;
}

[[maybe_unused]] bool TestFileBackedSIPDGLocalVsGlobal()
{
   constexpr Integer mesh_order = 1;
   constexpr Integer order = 2;
   constexpr Integer num_quad_1d = 5;
   using Geometry = HyperCube<2>;

   const std::string mesh_file =
      std::string( MFEM_DIR ) + "/data/fichera-quad.mesh";
   constexpr int generate_edges = 0;
   constexpr int refine = 0;
   mfem::Mesh mfem_mesh( mesh_file.c_str(), generate_edges, refine );
   mfem_mesh.SetCurvature( mesh_order );

   QuadMesh< mesh_order > local_mesh =
      MakeQuadMesh< mesh_order >( mfem_mesh );
   auto partition =
      MakeGlobalPartition< Geometry, mesh_order >( mfem_mesh );
   bool ok = CheckConformingGlobalPartitionShape< Geometry >( partition );

   FiniteElementOrders< order, order > orders;
   auto fe = MakeLobattoFiniteElement( orders );
   auto local_space = MakeFiniteElementSpace( local_mesh, fe );
   const auto conforming_partition =
      MakePartition(
         std::get<0>( partition.CellParts() ),
         std::get<0>( partition.InteriorFaceParts() ),
         std::get<1>( partition.InteriorFaceParts() ),
         std::get<2>( partition.InteriorFaceParts() ),
         std::get<3>( partition.InteriorFaceParts() ),
         std::get<0>( partition.BoundaryFaceParts() ),
         std::get<1>( partition.BoundaryFaceParts() ),
         std::get<2>( partition.BoundaryFaceParts() ),
         std::get<3>( partition.BoundaryFaceParts() ) );
   auto global_space =
      MakeMixedFiniteElementSpace(
         conforming_partition,
         std::tuple{ fe },
         std::tuple{ L2Restriction{ 0 } } );

   ok = Check(
      local_space.GetNumberOfFiniteElementDofs() ==
         global_space.GetNumberOfFiniteElementDofs(),
      "local/global file-backed spaces have different dof counts" ) && ok;

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;
   BoundaryFacets<"mesh"> boundary_facets;

   constexpr Real sigma = Real(-1.0);
   constexpr Real kappa = Real(( order + 1 ) * ( order + 1 ));
   auto diffusivity =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, 2 > & ) -> Real
      {
         return Real(1.0);
      };
   auto mu =
      MakeCoefficient<"diffusivity", PhysicalCoordinate>( diffusivity );
   auto tau =
      MakeCoefficient<"penalty", InverseFacetSize>(
         [=] GENDIL_HOST_DEVICE ( const Real & h_inv ) -> Real
         {
            return kappa * h_inv;
         } );

   auto diffusion_form =
      integrate( cells, mu * dot( grad( u ), grad( v ) ) )
      + integrate(
           interior_facets,
           -average( mu * dot( grad( u ), Normal{} ) ) * jump( v )
           + sigma * jump( u ) *
                average( mu * dot( grad( v ), Normal{} ) )
           + tau * mu * jump( u ) * jump( v ) )
      + integrate(
           boundary_facets,
           -mu * dot( grad( u ), Normal{} ) * v
           + sigma * u * mu * dot( grad( v ), Normal{} )
           + tau * mu * u * v );

   auto local_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">( local_space ),
         MakeIntegrationDomain<"mesh">( local_space ) );
   auto global_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">( global_space ),
         MakeIntegrationDomain<"mesh">( global_space ) );
   auto ir =
      MakeIntegrationRule(
         IntegrationRuleNumPoints< num_quad_1d, num_quad_1d >{} );

#if defined(GENDIL_USE_DEVICE)
   using LocalKernelPolicy =
      ThreadFirstKernelConfiguration<
         ThreadBlockLayout<num_quad_1d, num_quad_1d>,
         2>;
#else
   using LocalKernelPolicy = SerialKernelConfiguration;
#endif

   auto local_op =
      MakeGenericOperator< LocalKernelPolicy >(
         diffusion_form,
         local_ctx,
         ir );
   auto global_op =
      MakeGenericOperator< GlobalFaceKernelPolicy< num_quad_1d > >(
         diffusion_form,
         global_ctx,
         ir );

   Vector x( local_space.GetNumberOfFiniteElementDofs() );
   FillNonsymmetricDofs( local_space, x );
   Vector y_local( local_space.GetNumberOfFiniteElementDofs() );
   Vector y_global( local_space.GetNumberOfFiniteElementDofs() );
   local_op( x, y_local );
   global_op( x, y_global );

   mfem::Vector y_mfem =
      ApplyMFEMSIPDGReference( mfem_mesh, x, sigma, kappa );

   ok = CheckVectorNearMFEM(
           y_local,
           y_mfem,
           "file-backed local SIPDG vs MFEM" ) && ok;
   ok = CheckVectorNearMFEM(
           y_global,
           y_mfem,
           "file-backed global SIPDG vs MFEM" ) && ok;
   ok = CheckVectorNear(
           y_local,
           y_global,
           "file-backed local SIPDG vs global partition SIPDG" ) && ok;
   return ok;
}

[[maybe_unused]] bool TestBoundaryConnectivityAndExecution()
{
   mfem::Mesh mfem_mesh =
      mfem::Mesh::MakeCartesian2D(
         2,
         1,
         mfem::Element::QUADRILATERAL,
         false,
         2.0,
         1.0 );

   for ( int be = 0; be < mfem_mesh.GetNBE(); ++be )
   {
      const int face_id = mfem_mesh.GetBdrElementFaceIndex( be );
      const auto info = mfem_mesh.GetFaceInformation( face_id );
      mfem_mesh.SetBdrAttribute( be, info.element[0].index == 1 ? 17 : 3 );
   }

   auto full = MakeMFEMGlobalBoundaryFaceConnectivity< HyperCube<2> >(
      mfem_mesh,
      MFEMBoundaryMetadataOptions{ true } );
   auto full_without_boundary_ids =
      MakeMFEMGlobalBoundaryFaceConnectivity< HyperCube<2> >( mfem_mesh );
   bool ok = Check(
      !full_without_boundary_ids.boundary_element_ids_requested,
      "boundary element ids unexpectedly requested" );
   std::apply(
      [&] ( const auto & ... metadata_family )
      {
         ( [&]
           {
              for ( const auto & meta : metadata_family )
              {
                 ok = Check(
                         meta.boundary_element_id ==
                            MFEMInvalidBoundaryElementId,
                         "unrequested boundary element id is not the invalid sentinel" ) && ok;
              }
           }(), ... );
      },
      full_without_boundary_ids.metadata );

   const mfem::Array< int > face_to_boundary_element =
      mfem_mesh.GetFaceToBdrElMap();
   auto filtered =
      FilterMFEMBoundaryFaceConnectivityByAttributes< HyperCube<2> >(
         full,
         {17, 17} );
   ok = Check(
      CountFaces( filtered.connectivity ) > 0,
      "filtered boundary bundle is empty" ) && ok;

   std::apply(
      [&] ( const auto & ... metadata_family )
      {
         ( [&]
           {
              for ( const auto & meta : metadata_family )
              {
                 const auto info =
                    mfem_mesh.GetFaceInformation( meta.source_face_id );
                 ok = Check( meta.boundary_attribute == 17,
                             "filtered bundle contains unselected attribute" ) && ok;
                 ok = Check( info.element[0].index == 1,
                             "filtered bundle contains cell zero face" ) && ok;
                 ok = Check(
                         meta.boundary_element_id >= 0,
                         "requested boundary element id missing" ) && ok;
                 const bool source_face_in_range =
                    meta.source_face_id >= 0 &&
                    meta.source_face_id < face_to_boundary_element.Size();
                 ok = Check( source_face_in_range,
                             "metadata source face is outside face-to-boundary-element map" ) && ok;
                 if ( source_face_in_range )
                 {
                    const int expected_boundary_element_id =
                       face_to_boundary_element[ meta.source_face_id ];
                    ok = Check( expected_boundary_element_id >= 0,
                                "selected boundary face has no boundary element mapping" ) && ok;
                    if ( meta.boundary_element_id >= 0 )
                    {
                       ok = Check(
                               meta.boundary_element_id ==
                                  expected_boundary_element_id,
                               "boundary element id does not match GetFaceToBdrElMap" ) && ok;
                    }
                 }
              }
           }(), ... );
      },
      filtered.metadata );

   constexpr Integer mesh_order = 1;
   QuadMesh< mesh_order > mesh = MakeQuadMesh< mesh_order >( mfem_mesh );
   FiniteElementOrders<2, 2> orders;
   auto fe = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );
   auto partition =
      MakePartition(
         MakeCellPart( mesh ),
         MakeBoundaryFacePart< 0 >( filtered.connectivity ) );
   auto global_space =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe },
         DGDirectSumNumbering{} );

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto form = integrate( BoundaryFacets<"mesh">{}, u * v );
   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">( global_space ),
      MakeIntegrationDomain<"mesh">( global_space ) );
   constexpr Integer num_quad_1d = 5;
   auto ir =
      MakeIntegrationRule(
         IntegrationRuleNumPoints< num_quad_1d, num_quad_1d >{} );
   auto op =
      MakeGenericOperator< GlobalFaceKernelPolicy< num_quad_1d > >(
         form,
         ctx,
         ir );

   Vector x1( fe_space.GetNumberOfFiniteElementDofs() );
   FillNonsymmetricDofs( fe_space, x1 );
   Vector x2( fe_space.GetNumberOfFiniteElementDofs() );
   FillNonsymmetricDofs( fe_space, x2 );
   auto * x2_data = x2.WriteHostData();
   for ( GlobalIndex i = 0; i < 3; ++i )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         const std::array< GlobalIndex, 2 > dof{i, j};
         x2_data[ GlobalDofIndex(
            fe_space,
            0,
            FlattenLocalDof( fe_space, dof ) ) ] += Real(42.0);
      }
   }

   Vector y1( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y2( fe_space.GetNumberOfFiniteElementDofs() );
   op( x1, y1 );
   op( x2, y2 );
   const Vector y_ref = AssembleBoundaryMassReference(
      mfem_mesh,
      fe_space,
      filtered,
      x1 );

   ok = CheckVectorNear( y1, y_ref, "filtered boundary mass operator" ) && ok;
   ok = CheckVectorNear( y2, y1, "boundary dummy-cell poisoning" ) && ok;
   for ( GlobalIndex i = 0; i < 3; ++i )
   {
      for ( GlobalIndex j = 0; j < 3; ++j )
      {
         const std::array< GlobalIndex, 2 > dof{i, j};
         ok = CheckNear(
                 y1[ GlobalDofIndex(
                    fe_space,
                    0,
                    FlattenLocalDof( fe_space, dof ) ) ],
                 0.0,
                 "filtered boundary wrote into cell zero" ) && ok;
      }
   }
   return ok;
}

[[maybe_unused]] bool TestPeriodic()
{
   auto non_periodic =
      mfem::Mesh::MakeCartesian2D(
         2,
         1,
         mfem::Element::QUADRILATERAL,
         false,
         1.0,
         1.0,
         false );
   mfem::Vector x_translation({1.0, 0.0});
   std::vector< mfem::Vector > translations{ x_translation };
   mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      non_periodic,
      non_periodic.CreatePeriodicVertexMapping( translations ) );

   auto bundle =
      MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >( periodic );
   bool ok = Check(
      CountFaces( bundle.connectivity ) ==
         periodic.GetFaceIndices( mfem::FaceType::Interior ).Size(),
      "periodic source/output face count mismatch" );

   std::map< std::pair< GlobalIndex, GlobalIndex >, int > pair_counts;
   std::apply(
      [&] ( const auto & ... family )
      {
         ( [&]
           {
              for ( GlobalIndex i = 0; i < family.GetNumberOfFaces(); ++i )
              {
                 const auto r = family.records.host_pointer[ i ];
                 const auto p = std::minmax( r.minus_cell, r.plus_cell );
                 ++pair_counts[ p ];
              }
           }(), ... );
      },
      bundle.connectivity );

   bool found_duplicate_pair = false;
   for ( const auto & [pair, count] : pair_counts )
   {
      if ( count > 1 )
      {
         found_duplicate_pair = true;
      }
   }
   ok = Check(
      found_duplicate_pair,
      "periodic fixture did not preserve multiple faces for one cell pair" ) && ok;
   return ok;
}

[[maybe_unused]] void RunAdaptiveDiagnostic()
{
   mfem::Mesh mesh =
      mfem::Mesh::MakeCartesian2D(
         2,
         1,
         mfem::Element::QUADRILATERAL,
         false,
         2.0,
         1.0 );
   mesh.EnsureNCMesh( true );
   mfem::Array< int > refine;
   refine.Append( 0 );
   mesh.GeneralRefinement( refine, 1, 0 );
   (void)MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >( mesh );
}

[[maybe_unused]] void RunAdaptiveCellMeshDiagnostic()
{
   mfem::Mesh mesh =
      mfem::Mesh::MakeCartesian2D(
         2,
         1,
         mfem::Element::QUADRILATERAL,
         false,
         2.0,
         1.0 );
   mesh.SetCurvature( 1 );
   mesh.EnsureNCMesh( true );
   mfem::Array< int > refine;
   refine.Append( 0 );
   mesh.GeneralRefinement( refine, 1, 0 );
   (void)MakeQuadMesh< 1 >( mesh );
}

[[maybe_unused]] void RunSimplexDiagnostic()
{
   mfem::Mesh mesh =
      mfem::Mesh::MakeCartesian2D(
         1,
         1,
         mfem::Element::TRIANGLE,
         false,
         1.0,
         1.0 );
   (void)MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<2> >( mesh );
}

[[maybe_unused]] void RunSelfNeighborDiagnostic()
{
   mfem::Mesh non_periodic = mfem::Mesh::MakeCartesian1D( 1, 1.0 );
   mfem::Vector translation({1.0});
   std::vector< mfem::Vector > translations{ translation };
   mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      non_periodic,
      non_periodic.CreatePeriodicVertexMapping( translations ) );
   (void)MakeMFEMConformingGlobalInteriorFaceConnectivity< HyperCube<1> >(
      periodic );
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
   mfem::Device device(device_config);
   device.Print();

#if defined(TEST_QUAD_CONNECTIVITY)
   return TestQuadConnectivity() ? 0 : 1;
#elif defined(TEST_HEX_CONNECTIVITY)
   return TestHexConnectivity() ? 0 : 1;
#elif defined(TEST_GENERIC_OPERATOR)
   return TestGenericInteriorOperator() ? 0 : 1;
#elif defined(TEST_FILE_SIPDG_LOCAL_VS_GLOBAL)
   return TestFileBackedSIPDGLocalVsGlobal() ? 0 : 1;
#elif defined(TEST_BOUNDARY)
   return TestBoundaryConnectivityAndExecution() ? 0 : 1;
#elif defined(TEST_SMOKE)
   return TestSmoke() ? 0 : 1;
#elif defined(TEST_PERIODIC)
   return TestPeriodic() ? 0 : 1;
#elif defined(TEST_EXPECT_ADAPTIVE)
   RunAdaptiveDiagnostic();
   return 0;
#elif defined(TEST_EXPECT_ADAPTIVE_CELL_MESH)
   RunAdaptiveCellMeshDiagnostic();
   return 0;
#elif defined(TEST_EXPECT_SIMPLEX)
   RunSimplexDiagnostic();
   return 0;
#elif defined(TEST_EXPECT_SELF_NEIGHBOR)
   RunSelfNeighborDiagnostic();
   return 0;
#else
   bool ok = TestQuadConnectivity();
   ok = TestHexConnectivity() && ok;
   ok = TestGenericInteriorOperator() && ok;
   ok = TestFileBackedSIPDGLocalVsGlobal() && ok;
   ok = TestBoundaryConnectivityAndExecution() && ok;
   ok = TestSmoke() && ok;
   ok = TestPeriodic() && ok;
   return ok ? 0 : 1;
#endif
}
