// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include "../gendil/mfem_point_matrix_decoder_test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace gendil;
using namespace gendil::tests::mfem_nc_preflight;

namespace
{

Real max_loc1_error = 0.0;
Real max_loc2_error = 0.0;
int mixed_conforming_count = 0;
int mixed_leaf_count = 0;
int mixed_source_count = 0;
GlobalIndex mixed_built_conforming_count = 0;
GlobalIndex mixed_built_leaf_count = 0;

bool Check( bool condition, const std::string & message )
{
   if ( !condition )
   {
      std::cerr << "FAILED: " << message << "\n";
   }
   return condition;
}

template < Integer Dim >
bool CheckPoint(
   const Point< Dim > & got,
   const Point< Dim > & expected,
   const std::string & message )
{
   if ( MaxError( got, expected ) > DecoderTolerance() )
   {
      std::cerr << "FAILED: " << message
                << " got " << PointString( got )
                << " expected " << PointString( expected ) << "\n";
      return false;
   }
   return true;
}

template < Integer Dim >
bool CheckSize(
   const NonconformingHyperCubeFaceMap< Dim > & map,
   const std::array< Real, Dim > & expected,
   const std::string & message )
{
   bool ok = true;
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( std::abs( map.size[d] - expected[d] ) > DecoderTolerance() )
      {
         std::cerr << "FAILED: " << message << " axis " << d
                   << " got " << map.size[d]
                   << " expected " << expected[d] << "\n";
         ok = false;
      }
   }
   return ok;
}

template < typename ConnectivityTuple >
GlobalIndex CountConnectivityRecords( const ConnectivityTuple & tuple )
{
   GlobalIndex count = 0;
   std::apply(
      [&] ( const auto &... family )
      {
         ( ( count += family.GetNumberOfFaces() ), ... );
      },
      tuple );
   return count;
}

template < typename PartitionType >
GlobalIndex CountPartitionInteriorFaceRecords( const PartitionType & partition )
{
   GlobalIndex count = 0;
   std::apply(
      [&] ( const auto &... part )
      {
         ( ( count += part.face_mesh.GetNumberOfFaces() ), ... );
      },
      partition.InteriorFaceParts() );
   return count;
}

template < typename PartitionType >
GlobalIndex CountPartitionBoundaryFaceRecords( const PartitionType & partition )
{
   GlobalIndex count = 0;
   std::apply(
      [&] ( const auto &... part )
      {
         ( ( count += part.face_mesh.GetNumberOfFaces() ), ... );
      },
      partition.BoundaryFaceParts() );
   return count;
}

template < typename Mesh, typename = void >
struct HasLocalFaceInfo : std::false_type {};

template < typename Mesh >
struct HasLocalFaceInfo<
   Mesh,
   std::void_t<
      decltype(
         std::declval< const Mesh & >().GetLocalFaceInfo(
            GlobalIndex{},
            std::integral_constant< Integer, 0 >{} ) ) > >
   : std::true_type {};

template < typename MetadataTuple >
std::size_t CountMetadataRecords( const MetadataTuple & tuple )
{
   std::size_t count = 0;
   std::apply(
      [&] ( const auto &... family )
      {
         ( ( count += family.size() ), ... );
      },
      tuple );
   return count;
}

template < typename MetadataTuple >
std::vector< int > CollectSourceFaceIds( const MetadataTuple & tuple )
{
   std::vector< int > ids;
   std::apply(
      [&] ( const auto &... family )
      {
         ( ( [&]
            {
               for ( const auto & metadata : family )
               {
                  ids.push_back( metadata.source_face_id );
               }
            }() ), ... );
      },
      tuple );
   return ids;
}

template < typename MetadataTuple >
bool CheckFamilyLocalSourceOrder( const MetadataTuple & tuple )
{
   bool ok = true;
   std::apply(
      [&] ( const auto &... family )
      {
         ( ( [&]
            {
               int last = -1;
               for ( const auto & metadata : family )
               {
                  ok = ok && metadata.source_face_id > last;
                  last = metadata.source_face_id;
               }
            }() ), ... );
      },
      tuple );
   return ok;
}

template < typename Geometry, typename MetadataTuple >
std::map< Integer, std::vector< int > > SourceOrderByFamily(
   const MetadataTuple & tuple )
{
   std::map< Integer, std::vector< int > > order;
   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto family )
      {
         constexpr Integer Family =
            static_cast< Integer >( decltype( family )::value );
         std::vector< int > family_order;
         for ( const auto & metadata : std::get<Family>( tuple ) )
         {
            family_order.push_back( metadata.source_face_id );
         }
         if ( !family_order.empty() )
         {
            order[Family] = family_order;
         }
      } );
   return order;
}

std::array< int, 4 > RotateQuad( const std::array< int, 4 > & q, int rot )
{
   std::array< int, 4 > out{};
   for ( int i = 0; i < 4; ++i )
   {
      out[i] = q[( i + rot ) % 4];
   }
   return out;
}

mfem::Mesh MakeTwoQuadMesh( int coarse_rotation, int fine_rotation, bool refine_left )
{
   mfem::Mesh mesh( 2, 6, 2 );
   double x[2];
   const double points[6][2] = {
      { 0, 0 }, { 1, 0 }, { 2, 0 },
      { 0, 1 }, { 1, 1 }, { 2, 1 }
   };
   for ( const auto & p : points )
   {
      x[0] = p[0];
      x[1] = p[1];
      mesh.AddVertex( x );
   }

   const auto left = RotateQuad( { 0, 1, 4, 3 }, refine_left ? fine_rotation : coarse_rotation );
   const auto right = RotateQuad( { 1, 2, 5, 4 }, refine_left ? coarse_rotation : fine_rotation );
   int elem0[4];
   int elem1[4];
   for ( int i = 0; i < 4; ++i )
   {
      elem0[i] = left[i];
      elem1[i] = right[i];
   }
   mesh.AddQuad( elem0 );
   mesh.AddQuad( elem1 );
   mesh.FinalizeQuadMesh( true );

   mfem::Array< mfem::Refinement > refs;
   refs.Append( mfem::Refinement( refine_left ? 0 : 1 ) );
   mesh.GeneralRefinement( refs, 1, 0 );
   return mesh;
}

int HexVertexIndex( int x, int y, int z )
{
   if ( z == 0 )
   {
      if ( x == 0 && y == 0 ) { return 0; }
      if ( x == 1 && y == 0 ) { return 1; }
      if ( x == 1 && y == 1 ) { return 2; }
      return 3;
   }
   if ( x == 0 && y == 0 ) { return 4; }
   if ( x == 1 && y == 0 ) { return 5; }
   if ( x == 1 && y == 1 ) { return 6; }
   return 7;
}

std::array< int, 8 > CubePermutationFromSignedAxes( const Permutation< 3 > & p )
{
   std::array< int, 8 > out{};
   for ( int z = 0; z <= 1; ++z )
   {
      for ( int y = 0; y <= 1; ++y )
      {
         for ( int x = 0; x <= 1; ++x )
         {
            const int src[3] = { x, y, z };
            int dst[3]{};
            for ( int axis = 0; axis < 3; ++axis )
            {
               const int entry = p( axis );
               const int source_axis = std::abs( entry ) - 1;
               dst[axis] = entry > 0 ? src[source_axis] : 1 - src[source_axis];
            }
            out[ HexVertexIndex( x, y, z ) ] =
               HexVertexIndex( dst[0], dst[1], dst[2] );
         }
      }
   }
   return out;
}

mfem::Mesh MakeTwoHexMesh(
   const Permutation< 3 > & coarse_rotation,
   const Permutation< 3 > & fine_rotation,
   int refinement_axis,
   bool refine_low_side )
{
   mfem::Mesh mesh( 3, 12, 2 );
   double x[3];
   std::vector< std::array< double, 3 > > points;
   if ( refinement_axis == 0 )
   {
      points = {
         {0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},
         {0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1}
      };
   }
   else if ( refinement_axis == 1 )
   {
      points = {
         {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,2,0},{1,2,0},
         {0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1}
      };
   }
   else
   {
      points = {
         {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},
         {0,1,1},{1,1,1},{0,0,2},{1,0,2},{0,1,2},{1,1,2}
      };
   }
   for ( const auto & p : points )
   {
      x[0] = p[0];
      x[1] = p[1];
      x[2] = p[2];
      mesh.AddVertex( x );
   }

   std::array< int, 8 > low{};
   std::array< int, 8 > high{};
   if ( refinement_axis == 0 )
   {
      low = { 0, 1, 4, 3, 6, 7, 10, 9 };
      high = { 1, 2, 5, 4, 7, 8, 11, 10 };
   }
   else if ( refinement_axis == 1 )
   {
      low = { 0, 1, 3, 2, 6, 7, 9, 8 };
      high = { 2, 3, 5, 4, 8, 9, 11, 10 };
   }
   else
   {
      low = { 0, 1, 3, 2, 4, 5, 7, 6 };
      high = { 4, 5, 7, 6, 8, 9, 11, 10 };
   }

   const auto coarse_perm = CubePermutationFromSignedAxes( coarse_rotation );
   const auto fine_perm = CubePermutationFromSignedAxes( fine_rotation );
   auto orient = [] ( const std::array< int, 8 > & base,
                      const std::array< int, 8 > & perm )
   {
      std::array< int, 8 > out{};
      for ( int i = 0; i < 8; ++i ) { out[i] = base[perm[i]]; }
      return out;
   };

   const auto coarse = orient( refine_low_side ? high : low, coarse_perm );
   const auto fine = orient( refine_low_side ? low : high, fine_perm );
   const auto low_elem = refine_low_side ? fine : coarse;
   const auto high_elem = refine_low_side ? coarse : fine;
   int elem0[8];
   int elem1[8];
   for ( int i = 0; i < 8; ++i )
   {
      elem0[i] = low_elem[i];
      elem1[i] = high_elem[i];
   }
   mesh.AddHex( elem0 );
   mesh.AddHex( elem1 );
   mesh.FinalizeHexMesh( true );

   mfem::Array< mfem::Refinement > refs;
   refs.Append( mfem::Refinement( refine_low_side ? 0 : 1, mfem::Refinement::XYZ ) );
   mesh.GeneralRefinement( refs, 1, 0 );
   return mesh;
}

bool IsLeaf( const mfem::Mesh::FaceInformation & info )
{
   return info.topology == mfem::Mesh::FaceTopology::Nonconforming &&
          info.tag == mfem::Mesh::FaceInfoTag::LocalSlaveNonconforming;
}

SegmentPointMatrix CopyPointMatrix2D( const mfem::DenseMatrix & matrix )
{
   return SegmentPointMatrix{ { matrix( 0, 0 ), matrix( 0, 1 ) } };
}

SquarePointMatrix CopyPointMatrix3D( const mfem::DenseMatrix & matrix )
{
   return SquarePointMatrix{ {
      Point< 2 >{ matrix( 0, 0 ), matrix( 1, 0 ) },
      Point< 2 >{ matrix( 0, 1 ), matrix( 1, 1 ) },
      Point< 2 >{ matrix( 0, 2 ), matrix( 1, 2 ) },
      Point< 2 >{ matrix( 0, 3 ), matrix( 1, 3 ) }
   } };
}

PointMatrixDecodeResult< 2 > Decode2(
   int coarse_lf,
   int fine_lf,
   const std::array< Real, 2 > & pm )
{
   return DecodeMFEMPointMatrix(
      coarse_lf,
      fine_lf,
      TranslateMFEMFace2D( coarse_lf ),
      TranslateMFEMFace2D( fine_lf ),
      SegmentPointMatrix{ pm } );
}

bool ValidateDecoded2D(
   mfem::Mesh & mesh,
   int face_id,
   const mfem::Mesh::FaceInformation & info,
   const PointMatrixDecodeResult< 2 > & decoded,
   SegmentPointMatrix pm )
{
   auto * ft = mesh.GetFaceElementTransformations( face_id );
   const Integer coarse_g = TranslateMFEMFace2D( info.element[1].local_face_id );
   const Integer static_plus = HyperCube< 2 >::GetOppositeFaceIndex( coarse_g );
   bool ok = true;
   for ( Real t : { Real( 0 ), Real( 1 ), Real( 0.37 ) } )
   {
      const std::array< Real, 1 > tangent{ t };
      const auto q_plus = CanonicalFacePoint< 2 >( static_plus, tangent );
      const auto fine_native =
         ApplyOrientation( q_plus, decoded.value.plus_orientation );
      const Real fine_t =
         NativeToMFEMFacePoint2D( info.element[0].local_face_id, fine_native );
      const Real coarse_t =
         ( Real( 1 ) - fine_t ) * pm.columns[1] +
         fine_t * pm.columns[0];
      const auto coarse_native =
         MFEMFacePointToNative2D( info.element[1].local_face_id, coarse_t );
      mfem::IntegrationPoint ip;
      ip.x = fine_t;
      mfem::IntegrationPoint l1, l2;
      ft->Loc1.Transform( ip, l1 );
      ft->Loc2.Transform( ip, l2 );
      const Real loc1_error =
         std::max( std::abs( l1.x - fine_native[0] ),
                   std::abs( l1.y - fine_native[1] ) );
      const Real loc2_error =
         std::max( std::abs( l2.x - coarse_native[0] ),
                   std::abs( l2.y - coarse_native[1] ) );
      max_loc1_error = std::max( max_loc1_error, loc1_error );
      max_loc2_error = std::max( max_loc2_error, loc2_error );
      ok = Check(
         loc1_error <= DecoderTolerance(),
         "2D Loc1 agrees with selected plus orientation" ) && ok;
      ok = Check(
         loc2_error <= DecoderTolerance(),
         "2D Loc2 agrees with decoded point matrix map" ) && ok;
   }
   return ok;
}

bool ValidateDecoded3D(
   mfem::Mesh & mesh,
   int face_id,
   const mfem::Mesh::FaceInformation & info,
   const PointMatrixDecodeResult< 3 > & decoded,
   SquarePointMatrix pm )
{
   auto * ft = mesh.GetFaceElementTransformations( face_id );
   const Integer coarse_g = TranslateMFEMFace3D( info.element[1].local_face_id );
   const Integer static_plus = HyperCube< 3 >::GetOppositeFaceIndex( coarse_g );
   bool ok = true;
   const std::array< std::array< Real, 2 >, 5 > points{
      std::array< Real, 2 >{ 0, 0 },
      std::array< Real, 2 >{ 1, 0 },
      std::array< Real, 2 >{ 0, 1 },
      std::array< Real, 2 >{ 1, 1 },
      std::array< Real, 2 >{ Real( 0.23 ), Real( 0.67 ) }
   };
   for ( const auto & tangent : points )
   {
      const auto q_plus = CanonicalFacePoint< 3 >( static_plus, tangent );
      const auto fine_native =
         ApplyOrientation( q_plus, decoded.value.plus_orientation );
      const auto uv = NativeToMFEMFacePoint3D( info.element[0].local_face_id, fine_native );
      const Real u = uv[0];
      const Real v = uv[1];
      const Real su =
         ( Real( 1 ) - u ) * ( Real( 1 ) - v ) * pm.columns[0][0] +
         u * ( Real( 1 ) - v ) * pm.columns[1][0] +
         u * v * pm.columns[2][0] +
         ( Real( 1 ) - u ) * v * pm.columns[3][0];
      const Real sv =
         ( Real( 1 ) - u ) * ( Real( 1 ) - v ) * pm.columns[0][1] +
         u * ( Real( 1 ) - v ) * pm.columns[1][1] +
         u * v * pm.columns[2][1] +
         ( Real( 1 ) - u ) * v * pm.columns[3][1];
      const auto coarse_native =
         MFEMFacePointToNative3D(
            info.element[1].local_face_id,
            Point< 2 >{ su, sv } );
      mfem::IntegrationPoint ip;
      ip.x = u;
      ip.y = v;
      mfem::IntegrationPoint l1, l2;
      ft->Loc1.Transform( ip, l1 );
      ft->Loc2.Transform( ip, l2 );
      const Real loc1_error =
         std::max( { std::abs( l1.x - fine_native[0] ),
                     std::abs( l1.y - fine_native[1] ),
                     std::abs( l1.z - fine_native[2] ) } );
      const Real loc2_error =
         std::max( { std::abs( l2.x - coarse_native[0] ),
                     std::abs( l2.y - coarse_native[1] ),
                     std::abs( l2.z - coarse_native[2] ) } );
      max_loc1_error = std::max( max_loc1_error, loc1_error );
      max_loc2_error = std::max( max_loc2_error, loc2_error );
      ok = Check(
         loc1_error <= DecoderTolerance(),
         "3D Loc1 agrees with selected plus orientation" ) && ok;
      ok = Check(
         loc2_error <= DecoderTolerance(),
         "3D Loc2 agrees with decoded point matrix map" ) && ok;
   }
   return ok;
}

bool TestDeterministic2DNonidentity()
{
   mfem::Mesh mesh = MakeTwoQuadMesh( 0, 1, false );
   const auto & interior = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior.Size(); ++i )
   {
      const int face_id = interior[i];
      const auto info = mesh.GetFaceInformation( face_id );
      if ( !IsLeaf( info ) ) { continue; }
      if ( face_id != 6 ) { continue; }
      const auto pm = CopyPointMatrix2D( *info.point_matrix );
      const auto decoded = DecodeMFEMPointMatrix(
         info.element[1].local_face_id,
         info.element[0].local_face_id,
         TranslateMFEMFace2D( info.element[1].local_face_id ),
         TranslateMFEMFace2D( info.element[0].local_face_id ),
         pm );
      bool ok = true;
      ok = Check( IsSuccessfulDecode( decoded ), "2D deterministic decode succeeded" ) && ok;
      ok = Check( info.element[1].local_face_id == 1, "2D coarse local face" ) && ok;
      ok = Check( info.element[0].local_face_id == 2, "2D fine local face" ) && ok;
      ok = Check( decoded.value.plus_orientation == MakePermutation< 2 >( { 2, -1 } ),
                  "2D nonidentity selected orientation" ) && ok;
      ok = CheckPoint( decoded.value.minus_map.origin,
                       Point< 2 >{ 0.0, 0.0 },
                       "2D decoded origin" ) && ok;
      ok = CheckSize< 2 >( decoded.value.minus_map,
                           { 1.0, 0.5 },
                           "2D decoded size" ) && ok;
      ok = ValidateDecoded2D( mesh, face_id, info, decoded, pm ) && ok;
      return ok;
   }
   return Check( false, "2D deterministic leaf face was not found" );
}

bool TestDeterministic3DNonidentity()
{
   mfem::Mesh mesh = MakeTwoHexMesh(
      MakePermutation< 3 >( { -1, -2, 3 } ),
      MakePermutation< 3 >( { -1, 2, -3 } ),
      0,
      false );
   const auto & interior = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior.Size(); ++i )
   {
      const int face_id = interior[i];
      const auto info = mesh.GetFaceInformation( face_id );
      if ( !IsLeaf( info ) ) { continue; }
      if ( face_id != 8 ) { continue; }
      const auto pm = CopyPointMatrix3D( *info.point_matrix );
      const auto decoded = DecodeMFEMPointMatrix(
         info.element[1].local_face_id,
         info.element[0].local_face_id,
         TranslateMFEMFace3D( info.element[1].local_face_id ),
         TranslateMFEMFace3D( info.element[0].local_face_id ),
         pm );
      bool ok = true;
      ok = Check( IsSuccessfulDecode( decoded ), "3D deterministic decode succeeded" ) && ok;
      ok = Check( info.element[1].local_face_id == 4, "3D coarse local face" ) && ok;
      ok = Check( info.element[0].local_face_id == 2, "3D fine local face" ) && ok;
      ok = Check( decoded.value.plus_orientation == MakePermutation< 3 >( { 1, -2, -3 } ),
                  "3D nonidentity selected orientation" ) && ok;
      ok = CheckPoint( decoded.value.minus_map.origin,
                       Point< 3 >{ 0.0, 0.0, 0.5 },
                       "3D decoded origin" ) && ok;
      ok = CheckSize< 3 >( decoded.value.minus_map,
                           { 1.0, 0.5, 0.5 },
                           "3D decoded size" ) && ok;
      ok = ValidateDecoded3D( mesh, face_id, info, decoded, pm ) && ok;
      return ok;
   }
   return Check( false, "3D deterministic leaf face was not found" );
}

bool TestMFEMNonconformingBuilder2D()
{
   using Geometry = HyperCube< 2 >;

   mfem::Mesh mesh = MakeTwoQuadMesh( 0, 1, false );
   const auto bundle =
      MakeMFEMNonconformingGlobalInteriorFaceConnectivity< Geometry >( mesh );

   constexpr Integer family = 2;
   const auto & connectivity = std::get< family >( bundle.connectivity );
   const auto & metadata = std::get< family >( bundle.metadata );

   bool ok = true;
   ok = Check(
      static_cast< std::size_t >( connectivity.GetNumberOfFaces() ) == metadata.size(),
      "2D NC builder metadata aligns with family records" ) && ok;

   bool found = false;
   const auto & interior = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior.Size(); ++i )
   {
      const int face_id = interior[i];
      if ( face_id != 6 ) { continue; }
      const auto info = mesh.GetFaceInformation( face_id );
      if ( !IsLeaf( info ) ) { continue; }

      for ( GlobalIndex j = 0; j < connectivity.GetNumberOfFaces(); ++j )
      {
         if ( metadata[ static_cast< std::size_t >( j ) ].source_face_id != face_id )
         {
            continue;
         }

         const auto record = connectivity.records.host_pointer[j];
         const auto & meta = metadata[ static_cast< std::size_t >( j ) ];
         found = true;
         ok = Check( meta.fine_element_id == info.element[0].index,
                     "2D NC builder metadata fine element" ) && ok;
         ok = Check( meta.coarse_element_id == info.element[1].index,
                     "2D NC builder metadata coarse element" ) && ok;
         ok = Check( meta.fine_mfem_local_face_id == 2,
                     "2D NC builder metadata fine local face" ) && ok;
         ok = Check( meta.coarse_mfem_local_face_id == 1,
                     "2D NC builder metadata coarse local face" ) && ok;
         ok = Check( meta.ncface == info.ncface,
                     "2D NC builder metadata ncface" ) && ok;
         ok = Check( record.minus_cell == static_cast< GlobalIndex >( info.element[1].index ),
                     "2D NC builder coarse/minus cell" ) && ok;
         ok = Check( record.plus_cell == static_cast< GlobalIndex >( info.element[0].index ),
                     "2D NC builder fine/plus cell" ) && ok;
         ok = Check( record.plus_orientation == MakePermutation< 2 >( { 2, -1 } ),
                     "2D NC builder selected plus orientation" ) && ok;
         ok = CheckPoint( record.minus_nonconforming_map.origin,
                          Point< 2 >{ 0.0, 0.0 },
                          "2D NC builder map origin" ) && ok;
         ok = CheckSize< 2 >( record.minus_nonconforming_map,
                              { 1.0, 0.5 },
                              "2D NC builder map size" ) && ok;
      }
   }

   ok = Check( found, "2D deterministic NC builder record was found" ) && ok;
   return ok;
}

bool TestMFEMNonconformingBuilder3D()
{
   using Geometry = HyperCube< 3 >;

   mfem::Mesh mesh = MakeTwoHexMesh(
      MakePermutation< 3 >( { -1, -2, 3 } ),
      MakePermutation< 3 >( { -1, 2, -3 } ),
      0,
      false );
   const auto bundle =
      MakeMFEMNonconformingGlobalInteriorFaceConnectivity< Geometry >( mesh );

   constexpr Integer family = 0;
   const auto & connectivity = std::get< family >( bundle.connectivity );
   const auto & metadata = std::get< family >( bundle.metadata );

   bool ok = true;
   ok = Check(
      static_cast< std::size_t >( connectivity.GetNumberOfFaces() ) == metadata.size(),
      "3D NC builder metadata aligns with family records" ) && ok;

   bool found = false;
   const auto & interior = mesh.GetFaceIndices( mfem::FaceType::Interior );
   for ( int i = 0; i < interior.Size(); ++i )
   {
      const int face_id = interior[i];
      if ( face_id != 8 ) { continue; }
      const auto info = mesh.GetFaceInformation( face_id );
      if ( !IsLeaf( info ) ) { continue; }

      for ( GlobalIndex j = 0; j < connectivity.GetNumberOfFaces(); ++j )
      {
         if ( metadata[ static_cast< std::size_t >( j ) ].source_face_id != face_id )
         {
            continue;
         }

         const auto record = connectivity.records.host_pointer[j];
         const auto & meta = metadata[ static_cast< std::size_t >( j ) ];
         found = true;
         ok = Check( meta.fine_element_id == info.element[0].index,
                     "3D NC builder metadata fine element" ) && ok;
         ok = Check( meta.coarse_element_id == info.element[1].index,
                     "3D NC builder metadata coarse element" ) && ok;
         ok = Check( meta.fine_mfem_local_face_id == 2,
                     "3D NC builder metadata fine local face" ) && ok;
         ok = Check( meta.coarse_mfem_local_face_id == 4,
                     "3D NC builder metadata coarse local face" ) && ok;
         ok = Check( meta.ncface == info.ncface,
                     "3D NC builder metadata ncface" ) && ok;
         ok = Check( record.minus_cell == static_cast< GlobalIndex >( info.element[1].index ),
                     "3D NC builder coarse/minus cell" ) && ok;
         ok = Check( record.plus_cell == static_cast< GlobalIndex >( info.element[0].index ),
                     "3D NC builder fine/plus cell" ) && ok;
         ok = Check( record.plus_orientation == MakePermutation< 3 >( { 1, -2, -3 } ),
                     "3D NC builder selected plus orientation" ) && ok;
         ok = CheckPoint( record.minus_nonconforming_map.origin,
                          Point< 3 >{ 0.0, 0.0, 0.5 },
                          "3D NC builder map origin" ) && ok;
         ok = CheckSize< 3 >( record.minus_nonconforming_map,
                              { 1.0, 0.5, 0.5 },
                              "3D NC builder map size" ) && ok;
      }
   }

   ok = Check( found, "3D deterministic NC builder record was found" ) && ok;
   return ok;
}

bool TestMixedAdaptiveSourceCompleteness()
{
   using Geometry = HyperCube< 2 >;

   mfem::Mesh mesh = MakeTwoQuadMesh( 0, 1, false );
   const auto & interior = mesh.GetFaceIndices( mfem::FaceType::Interior );
   std::set< int > source_ids;
   std::map< int, int > leaf_count_by_coarse_cell;
   int conforming = 0;
   int leaves = 0;
   int unsupported = 0;
   std::map< Integer, std::vector< int > > conforming_order_by_family;
   std::map< Integer, std::vector< int > > leaf_order_by_family;
   for ( int i = 0; i < interior.Size(); ++i )
   {
      const int face_id = interior[i];
      const auto info = mesh.GetFaceInformation( face_id );
      source_ids.insert( face_id );
      if ( info.IsConforming() )
      {
         ++conforming;
         conforming_order_by_family[
            TranslateMFEMFace2D( info.element[0].local_face_id ) ].push_back( face_id );
      }
      else if ( IsLeaf( info ) )
      {
         ++leaves;
         leaf_count_by_coarse_cell[ info.element[1].index ]++;
         leaf_order_by_family[
            TranslateMFEMFace2D( info.element[1].local_face_id ) ].push_back( face_id );
      }
      else
      {
         ++unsupported;
      }
   }

   bool repeated_coarse_writes = false;
   for ( const auto & entry : leaf_count_by_coarse_cell )
   {
      repeated_coarse_writes = repeated_coarse_writes || entry.second > 1;
   }
   bool family_order = true;
   for ( const auto & entry : conforming_order_by_family )
   {
      family_order = family_order &&
         std::is_sorted( entry.second.begin(), entry.second.end() );
   }
   for ( const auto & entry : leaf_order_by_family )
   {
      family_order = family_order &&
         std::is_sorted( entry.second.begin(), entry.second.end() );
   }

   const auto bundle =
      MakeMFEMGlobalInteriorFaceConnectivity< Geometry >( mesh );
   const auto built_conforming =
      CountConnectivityRecords( bundle.conforming.connectivity );
   const auto built_leaves =
      CountConnectivityRecords( bundle.nonconforming.connectivity );
   const auto built_conforming_metadata =
      CountMetadataRecords( bundle.conforming.metadata );
   const auto built_leaf_metadata =
      CountMetadataRecords( bundle.nonconforming.metadata );

   auto built_source_ids = CollectSourceFaceIds( bundle.conforming.metadata );
   auto built_leaf_ids = CollectSourceFaceIds( bundle.nonconforming.metadata );
   built_source_ids.insert(
      built_source_ids.end(),
      built_leaf_ids.begin(),
      built_leaf_ids.end() );
   std::sort( built_source_ids.begin(), built_source_ids.end() );
   const bool unique_built_sources =
      std::adjacent_find( built_source_ids.begin(), built_source_ids.end() ) ==
      built_source_ids.end();
   const std::set< int > built_source_set(
      built_source_ids.begin(),
      built_source_ids.end() );

   const auto built_conforming_order =
      SourceOrderByFamily< Geometry >( bundle.conforming.metadata );
   const auto built_leaf_order =
      SourceOrderByFamily< Geometry >( bundle.nonconforming.metadata );

   bool no_master_emitted = true;
   for ( int source_id : built_source_ids )
   {
      const auto info = mesh.GetFaceInformation( source_id );
      no_master_emitted = no_master_emitted &&
         ( info.IsConforming() || IsLeaf( info ) );
   }

   bool ok = true;
   mixed_conforming_count = conforming;
   mixed_leaf_count = leaves;
   mixed_source_count = interior.Size();
   mixed_built_conforming_count = built_conforming;
   mixed_built_leaf_count = built_leaves;
   ok = Check( unsupported == 0, "mixed mesh has no unsupported source interior faces" ) && ok;
   ok = Check( conforming + leaves == interior.Size(),
               "conforming plus leaf outputs cover all source interior faces" ) && ok;
   ok = Check( static_cast< int >( source_ids.size() ) == interior.Size(),
               "source face ids are unique" ) && ok;
   ok = Check( built_conforming == conforming,
               "complete builder conforming record count" ) && ok;
   ok = Check( built_leaves == leaves,
               "complete builder NC leaf record count" ) && ok;
   ok = Check( built_conforming + built_leaves == interior.Size(),
               "complete builder covers every source interior face exactly once" ) && ok;
   ok = Check( built_conforming_metadata == static_cast< std::size_t >( built_conforming ),
               "complete builder conforming metadata count" ) && ok;
   ok = Check( built_leaf_metadata == static_cast< std::size_t >( built_leaves ),
               "complete builder NC metadata count" ) && ok;
   ok = Check( unique_built_sources,
               "complete builder source ids are unique" ) && ok;
   ok = Check( built_source_set == source_ids,
               "complete builder source id set matches MFEM enumeration" ) && ok;
   ok = Check( no_master_emitted,
               "complete builder emits no coarse/master NC source face" ) && ok;
   ok = Check( built_conforming_order == conforming_order_by_family,
               "complete builder conforming family order follows MFEM source enumeration" ) && ok;
   ok = Check( built_leaf_order == leaf_order_by_family,
               "complete builder NC family order follows MFEM source enumeration" ) && ok;
   ok = Check( CheckFamilyLocalSourceOrder( bundle.conforming.metadata ),
               "complete builder conforming family ids are sorted" ) && ok;
   ok = Check( CheckFamilyLocalSourceOrder( bundle.nonconforming.metadata ),
               "complete builder NC family ids are sorted" ) && ok;
   ok = Check( leaves > 0 && conforming > 0,
               "mixed mesh contains both conforming and nonconforming leaves" ) && ok;
   ok = Check( repeated_coarse_writes,
               "several leaves reference one coarse cell" ) && ok;
   ok = Check( family_order,
               "family-local order follows MFEM source enumeration" ) && ok;
   return ok;
}

bool TestMFEMGlobalPartition2D()
{
   using Geometry = HyperCube< 2 >;
   constexpr Integer mesh_order = 1;

   mfem::Mesh mesh = MakeTwoQuadMesh( 0, 1, false );
   const auto partition =
      MakeGlobalPartition< Geometry, mesh_order >( mesh );
   using PartitionType = decltype( partition );
   using CellMeshType =
      typename std::tuple_element_t<
         0,
         typename PartitionType::cell_parts_type >::mesh_type;

   static_assert( is_partition_v< PartitionType > );
   static_assert( !HasLocalFaceInfo< CellMeshType >::value );

   const auto direct_interior =
      MakeMFEMGlobalInteriorFaceConnectivity< Geometry >( mesh );
   const auto direct_boundary =
      MakeMFEMGlobalBoundaryFaceConnectivity< Geometry >( mesh );

   const GlobalIndex expected_interior_records =
      CountConnectivityRecords( direct_interior.conforming.connectivity ) +
      CountConnectivityRecords( direct_interior.nonconforming.connectivity );
   const GlobalIndex expected_boundary_records =
      CountConnectivityRecords( direct_boundary.connectivity );

   auto fe = MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe },
         std::tuple{ L2Restriction{ 0 } } );

   bool ok = true;
   ok = Check( partition.GetNumberOfCellParts() == 1,
               "2D global partition has one cell part" ) && ok;
   ok = Check( partition.GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "2D global partition has conforming and NC interior face families" ) && ok;
   ok = Check( partition.GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "2D global partition has one boundary family per local face" ) && ok;
   ok = Check( CountPartitionInteriorFaceRecords( partition ) == expected_interior_records,
               "2D global partition interior record count matches direct builders" ) && ok;
   ok = Check( CountPartitionBoundaryFaceRecords( partition ) == expected_boundary_records,
               "2D global partition boundary record count matches direct builder" ) && ok;
   ok = Check( mixed.GetNumberOfCellFiniteElementSpaces() == 1,
               "2D global partition mixed FES has one cell space" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "2D global partition mixed FES exposes partition interior face parts" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "2D global partition mixed FES exposes partition boundary face parts" ) && ok;
   return ok;
}

bool TestMFEMGlobalPartition1D()
{
   using Geometry = HyperCube< 1 >;
   constexpr Integer mesh_order = 1;

   mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 3, 1.0 );
   mesh.SetCurvature( mesh_order );
   const auto partition =
      MakeGlobalPartition< Geometry, mesh_order >( mesh );
   using PartitionType = decltype( partition );
   using CellMeshType =
      typename std::tuple_element_t<
         0,
         typename PartitionType::cell_parts_type >::mesh_type;

   static_assert( is_partition_v< PartitionType > );
   static_assert( !HasLocalFaceInfo< CellMeshType >::value );

   const auto direct_interior =
      MakeMFEMGlobalInteriorFaceConnectivity< Geometry >( mesh );
   const auto direct_boundary =
      MakeMFEMGlobalBoundaryFaceConnectivity< Geometry >( mesh );

   const GlobalIndex expected_interior_records =
      CountConnectivityRecords( direct_interior.conforming.connectivity ) +
      CountConnectivityRecords( direct_interior.nonconforming.connectivity );
   const GlobalIndex expected_boundary_records =
      CountConnectivityRecords( direct_boundary.connectivity );

   auto fe = MakeLobattoFiniteElement( FiniteElementOrders< 1 >{} );
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe },
         std::tuple{ L2Restriction{ 0 } } );

   bool ok = true;
   ok = Check( partition.GetNumberOfCellParts() == 1,
               "1D global partition has one cell part" ) && ok;
   ok = Check( partition.GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "1D global partition has conforming and NC interior face families" ) && ok;
   ok = Check( partition.GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "1D global partition has one boundary family per local face" ) && ok;
   ok = Check( CountPartitionInteriorFaceRecords( partition ) == expected_interior_records,
               "1D global partition interior record count matches direct builders" ) && ok;
   ok = Check( CountPartitionBoundaryFaceRecords( partition ) == expected_boundary_records,
               "1D global partition boundary record count matches direct builder" ) && ok;
   ok = Check( mixed.GetNumberOfCellFiniteElementSpaces() == 1,
               "1D global partition mixed FES has one cell space" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "1D global partition mixed FES exposes partition interior face parts" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "1D global partition mixed FES exposes partition boundary face parts" ) && ok;
   return ok;
}

bool TestMFEMGlobalPartition3D()
{
   using Geometry = HyperCube< 3 >;
   constexpr Integer mesh_order = 1;

   mfem::Mesh mesh = MakeTwoHexMesh(
      MakePermutation< 3 >( { -1, -2, 3 } ),
      MakePermutation< 3 >( { -1, 2, -3 } ),
      0,
      false );
   const auto partition =
      MakeGlobalPartition< Geometry, mesh_order >( mesh );
   using PartitionType = decltype( partition );
   using CellMeshType =
      typename std::tuple_element_t<
         0,
         typename PartitionType::cell_parts_type >::mesh_type;

   static_assert( is_partition_v< PartitionType > );
   static_assert( !HasLocalFaceInfo< CellMeshType >::value );

   const auto direct_interior =
      MakeMFEMGlobalInteriorFaceConnectivity< Geometry >( mesh );
   const auto direct_boundary =
      MakeMFEMGlobalBoundaryFaceConnectivity< Geometry >( mesh );

   const GlobalIndex expected_interior_records =
      CountConnectivityRecords( direct_interior.conforming.connectivity ) +
      CountConnectivityRecords( direct_interior.nonconforming.connectivity );
   const GlobalIndex expected_boundary_records =
      CountConnectivityRecords( direct_boundary.connectivity );

   auto fe = MakeLobattoFiniteElement( FiniteElementOrders< 1, 1, 1 >{} );
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe },
         std::tuple{ L2Restriction{ 0 } } );

   bool ok = true;
   ok = Check( partition.GetNumberOfCellParts() == 1,
               "3D global partition has one cell part" ) && ok;
   ok = Check( partition.GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "3D global partition has conforming and NC interior face families" ) && ok;
   ok = Check( partition.GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "3D global partition has one boundary family per local face" ) && ok;
   ok = Check( CountPartitionInteriorFaceRecords( partition ) == expected_interior_records,
               "3D global partition interior record count matches direct builders" ) && ok;
   ok = Check( CountPartitionBoundaryFaceRecords( partition ) == expected_boundary_records,
               "3D global partition boundary record count matches direct builder" ) && ok;
   ok = Check( mixed.GetNumberOfCellFiniteElementSpaces() == 1,
               "3D global partition mixed FES has one cell space" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfInteriorFaceParts() == 2 * Geometry::num_faces,
               "3D global partition mixed FES exposes partition interior face parts" ) && ok;
   ok = Check( mixed.GetPartition().GetNumberOfBoundaryFaceParts() == Geometry::num_faces,
               "3D global partition mixed FES exposes partition boundary face parts" ) && ok;
   return ok;
}

bool TestSpecializedAdvectionCompileSpike()
{
   using Geometry = HyperCube< 2 >;
   using Connectivity = UnstructuredNonconformingInteriorFaceConnectivity< Geometry, 2 >;
   using Record = typename Connectivity::record_type;

   NonconformingHyperCubeFaceMap< 2 > lower_map;
   lower_map.origin = Point< 2 >{ 0.0, 0.0 };
   lower_map.size = { 1.0, 0.5 };
   NonconformingHyperCubeFaceMap< 2 > upper_map;
   upper_map.origin = Point< 2 >{ 0.0, 0.5 };
   upper_map.size = { 1.0, 0.5 };

   const auto lower_orientation =
      Decode2(
         1,
         2,
         { 0.5, 0.0 } ).value.plus_orientation;
   const auto upper_orientation =
      Decode2(
         1,
         2,
         { 1.0, 0.5 } ).value.plus_orientation;

   Connectivity connectivity(
      std::vector< Record >{
         Record{ 0, 0, lower_orientation, lower_map },
         Record{ 0, 1, upper_orientation, upper_map }
      } );

   constexpr Integer p = 2;
   constexpr Integer q1d = p + 2;
   Cartesian2DMesh coarse_mesh( 1.0, 1.0, 1, 1, Point< 2 >{ 0.0, 0.0 } );
   Cartesian2DMesh fine_mesh( 1.0, 0.5, 1, 2, Point< 2 >{ 1.0, 0.0 } );
   auto fe = MakeLobattoFiniteElement( FiniteElementOrders< p, p >{} );
   L2Restriction coarse_restriction{ 0 };
   auto coarse_fes = MakeFiniteElementSpace( coarse_mesh, fe, coarse_restriction );
   const Integer coarse_dofs = coarse_fes.GetNumberOfFiniteElementDofs();
   L2Restriction fine_restriction{ coarse_dofs };
   auto fine_fes = MakeFiniteElementSpace( fine_mesh, fe, fine_restriction );
   const Integer fine_dofs = fine_fes.GetNumberOfFiniteElementDofs();

   auto int_rule = MakeIntegrationRule( IntegrationRuleNumPoints< q1d, q1d >{} );
   auto velocity = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 2 > &,
      Real ( & v )[2] )
   {
      v[0] = 1.0;
      v[1] = 0.0;
   };

   auto partition =
      MakePartition(
         MakeCellPart( coarse_mesh ),
         MakeCellPart( fine_mesh ),
         MakeInteriorFacePart< 0, 1 >( connectivity ) );
   [[maybe_unused]] auto global_space =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ fe, fe },
         std::tuple{ coarse_restriction, fine_restriction } );

   auto face_operator =
      MakeAdvectionFaceOperator< SerialKernelConfiguration >(
         coarse_fes,
         fine_fes,
         connectivity,
         int_rule,
         velocity );

   Vector input( coarse_dofs + fine_dofs );
   Vector output( coarse_dofs + fine_dofs );
   Real * in = input.WriteHostData();
   for ( Integer i = 0; i < coarse_dofs + fine_dofs; ++i )
   {
      in[i] = Real( 0.125 ) + Real( i + 1 ) * Real( 0.0625 );
   }
   output = 0.0;
   face_operator( input, output );

   const Real * out = output.ReadHostData();
   Real coarse_norm = 0.0;
   Real fine_norm = 0.0;
   for ( Integer i = 0; i < coarse_dofs; ++i )
   {
      coarse_norm += std::abs( out[i] );
   }
   for ( Integer i = 0; i < fine_dofs; ++i )
   {
      fine_norm += std::abs( out[coarse_dofs + i] );
   }

   bool ok = true;
   ok = Check( connectivity.GetNumberOfFaces() == 2,
               "advection spike has two NC leaf records" ) && ok;
   ok = Check( coarse_norm > DecoderTolerance(),
               "advection spike accumulates repeated coarse writes" ) && ok;
   ok = Check( fine_norm > DecoderTolerance(),
               "advection spike writes fine-side residuals" ) && ok;
   return ok;
}

} // namespace

int main()
{
   bool ok = true;
   ok = TestDeterministic2DNonidentity() && ok;
   ok = TestDeterministic3DNonidentity() && ok;
   ok = TestMFEMNonconformingBuilder2D() && ok;
   ok = TestMFEMNonconformingBuilder3D() && ok;
   ok = TestMixedAdaptiveSourceCompleteness() && ok;
   ok = TestMFEMGlobalPartition1D() && ok;
   ok = TestMFEMGlobalPartition2D() && ok;
   ok = TestMFEMGlobalPartition3D() && ok;
   ok = TestSpecializedAdvectionCompileSpike() && ok;
   std::cout << "MFEM oracle max errors: Loc1=" << max_loc1_error
             << " Loc2=" << max_loc2_error << "\n";
   std::cout << "Mixed adaptive source counts: conforming="
             << mixed_conforming_count
             << " leaves=" << mixed_leaf_count
             << " source_faces=" << mixed_source_count << "\n";
   std::cout << "Mixed adaptive builder counts: conforming="
             << mixed_built_conforming_count
             << " leaves=" << mixed_built_leaf_count << "\n";
   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": MFEM nonconforming point-matrix cross-check\n";
   return ok ? 0 : 1;
}
