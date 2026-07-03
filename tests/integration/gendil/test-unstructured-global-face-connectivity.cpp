// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <tuple>
#include <vector>

using namespace gendil;

namespace
{

template < Integer Dim >
Permutation< Dim > Orientation( const std::array< LocalIndex, Dim > & dims )
{
   Permutation< Dim > orientation;
   orientation.dimension_indices = dims;
   return orientation;
}

template < typename Geometry >
void SetBoundaryFaces( UnstructuredConformingConnectivity< Geometry > & connectivity,
                       const GlobalIndex num_cells )
{
   constexpr Integer Dim = Geometry::geometry_dim;
   for ( GlobalIndex cell = 0; cell < num_cells; ++cell )
   {
      for ( Integer face = 0; face < Geometry::num_faces; ++face )
      {
         connectivity[ cell ].faces[ face ] =
            { GlobalIndex{}, {}, MakeReferencePermutation< Dim >(), {}, {}, true };
      }
   }
}

template < typename Geometry >
void SetInteriorFace(
   UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex cell_a,
   const Integer face_a,
   const GlobalIndex cell_b,
   const Integer face_b,
   const Permutation< Geometry::geometry_dim > & orientation_a_to_b,
   const Permutation< Geometry::geometry_dim > & orientation_b_to_a )
{
   connectivity[ cell_a ].faces[ face_a ] =
      { cell_b, {}, orientation_a_to_b, {}, {}, false };
   connectivity[ cell_b ].faces[ face_b ] =
      { cell_a, {}, orientation_b_to_a, {}, {}, false };
}

template < typename Geometry >
void SetOneWayInteriorFace(
   UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex cell,
   const Integer face,
   const GlobalIndex neighbor,
   const Permutation< Geometry::geometry_dim > & orientation )
{
   connectivity[ cell ].faces[ face ] =
      { neighbor, {}, orientation, {}, {}, false };
}

template < typename Connectivity >
struct ConnectivityOnlyMesh
{
   Connectivity connectivity;
   GlobalIndex num_cells;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfCells() const
   {
      return num_cells;
   }
};

bool Near( const Real a, const Real b, const Real tol = 1.0e-12 )
{
   return std::abs( a - b ) <= tol;
}

bool TestSingleCellBoundaryCounts()
{
   using Geometry = HyperCube< 2 >;
   UnstructuredConformingConnectivity< Geometry > connectivity( 1 );
   SetBoundaryFaces( connectivity, 1 );

   auto boundary_faces = MakeUnstructuredBoundaryFaceConnectivity( connectivity, 1 );
   static_assert( std::tuple_size_v< decltype( boundary_faces ) > == Geometry::num_faces );

   bool ok = true;
   std::apply(
      [&]( const auto & ... face_meshes )
      {
         ( ( ok = ok && face_meshes.GetNumberOfFaces() == 1 ), ... );
      },
      boundary_faces );

   auto face_info = std::get< 3 >( boundary_faces ).GetGlobalFaceInfo( 0 );
   ok = ok && face_info.MinusSide().GetCellIndex() == 0;
   ok = ok && IsBoundaryFace( face_info );
   ok = ok && face_info.IsBoundary();
   ok = ok && face_info.IsConforming();

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": unstructured global boundary faces for one quad\n";
   return ok;
}

bool TestTwoCellInteriorBucketAndOrientation()
{
   using Geometry = HyperCube< 2 >;
   const auto expected_orientation = Orientation< 2 >( { -1, 2 } );

   UnstructuredConformingConnectivity< Geometry > connectivity( 2 );
   SetBoundaryFaces( connectivity, 2 );
   SetInteriorFace(
      connectivity,
      0,
      2,
      1,
      0,
      expected_orientation,
      MakeReferencePermutation< 2 >() );

   auto interior_faces = MakeUnstructuredInteriorFaceConnectivity( connectivity, 2 );
   static_assert( std::tuple_size_v< decltype( interior_faces ) > == Geometry::num_faces );
   static_assert(
      global_face_mesh_has_static_face_family_v<
         std::tuple_element_t< 2, decltype( interior_faces ) > > );
   static_assert(
      mesh::GlobalFaceMeshConnectivity<
         std::tuple_element_t< 2, decltype( interior_faces ) > > );

   bool ok = true;
   ok = ok && std::get< 0 >( interior_faces ).GetNumberOfFaces() == 0;
   ok = ok && std::get< 1 >( interior_faces ).GetNumberOfFaces() == 0;
   ok = ok && std::get< 2 >( interior_faces ).GetNumberOfFaces() == 1;
   ok = ok && std::get< 3 >( interior_faces ).GetNumberOfFaces() == 0;

   auto face_info = std::get< 2 >( interior_faces ).GetGlobalFaceInfo( 0 );
   ok = ok && face_info.MinusSide().GetCellIndex() == 0;
   ok = ok && face_info.PlusSide().GetCellIndex() == 1;
   using MinusOrientation =
      std::remove_cvref_t< decltype( face_info.MinusSide().GetOrientation() ) >;
   ok = ok && MinusOrientation::value == MakeReferencePermutation< 2 >();
   ok = ok && face_info.PlusSide().GetOrientation() == expected_orientation;
   ok = ok && !IsBoundaryFace( face_info );
   ok = ok && !face_info.IsBoundary();
   ok = ok && face_info.IsConforming();

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": unstructured global interior face bucket and orientation\n";
   return ok;
}

bool TestDeterministicInteriorOrdering()
{
   using Geometry = HyperCube< 2 >;
   UnstructuredConformingConnectivity< Geometry > connectivity( 3 );
   SetBoundaryFaces( connectivity, 3 );
   SetInteriorFace(
      connectivity,
      0,
      2,
      1,
      0,
      MakeReferencePermutation< 2 >(),
      MakeReferencePermutation< 2 >() );
   SetInteriorFace(
      connectivity,
      1,
      2,
      2,
      0,
      MakeReferencePermutation< 2 >(),
      MakeReferencePermutation< 2 >() );

   auto interior_faces = MakeUnstructuredInteriorFaceConnectivity( connectivity, 3 );
   const auto & x_plus_faces = std::get< 2 >( interior_faces );

   bool ok = true;
   ok = ok && x_plus_faces.GetNumberOfFaces() == 2;

   auto first = x_plus_faces.GetGlobalFaceInfo( 0 );
   auto second = x_plus_faces.GetGlobalFaceInfo( 1 );
   ok = ok && first.MinusSide().GetCellIndex() == 0;
   ok = ok && first.PlusSide().GetCellIndex() == 1;
   ok = ok && second.MinusSide().GetCellIndex() == 1;
   ok = ok && second.PlusSide().GetCellIndex() == 2;

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": deterministic unstructured global interior ordering\n";
   return ok;
}

bool TestMultipleFacetsBetweenSameCellPair()
{
   using Geometry = HyperCube< 2 >;
   const auto face_one_orientation = Orientation< 2 >( { -1, 2 } );
   const auto face_two_orientation = Orientation< 2 >( { 2, 1 } );

   UnstructuredConformingConnectivity< Geometry > connectivity( 2 );
   SetBoundaryFaces( connectivity, 2 );
   SetInteriorFace(
      connectivity,
      0,
      1,
      1,
      3,
      face_one_orientation,
      MakeReferencePermutation< 2 >() );
   SetInteriorFace(
      connectivity,
      0,
      2,
      1,
      0,
      face_two_orientation,
      MakeReferencePermutation< 2 >() );

   auto interior_faces = MakeUnstructuredInteriorFaceConnectivity( connectivity, 2 );

   bool ok = true;
   ok = ok && std::get< 0 >( interior_faces ).GetNumberOfFaces() == 0;
   ok = ok && std::get< 1 >( interior_faces ).GetNumberOfFaces() == 1;
   ok = ok && std::get< 2 >( interior_faces ).GetNumberOfFaces() == 1;
   ok = ok && std::get< 3 >( interior_faces ).GetNumberOfFaces() == 0;

   const auto first = std::get< 1 >( interior_faces ).GetGlobalFaceInfo( 0 );
   ok = ok && first.MinusSide().GetCellIndex() == 0;
   ok = ok && first.PlusSide().GetCellIndex() == 1;
   ok = ok && first.PlusSide().GetOrientation() == face_one_orientation;

   const auto second = std::get< 2 >( interior_faces ).GetGlobalFaceInfo( 0 );
   ok = ok && second.MinusSide().GetCellIndex() == 0;
   ok = ok && second.PlusSide().GetCellIndex() == 1;
   ok = ok && second.PlusSide().GetOrientation() == face_two_orientation;

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": multiple unstructured facets between same cell pair\n";
   return ok;
}

int RunInvalidNeighborDiagnostic()
{
   using Geometry = HyperCube< 2 >;
   UnstructuredConformingConnectivity< Geometry > connectivity( 1 );
   SetBoundaryFaces( connectivity, 1 );
   SetOneWayInteriorFace(
      connectivity,
      0,
      2,
      1,
      MakeReferencePermutation< 2 >() );

   [[maybe_unused]] auto interior_faces =
      MakeUnstructuredInteriorFaceConnectivity( connectivity, 1 );
   return 0;
}

int RunSelfNeighborDiagnostic()
{
   using Geometry = HyperCube< 2 >;
   UnstructuredConformingConnectivity< Geometry > connectivity( 1 );
   SetBoundaryFaces( connectivity, 1 );
   SetOneWayInteriorFace(
      connectivity,
      0,
      2,
      0,
      MakeReferencePermutation< 2 >() );

   [[maybe_unused]] auto interior_faces =
      MakeUnstructuredInteriorFaceConnectivity( connectivity, 1 );
   return 0;
}

bool TestNonconformingFaceInfoAndAliases()
{
   using Geometry = HyperCube< 2 >;
   using Connectivity = UnstructuredNonconformingInteriorFaceConnectivity< Geometry, 2 >;
   using Info = typename Connectivity::face_info_type;

   static_assert( is_hypercube_geometry< Geometry >::value );
   static_assert( Geometry::GetOppositeFaceIndex( 2 ) == 0 );
   static_assert( std::tuple_size_v< UnstructuredConformingInteriorConnectivityTuple< Geometry > > == Geometry::num_faces );
   static_assert( std::tuple_size_v< UnstructuredNonconformingInteriorConnectivityTuple< Geometry > > == Geometry::num_faces );
   static_assert( std::tuple_size_v< UnstructuredBoundaryFaceConnectivityTuple< Geometry > > == Geometry::num_faces );
   static_assert( mesh::GlobalFaceMeshConnectivity< Connectivity > );
   static_assert( global_face_mesh_has_static_face_family_v< Connectivity > );
   static_assert( global_face_mesh_minus_local_face_index_v< Connectivity > == 2 );
   static_assert( global_face_mesh_plus_local_face_index_v< Connectivity > == 0 );
   static_assert( std::same_as< typename Info::minus_side_type::orientation_type, IdentityOrientation< 2 > > );
   static_assert( std::same_as< typename Info::plus_side_type::orientation_type, Permutation< 2 > > );
   static_assert( std::same_as< typename Info::minus_side_type::conformity_type, NonconformingHyperCubeFaceMap< 2 > > );
   static_assert( std::same_as< typename Info::plus_side_type::conformity_type, ConformingFaceMap< 2 > > );
   static_assert( !Info::minus_side_type::boundary_type::value );
   static_assert( !Info::plus_side_type::boundary_type::value );
   static_assert( !Info::minus_side_type::is_conforming );
   static_assert( Info::plus_side_type::is_conforming );
   static_assert( std::is_trivially_copyable_v< typename Connectivity::record_type > );

   Connectivity empty;
   bool ok = empty.GetNumberOfFaces() == 0;

   NonconformingHyperCubeFaceMap< 2 > map;
   map.origin = Point< 2 >{ 0.0, 0.25 };
   map.size = { 1.0, 0.5 };

   std::vector< typename Connectivity::record_type > host_records{
      { 7, 9, Orientation< 2 >( { 1, -2 } ), map }
   };
   Connectivity connectivity( host_records );
   host_records[0].minus_cell = 42;
   Connectivity copied_connectivity = connectivity;
   Connectivity assigned_connectivity;
   assigned_connectivity = connectivity;

   ok = ok && connectivity.GetNumberOfFaces() == 1;
   ok = ok && copied_connectivity.GetGlobalFaceInfo( 0 ).MinusSide().GetCellIndex() == 7;
   ok = ok && assigned_connectivity.GetGlobalFaceInfo( 0 ).PlusSide().GetCellIndex() == 9;

   const auto info = connectivity.GetGlobalFaceInfo( 0 );
   ok = ok && info.MinusSide().GetCellIndex() == 7;
   ok = ok && info.PlusSide().GetCellIndex() == 9;
   ok = ok && !info.IsBoundary();
   ok = ok && !info.IsConforming();
   using MinusOrientation =
      std::remove_cvref_t< decltype( info.MinusSide().GetOrientation() ) >;
   ok = ok && MinusOrientation::value == MakeReferencePermutation< 2 >();
   ok = ok && info.PlusSide().GetOrientation() == Orientation< 2 >( { 1, -2 } );

   const Point< 2 > mapped =
      info.MinusSide().MapReferenceToFaceCoordinates( Point< 2 >{ 1.0, 0.5 } );
   ok = ok && Near( mapped[0], 1.0 );
   ok = ok && Near( mapped[1], 0.5 );
   ok = ok && Near( info.MinusSide().Measure(), 0.5 );
   ok = ok && Near( info.PlusSide().Measure(), 1.0 );

   GlobalIndex visited = 0;
   mesh::GlobalFaceIterator< SerialKernelConfiguration >(
      connectivity,
      [&] ( GlobalIndex face_index )
      {
         const auto face_info = connectivity.GetGlobalFaceInfo( face_index );
         ok = ok && face_info.MinusSide().GetCellIndex() == 7;
         ++visited;
      } );
   ok = ok && visited == 1;

   Cartesian2DMesh mesh( 1.0, 1.0, 1, 1, Point< 2 >{ 0.0, 0.0 } );
   auto finite_element = MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto partition =
      MakePartition(
         MakeCellPart( mesh ),
         MakeInteriorFacePart< 0, 0 >( connectivity ) );
   auto mixed =
      MakeMixedFiniteElementSpace(
         partition,
         std::tuple{ finite_element },
         DGDirectSumNumbering{} );
   auto wf_ctx =
      MakeWeakFormContext(
         MakeTrialField<"u">( mixed ),
         MakeIntegrationDomain<"mesh">( mixed ) );
   using IntegrationRule =
      decltype( MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} ) );
   ForEachInteriorFaceFiniteElementSpace(
      wf_ctx,
      InteriorFacets<"mesh">{},
      [&] ( const auto& batch )
      {
         [[maybe_unused]] auto mesh_qd =
            MakeGlobalFacetMeshQuadData< IntegrationRule >( batch );
         auto restricted =
            MakeRestrictedWeakFormContext<"u", "u">(
               wf_ctx,
               InteriorFacets<"mesh">{},
               batch );
         [[maybe_unused]] auto fe_qd =
            MakeGlobalFacetFiniteElementQuadData< IntegrationRule >(
               restricted.template fe_field<"u">().space );
      } );

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": unstructured nonconforming global face contract\n";
   return ok;
}

bool TestMeshFactoryOverload()
{
   using Geometry = HyperCube< 1 >;
   UnstructuredConformingConnectivity< Geometry > connectivity( 2 );
   SetBoundaryFaces( connectivity, 2 );
   SetInteriorFace(
      connectivity,
      0,
      1,
      1,
      0,
      MakeReferencePermutation< 1 >(),
      MakeReferencePermutation< 1 >() );

   ConnectivityOnlyMesh< decltype( connectivity ) > mesh{ connectivity, 2 };
   auto interior_faces = MakeUnstructuredInteriorFaceConnectivity( mesh );
   auto boundary_faces = MakeUnstructuredBoundaryFaceConnectivity( mesh );
   static_assert(
      mesh::GlobalFaceMeshConnectivity<
         std::tuple_element_t< 0, decltype( boundary_faces ) > > );

   bool ok = true;
   ok = ok && std::get< 1 >( interior_faces ).GetNumberOfFaces() == 1;
   ok = ok && std::get< 0 >( boundary_faces ).GetNumberOfFaces() == 1;
   ok = ok && std::get< 1 >( boundary_faces ).GetNumberOfFaces() == 1;

   std::cout << ( ok ? "PASS" : "FAIL" )
             << ": unstructured mesh global face factory overloads\n";
   return ok;
}

} // namespace

int main( int argc, char ** argv )
{
   if ( argc == 2 )
   {
      const std::string mode = argv[1];
      if ( mode == "invalid-neighbor" )
      {
         return RunInvalidNeighborDiagnostic();
      }
      if ( mode == "self-neighbor" )
      {
         return RunSelfNeighborDiagnostic();
      }
      std::cerr << "Unknown test mode: " << mode << "\n";
      return 1;
   }

   bool ok = true;
   ok = TestSingleCellBoundaryCounts() && ok;
   ok = TestTwoCellInteriorBucketAndOrientation() && ok;
   ok = TestDeterministicInteriorOrdering() && ok;
   ok = TestMultipleFacetsBetweenSameCellPair() && ok;
   ok = TestNonconformingFaceInfoAndAliases() && ok;
   ok = TestMeshFactoryOverload() && ok;

   return ok ? 0 : 1;
}
