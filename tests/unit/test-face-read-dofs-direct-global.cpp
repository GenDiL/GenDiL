// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gendil;

namespace
{

template < Integer Dim >
struct TestFaceView
{
   using orientation_type = Permutation< Dim >;

   static constexpr Integer dim = Dim;
   static constexpr bool is_conforming = true;

   GlobalIndex cell_index;
   orientation_type orientation;

   GENDIL_HOST_DEVICE
   GlobalIndex GetCellIndex() const { return cell_index; }

   GENDIL_HOST_DEVICE
   const orientation_type & GetOrientation() const { return orientation; }

   GENDIL_HOST_DEVICE
   Point< Dim > GetReferenceNormal() const { return {}; }
};

template < typename ShapeFunctions >
struct TestFiniteElement
{
   using shape_functions = ShapeFunctions;

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumDofs()
   {
      return Product(
         orders_to_num_dofs< typename shape_functions::orders >{} );
   }
};

template < Integer SpaceDim, typename ShapeFunctions >
struct TestFiniteElementSpace
{
   static constexpr Integer Dim = SpaceDim;
   using finite_element_type = TestFiniteElement< ShapeFunctions >;
};

struct DirectGlobalSerialKernelConfiguration :
   public HostKernelConfiguration< 1 >
{
   using face_read_dofs_policy = DirectGlobalFaceReadDofsPolicy;
};

template < Integer Dim >
std::vector< Permutation< Dim > > MakeAllSignedPermutations()
{
   std::array< LocalIndex, Dim > axes{};
   for ( Integer i = 0; i < Dim; ++i )
   {
      axes[ i ] = static_cast< LocalIndex >( i + 1 );
   }

   std::vector< Permutation< Dim > > orientations;
   do
   {
      for ( Integer sign_mask = 0; sign_mask < ( Integer{ 1 } << Dim );
            ++sign_mask )
      {
         Permutation< Dim > orientation{};
         for ( Integer i = 0; i < Dim; ++i )
         {
            const bool reversed = ( sign_mask & ( Integer{ 1 } << i ) ) != 0;
            orientation( i ) = reversed ? -axes[ i ] : axes[ i ];
         }
         orientations.push_back( orientation );
      }
   } while ( std::next_permutation( axes.begin(), axes.end() ) );

   return orientations;
}

template < Integer Dim >
void PrintIndex( const std::array< GlobalIndex, Dim > & index )
{
   std::cout << "(";
   for ( Integer i = 0; i < Dim; ++i )
   {
      if ( i > 0 )
      {
         std::cout << ",";
      }
      std::cout << index[ i ];
   }
   std::cout << ")";
}

template < Integer Dim >
void PrintOrientation( const Permutation< Dim > & orientation )
{
   std::cout << "(";
   for ( Integer i = 0; i < Dim; ++i )
   {
      if ( i > 0 )
      {
         std::cout << ",";
      }
      std::cout << orientation( i );
   }
   std::cout << ")";
}

template < typename Layout, Integer Dim, size_t... Is >
GlobalIndex OffsetAt(
   const Layout & layout,
   const std::array< GlobalIndex, Dim > & index,
   std::index_sequence< Is... > )
{
   return layout.Offset( index[ Is ]... );
}

template < typename Layout, Integer Dim >
GlobalIndex OffsetAt(
   const Layout & layout,
   const std::array< GlobalIndex, Dim > & index )
{
   return OffsetAt(
      layout,
      index,
      std::make_index_sequence< Dim >{} );
}

template < Integer Dim >
Real IndexEncodedValue(
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index )
{
   Real value = static_cast< Real >( 10000 * ( element_index + 1 ) );
   Real scale = 1.0;
   for ( Integer i = 0; i < Dim; ++i )
   {
      value += scale * static_cast< Real >( native_indices[ i ] + 1 );
      scale *= 10.0;
   }
   return value;
}

template < typename DofShape >
bool RunOrientationMapCase( const char * case_name )
{
   constexpr Integer dim = DofShape::size();
   const auto sizes = to_array( DofShape{} );
   const auto orientations = MakeAllSignedPermutations< dim >();
   const auto identity = MakeReferencePermutation< dim >();
   bool success = true;
   Integer num_failures_reported = 0;

   for ( const auto & orientation : orientations )
   {
      const auto oriented_layout = MakeOrientedLayout( sizes, orientation );

      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, dim > reference_indices{
            static_cast< GlobalIndex >( k )... };
         const auto native_indices =
            DirectGlobalFaceReadNativeIndex(
               reference_indices,
               sizes,
               orientation );

         const GlobalIndex fifo_offset =
            FaceReadDofsFIFOOffset( reference_indices, sizes );
         const GlobalIndex oriented_offset =
            OffsetAt( oriented_layout, native_indices );

         const bool offsets_match = oriented_offset == fifo_offset;
         const bool identity_match =
            orientation != identity || native_indices == reference_indices;

         if ( !offsets_match || !identity_match )
         {
            success = false;
            if ( num_failures_reported < 8 )
            {
               ++num_failures_reported;
               std::cout
                  << "Orientation-map mismatch in " << case_name
                  << ": orientation=";
               PrintOrientation( orientation );
               std::cout << ", reference=";
               PrintIndex( reference_indices );
               std::cout << ", native=";
               PrintIndex( native_indices );
               std::cout << ", fifo_offset=" << fifo_offset
                         << ", oriented_offset=" << oriented_offset
                         << ", identity_match=" << identity_match
                         << "\n";
            }
         }
      });
   }

   if ( success )
   {
      std::cout << "PASS orientation map: " << case_name << "\n";
   }
   return success;
}

template < Integer Dim >
std::array< GlobalIndex, Dim > OrientedAxisRadices(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   std::array< GlobalIndex, Dim > radices{};
   for ( Integer native_dim = 0; native_dim < Dim; ++native_dim )
   {
      const LocalIndex o = orientation( native_dim );
      const Integer oriented_axis =
         static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
      radices[ oriented_axis ] =
         static_cast< GlobalIndex >( sizes[ native_dim ] );
   }
   return radices;
}

bool RunFocusedAnisotropicSwapCase()
{
   using DofShape = std::index_sequence< 3, 4 >;
   constexpr Integer dim = DofShape::size();

   const auto sizes = to_array( DofShape{} );
   const std::array< GlobalIndex, dim > reference_extents{
      static_cast< GlobalIndex >( sizes[ 0 ] ),
      static_cast< GlobalIndex >( sizes[ 1 ] ) };
   const std::array< GlobalIndex, dim > expected_reference_extents{ 3, 4 };

   const Permutation< dim > orientation{ { 2, -1 } };
   const auto oriented_radices =
      OrientedAxisRadices( sizes, orientation );
   const std::array< GlobalIndex, dim > expected_oriented_radices{ 4, 3 };
   const auto oriented_layout = MakeOrientedLayout( sizes, orientation );

   bool success =
      reference_extents == expected_reference_extents &&
      oriented_radices == expected_oriented_radices;
   Integer num_failures_reported = 0;

   UnitLoop< DofShape >( [&] ( auto... k )
   {
      const std::array< GlobalIndex, dim > reference_indices{
         static_cast< GlobalIndex >( k )... };
      const auto native_indices =
         DirectGlobalFaceReadNativeIndex(
            reference_indices,
            sizes,
            orientation );

      const GlobalIndex fifo_offset =
         FaceReadDofsFIFOOffset( reference_indices, sizes );
      const GlobalIndex oriented_offset =
         OffsetAt( oriented_layout, native_indices );

      if ( oriented_offset != fifo_offset )
      {
         success = false;
         if ( num_failures_reported < 8 )
         {
            ++num_failures_reported;
            std::cout
               << "Focused anisotropic swap mismatch: reference=";
            PrintIndex( reference_indices );
            std::cout << ", native=";
            PrintIndex( native_indices );
            std::cout << ", fifo_offset=" << fifo_offset
                      << ", oriented_offset=" << oriented_offset
                      << "\n";
         }
      }
   });

   const std::array< GlobalIndex, dim > trap_reference{ 0, 1 };
   const std::array< GlobalIndex, dim > tempting_native{
      trap_reference[ 1 ],
      static_cast< GlobalIndex >( sizes[ 1 ] - 1 - trap_reference[ 0 ] ) };
   const GlobalIndex trap_fifo_offset =
      FaceReadDofsFIFOOffset( trap_reference, sizes );
   const GlobalIndex tempting_offset =
      OffsetAt( oriented_layout, tempting_native );

   if ( tempting_offset == trap_fifo_offset )
   {
      success = false;
      std::cout
         << "Focused anisotropic swap failed to catch the tempting direct "
         << "formula for reference=";
      PrintIndex( trap_reference );
      std::cout << ", tempting_native=";
      PrintIndex( tempting_native );
      std::cout << ", fifo_offset=" << trap_fifo_offset
                << ", tempting_offset=" << tempting_offset
                << "\n";
   }

   if ( success )
   {
      std::cout
         << "PASS focused anisotropic swap: shape (3,4), orientation (2,-1), "
         << "reference extents are original (3,4), oriented-axis radices are "
         << "(4,3).\n";
   }
   else
   {
      std::cout
         << "FAILED focused anisotropic swap: expected original reference "
         << "extents (3,4) and oriented-axis radices (4,3).\n";
   }

   return success;
}

template < typename Space, size_t... Is >
auto MakeGlobalDofView(
   std::vector< Real > & data,
   const std::array< GlobalIndex, Space::Dim + 1 > & sizes,
   std::index_sequence< Is... > )
{
   return MakeFIFOView( data.data(), sizes[ Is ]... );
}

template < typename Space >
auto MakeGlobalDofView(
   std::vector< Real > & data,
   const std::array< GlobalIndex, Space::Dim + 1 > & sizes )
{
   return MakeGlobalDofView< Space >(
      data,
      sizes,
      std::make_index_sequence< Space::Dim + 1 >{} );
}

template < typename LocalDofs, typename DofShape >
bool CompareLocalDofs(
   const char * case_name,
   const LocalDofs & full_shared,
   const LocalDofs & direct_global,
   const Permutation< DofShape::size() > & orientation )
{
   constexpr Integer dim = DofShape::size();
   bool success = true;
   Integer num_failures_reported = 0;

   UnitLoop< DofShape >( [&] ( auto... k )
   {
      const Real full_value = full_shared( k... );
      const Real direct_value = direct_global( k... );
      if ( std::abs( full_value - direct_value ) > 1e-12 )
      {
         success = false;
         if ( num_failures_reported < 8 )
         {
            ++num_failures_reported;
            const std::array< GlobalIndex, dim > reference_indices{
               static_cast< GlobalIndex >( k )... };
            std::cout
               << "ReadDofs policy mismatch in " << case_name
               << ": orientation=";
            PrintOrientation( orientation );
            std::cout << ", reference=";
            PrintIndex( reference_indices );
            std::cout << ", FullShared=" << full_value
                      << ", DirectGlobal=" << direct_value
                      << "\n";
         }
      }
   });

   return success;
}

template < typename Space >
bool RunScalarReadDofsPolicyCase( const char * case_name )
{
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   using DofShape = orders_to_num_dofs< typename ShapeFunctions::orders >;
   constexpr Integer dim = Space::Dim;
   constexpr GlobalIndex num_elements = 2;
   constexpr GlobalIndex element_index = 1;

   const auto dof_sizes = to_array( DofShape{} );
   std::array< GlobalIndex, dim + 1 > global_sizes{};
   GlobalIndex num_values = num_elements;
   for ( Integer i = 0; i < dim; ++i )
   {
      global_sizes[ i ] = static_cast< GlobalIndex >( dof_sizes[ i ] );
      num_values *= global_sizes[ i ];
   }
   global_sizes[ dim ] = num_elements;

   std::vector< Real > global_data( num_values );
   auto global_dofs = MakeGlobalDofView< Space >( global_data, global_sizes );

   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, dim > native_indices{
            static_cast< GlobalIndex >( k )... };
         global_dofs( k..., element ) =
            IndexEncodedValue( native_indices, element );
      });
   }

   Space fe_space{};
   Real * no_shared_memory = nullptr;
   KernelContext< HostKernelConfiguration< 1 >, 0 >
      full_shared_context( no_shared_memory );
   KernelContext< DirectGlobalSerialKernelConfiguration, 0 >
      direct_global_context( no_shared_memory );

   bool success = true;
   const auto orientations = MakeAllSignedPermutations< dim >();
   for ( const auto & orientation : orientations )
   {
      const TestFaceView< dim > face{ element_index, orientation };
      const auto full_shared =
         ReadDofs(
            full_shared_context,
            fe_space,
            face,
            global_dofs );
      const auto direct_global =
         ReadDofs(
            direct_global_context,
            fe_space,
            face,
            global_dofs );

      success =
         CompareLocalDofs< decltype( full_shared ), DofShape >(
            case_name,
            full_shared,
            direct_global,
            orientation ) &&
         success;
   }

   if ( success )
   {
      std::cout << "PASS scalar ReadDofs policy equivalence: "
                << case_name << "\n";
   }
   return success;
}

bool RunVectorFaceReadAudit()
{
   std::cout
      << "Vector face ReadDofs DirectGlobal audit: current vector face reads "
      << "apply the scalar orientation independently per component; no "
      << "component permutation/sign transform is implemented in the current "
      << "FullShared read path. DirectGlobal vector reads remain out of scope "
      << "for this scalar prototype.\n";
   return true;
}

} // namespace

int main()
{
   using Shape1D =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 > >;
   using Shape2D =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 2 >,
         GaussLegendreShapeFunctions< 3 > >;
   using Shape3D =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 1 >,
         GaussLegendreShapeFunctions< 2 >,
         GaussLegendreShapeFunctions< 3 > >;

   using Space1D = TestFiniteElementSpace< 1, Shape1D >;
   using Space2D = TestFiniteElementSpace< 2, Shape2D >;
   using Space3D = TestFiniteElementSpace< 3, Shape3D >;

   bool success = true;

   success =
      RunOrientationMapCase<
         orders_to_num_dofs< typename Shape1D::orders > >(
            "1D shape (4)" ) &&
      success;
   success =
      RunOrientationMapCase<
         orders_to_num_dofs< typename Shape2D::orders > >(
            "2D anisotropic shape (3,4)" ) &&
      success;
   success =
      RunOrientationMapCase<
         orders_to_num_dofs< typename Shape3D::orders > >(
            "3D anisotropic shape (2,3,4)" ) &&
      success;
   success =
      RunFocusedAnisotropicSwapCase() &&
      success;

   success =
      RunScalarReadDofsPolicyCase< Space1D >(
         "1D shape (4)" ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space2D >(
         "2D anisotropic shape (3,4)" ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space3D >(
         "3D anisotropic shape (2,3,4)" ) &&
      success;

   success = RunVectorFaceReadAudit() && success;

   return success ? 0 : 1;
}
