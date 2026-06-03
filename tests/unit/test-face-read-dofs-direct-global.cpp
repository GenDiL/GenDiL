// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
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
   public HostKernelConfiguration
{
   using face_read_dofs_policy = DirectGlobalFaceReadDofsPolicy;
};

struct FullSharedSerialKernelConfiguration :
   public HostKernelConfiguration
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
};

template < Integer SpaceDim, typename FiniteElement >
struct TestFiniteElementSpaceFromFiniteElement
{
   static constexpr Integer Dim = SpaceDim;
   using finite_element_type = FiniteElement;
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

template < Integer Dim >
bool ExpectedShapeCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   return OrientedTensorDofShapeIsCompatible( sizes, orientation );
}

template < Integer Dim >
std::array< GlobalIndex, Dim > ShapeCompatibleNativeIndex(
   const std::array< GlobalIndex, Dim > & reference_indices,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   std::array< GlobalIndex, Dim > native_indices{};
   for ( Integer native_dim = 0; native_dim < Dim; ++native_dim )
   {
      const LocalIndex o = orientation( native_dim );
      const Integer reference_axis =
         static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
      native_indices[ native_dim ] =
         o > 0
            ? reference_indices[ reference_axis ]
            : static_cast< GlobalIndex >(
                 sizes[ native_dim ] - 1 -
                 reference_indices[ reference_axis ] );
   }
   return native_indices;
}

template < typename DofShape >
bool RunShapeCompatibilityClassificationCase( const char * case_name )
{
   constexpr Integer dim = DofShape::size();
   const auto sizes = to_array( DofShape{} );
   const auto orientations = MakeAllSignedPermutations< dim >();

   bool success = true;
   Integer num_failures_reported = 0;
   for ( const auto & orientation : orientations )
   {
      const bool expected =
         ExpectedShapeCompatible( sizes, orientation );
      const bool actual =
         FaceReadDofsOrientationIsShapeCompatible(
            sizes,
            orientation );

      if ( actual != expected )
      {
         success = false;
         if ( num_failures_reported < 8 )
         {
            ++num_failures_reported;
            std::cout
               << "Shape-compatibility mismatch in " << case_name
               << ": orientation=";
            PrintOrientation( orientation );
            std::cout << ", expected=" << expected
                      << ", actual=" << actual
                      << "\n";
         }
      }
   }

   if ( success )
   {
      std::cout
         << "PASS shape-compatible orientation classification: "
         << case_name << "\n";
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

template < typename Space >
bool RunOrientedGlobalViewCase(
   const char * case_name,
   const std::vector< Permutation< Space::Dim > > & orientations )
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

   const auto identity = MakeReferencePermutation< dim >();
   bool success = true;
   Integer num_failures_reported = 0;

   for ( const auto & orientation : orientations )
   {
      if ( !FaceReadDofsOrientationIsShapeCompatible(
              dof_sizes,
              orientation ) )
      {
         success = false;
         std::cout
            << "Oriented view case was given unsupported orientation in "
            << case_name << ": orientation=";
         PrintOrientation( orientation );
         continue;
      }

      const auto oriented_global_dofs =
         MakeOrientedGlobalDofView(
            global_dofs,
            element_index,
            dof_sizes,
            orientation );

      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, dim > reference_indices{
            static_cast< GlobalIndex >( k )... };
         const auto native_indices =
            ShapeCompatibleNativeIndex(
               reference_indices,
               dof_sizes,
               orientation );
         const Real expected =
            IndexEncodedValue( native_indices, element_index );
         const Real actual = oriented_global_dofs( k... );
         const bool identity_match =
            orientation != identity || native_indices == reference_indices;

         if ( std::abs( expected - actual ) > 1e-12 ||
              !identity_match )
         {
            success = false;
            if ( num_failures_reported < 8 )
            {
               ++num_failures_reported;
               std::cout
                  << "Oriented global view mismatch in " << case_name
                  << ": orientation=";
               PrintOrientation( orientation );
               std::cout << ", reference=";
               PrintIndex( reference_indices );
               std::cout << ", native=";
               PrintIndex( native_indices );
               std::cout << ", expected=" << expected
                         << ", actual=" << actual
                         << ", identity_match=" << identity_match
                         << "\n";
            }
         }
      });
   }

   if ( success )
   {
      std::cout << "PASS oriented global view: " << case_name << "\n";
   }
   return success;
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
bool RunScalarReadDofsPolicyCase(
   const char * case_name,
   const std::vector< Permutation< Space::Dim > > & orientations )
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
   KernelContext< FullSharedSerialKernelConfiguration, 0 >
      full_shared_context( no_shared_memory );
   KernelContext< HostKernelConfiguration, 0 >
      default_context( no_shared_memory );

   bool success = true;
   for ( const auto & orientation : orientations )
   {
      if ( !FaceReadDofsOrientationIsShapeCompatible(
              dof_sizes,
              orientation ) )
      {
         success = false;
         std::cout
            << "Scalar policy case was given unsupported orientation in "
            << case_name << ": orientation=";
         PrintOrientation( orientation );
         continue;
      }

      const TestFaceView< dim > face{ element_index, orientation };
      const auto full_shared =
         ReadDofs(
            full_shared_context,
            fe_space,
            face,
            global_dofs );
      const auto direct_global =
         ReadDofs(
            default_context,
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

template < typename Space >
bool RunScalarWriteDofsSupportedCase(
   const char * case_name,
   const std::vector< Permutation< Space::Dim > > & orientations )
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

   Space fe_space{};
   Real * no_shared_memory = nullptr;
   KernelContext< HostKernelConfiguration, 0 >
      context( no_shared_memory );

   bool success = true;
   Integer num_failures_reported = 0;
   for ( const auto & orientation : orientations )
   {
      if ( !OrientedTensorDofShapeIsCompatible(
              dof_sizes,
              orientation ) )
      {
         success = false;
         std::cout
            << "Scalar write case was given unsupported orientation in "
            << case_name << ": orientation=";
         PrintOrientation( orientation );
         std::cout << "\n";
         continue;
      }

      std::vector< Real > global_data( num_values, -1.0 );
      auto global_dofs = MakeGlobalDofView< Space >( global_data, global_sizes );
      auto local_dofs = MakeSerialRecursiveArray< Real >( DofShape{} );

      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, dim > reference_indices{
            static_cast< GlobalIndex >( k )... };
         local_dofs( k... ) =
            IndexEncodedValue( reference_indices, 0 );
      });

      const TestFaceView< dim > face{ element_index, orientation };
      WriteDofs(
         context,
         fe_space,
         face,
         local_dofs,
         global_dofs );

      UnitLoop< DofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, dim > reference_indices{
            static_cast< GlobalIndex >( k )... };
         const auto native_indices =
            ShapeCompatibleNativeIndex(
               reference_indices,
               dof_sizes,
               orientation );
         const Real expected =
            IndexEncodedValue( reference_indices, 0 );
         const Real actual =
            FaceReadDofsGlobalValueAt(
               global_dofs,
               native_indices,
               element_index );

         if ( std::abs( expected - actual ) > 1e-12 )
         {
            success = false;
            if ( num_failures_reported < 8 )
            {
               ++num_failures_reported;
               std::cout
                  << "WriteDofs orientation mismatch in " << case_name
                  << ": orientation=";
               PrintOrientation( orientation );
               std::cout << ", reference=";
               PrintIndex( reference_indices );
               std::cout << ", native=";
               PrintIndex( native_indices );
               std::cout << ", expected=" << expected
                         << ", actual=" << actual
                         << "\n";
            }
         }
      });
   }

   if ( success )
   {
      std::cout << "PASS scalar WriteDofs supported orientation smoke: "
                << case_name << "\n";
   }
   return success;
}

template < typename DofShape >
bool RunUnsupportedOrientationCase(
   const char * case_name,
   const std::vector< Permutation< DofShape::size() > > & orientations )
{
   const auto dof_sizes = to_array( DofShape{} );
   bool success = true;

   for ( const auto & orientation : orientations )
   {
      const bool supported =
         FaceReadDofsOrientationIsShapeCompatible(
            dof_sizes,
            orientation );
      if ( supported )
      {
         success = false;
         std::cout
            << "Unsupported orientation classified as supported in "
            << case_name << ": orientation=";
         PrintOrientation( orientation );
      }
   }

   if ( success )
   {
      std::cout
         << "PASS unsupported oriented tensor DOF orientation classification: "
         << case_name << "\n";
   }

   return success;
}

} // namespace

int main()
{
   static_assert(
      std::is_same_v<
         face_read_dofs_policy_t< HostKernelConfiguration >,
         DirectGlobalFaceReadDofsPolicy >,
      "Default scalar face ReadDofs policy should be DirectGlobal." );
   static_assert(
      std::is_same_v<
         face_read_dofs_policy_t< FullSharedSerialKernelConfiguration >,
         FullSharedFaceReadDofsPolicy >,
      "Explicit FullShared face ReadDofs override should be preserved." );
   static_assert(
      std::is_same_v<
         face_read_dofs_policy_t< DirectGlobalSerialKernelConfiguration >,
         DirectGlobalFaceReadDofsPolicy >,
      "Explicit DirectGlobal face ReadDofs override should be preserved." );

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
   using Shape2DEqual =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 > >;

   using Space1D = TestFiniteElementSpace< 1, Shape1D >;
   using Space2D = TestFiniteElementSpace< 2, Shape2D >;
   using Space3D = TestFiniteElementSpace< 3, Shape3D >;
   using Space2DEqual = TestFiniteElementSpace< 2, Shape2DEqual >;

   const std::vector< Permutation< 1 > > orientations_1d{
      Permutation< 1 >{ { 1 } },
      Permutation< 1 >{ { -1 } } };
   const std::vector< Permutation< 2 > > anisotropic_2d_identity{
      Permutation< 2 >{ { 1, 2 } } };
   const std::vector< Permutation< 3 > > anisotropic_3d_identity{
      Permutation< 3 >{ { 1, 2, 3 } } };
   const std::vector< Permutation< 2 > > anisotropic_2d_flips{
      Permutation< 2 >{ { -1, 2 } },
      Permutation< 2 >{ { 1, -2 } },
      Permutation< 2 >{ { -1, -2 } } };
   const std::vector< Permutation< 3 > > anisotropic_3d_flips{
      Permutation< 3 >{ { -1, 2, 3 } },
      Permutation< 3 >{ { 1, -2, 3 } },
      Permutation< 3 >{ { 1, 2, -3 } },
      Permutation< 3 >{ { -1, -2, -3 } } };
   const std::vector< Permutation< 2 > > equal_extent_identity_flips{
      Permutation< 2 >{ { 1, 2 } },
      Permutation< 2 >{ { -1, 2 } },
      Permutation< 2 >{ { 1, -2 } },
      Permutation< 2 >{ { -1, -2 } } };
   const std::vector< Permutation< 2 > > equal_extent_swaps{
      Permutation< 2 >{ { 2, 1 } },
      Permutation< 2 >{ { 2, -1 } },
      Permutation< 2 >{ { -2, 1 } },
      Permutation< 2 >{ { -2, -1 } } };
   const std::vector< Permutation< 2 > > anisotropic_2d_swaps{
      Permutation< 2 >{ { 2, 1 } },
      Permutation< 2 >{ { 2, -1 } },
      Permutation< 2 >{ { -2, 1 } },
      Permutation< 2 >{ { -2, -1 } } };
   const std::vector< Permutation< 3 > > anisotropic_3d_swaps{
      Permutation< 3 >{ { 2, 1, 3 } },
      Permutation< 3 >{ { 1, 3, 2 } },
      Permutation< 3 >{ { -3, 2, 1 } } };

   bool success = true;

   success =
      RunShapeCompatibilityClassificationCase<
         orders_to_num_dofs< typename Shape1D::orders > >(
            "1D shape (4)" ) &&
      success;
   success =
      RunShapeCompatibilityClassificationCase<
         orders_to_num_dofs< typename Shape2D::orders > >(
            "2D anisotropic shape (3,4)" ) &&
      success;
   success =
      RunShapeCompatibilityClassificationCase<
         orders_to_num_dofs< typename Shape3D::orders > >(
            "3D anisotropic shape (2,3,4)" ) &&
      success;
   success =
      RunShapeCompatibilityClassificationCase<
         orders_to_num_dofs< typename Shape2DEqual::orders > >(
            "2D equal shape (4,4)" ) &&
      success;

   success =
      RunOrientedGlobalViewCase< Space1D >(
         "1D identity and flip shape (4)",
         orientations_1d ) &&
      success;
   success =
      RunOrientedGlobalViewCase< Space2D >(
         "2D anisotropic identity shape (3,4)",
         anisotropic_2d_identity ) &&
      success;
   success =
      RunOrientedGlobalViewCase< Space3D >(
         "3D anisotropic identity shape (2,3,4)",
         anisotropic_3d_identity ) &&
      success;
   success =
      RunOrientedGlobalViewCase< Space2DEqual >(
         "2D equal-extent identity/flips shape (4,4)",
         equal_extent_identity_flips ) &&
      success;
   success =
      RunOrientedGlobalViewCase< Space2DEqual >(
         "2D equal-extent swaps shape (4,4)",
         equal_extent_swaps ) &&
      success;

   success =
      RunScalarReadDofsPolicyCase< Space1D >(
         "1D identity and flip shape (4)",
         orientations_1d ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space2D >(
         "2D anisotropic identity shape (3,4)",
         anisotropic_2d_identity ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space3D >(
         "3D anisotropic identity shape (2,3,4)",
         anisotropic_3d_identity ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space2DEqual >(
         "2D equal-extent identity/flips shape (4,4)",
         equal_extent_identity_flips ) &&
      success;
   success =
      RunScalarReadDofsPolicyCase< Space2DEqual >(
         "2D equal-extent swaps shape (4,4)",
         equal_extent_swaps ) &&
      success;

   success =
      RunScalarWriteDofsSupportedCase< Space1D >(
         "1D identity and flip shape (4)",
         orientations_1d ) &&
      success;
   success =
      RunScalarWriteDofsSupportedCase< Space2D >(
         "2D anisotropic identity shape (3,4)",
         anisotropic_2d_identity ) &&
      success;
   success =
      RunScalarWriteDofsSupportedCase< Space2DEqual >(
         "2D equal-extent identity/flips shape (4,4)",
         equal_extent_identity_flips ) &&
      success;
   success =
      RunScalarWriteDofsSupportedCase< Space2DEqual >(
         "2D equal-extent swaps shape (4,4)",
         equal_extent_swaps ) &&
      success;

   success =
      RunUnsupportedOrientationCase<
         orders_to_num_dofs< typename Shape2D::orders > >(
            "2D anisotropic flipped axes shape (3,4)",
            anisotropic_2d_flips ) &&
      success;
   success =
      RunUnsupportedOrientationCase<
         orders_to_num_dofs< typename Shape3D::orders > >(
            "3D anisotropic flipped axes shape (2,3,4)",
            anisotropic_3d_flips ) &&
      success;
   success =
      RunUnsupportedOrientationCase<
         orders_to_num_dofs< typename Shape2D::orders > >(
            "2D anisotropic swapped axes shape (3,4)",
            anisotropic_2d_swaps ) &&
      success;
   success =
      RunUnsupportedOrientationCase<
         orders_to_num_dofs< typename Shape3D::orders > >(
            "3D anisotropic swapped axes shape (2,3,4)",
            anisotropic_3d_swaps ) &&
      success;

   return success ? 0 : 1;
}
