// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

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

template < Integer SpaceDim, typename FiniteElement >
struct TestFiniteElementSpaceFromFiniteElement
{
   static constexpr Integer Dim = SpaceDim;
   using finite_element_type = FiniteElement;
};

struct FullSharedSerialKernelConfiguration :
   public HostKernelConfiguration
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
};

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

Real VectorIndexEncodedValue(
   const Integer component,
   const std::array< GlobalIndex, 2 > & native_indices,
   const GlobalIndex element_index )
{
   Real value = static_cast< Real >( 100000 * ( component + 1 ) );
   value += static_cast< Real >( 10000 * ( element_index + 1 ) );
   value += static_cast< Real >( native_indices[ 0 ] + 1 );
   value += static_cast< Real >( 10 * ( native_indices[ 1 ] + 1 ) );
   return value;
}

bool RunVectorReadDofsPolicySmoke()
{
   using Shape2DEqual =
      TensorShapeFunctions<
         GaussLegendreShapeFunctions< 3 >,
         GaussLegendreShapeFunctions< 3 > >;
   using VectorFE2DEqual =
      FiniteElement<
         HyperCube< 2 >,
         VectorShapeFunctions< Shape2DEqual, Shape2DEqual > >;
   using VectorSpace2DEqual =
      TestFiniteElementSpaceFromFiniteElement< 2, VectorFE2DEqual >;
   using DofShapes =
      typename VectorSpace2DEqual::finite_element_type::
         shape_functions::dof_shape;
   using ComponentShape = std::tuple_element_t< 0, DofShapes >;

   static_assert(
      FaceSpeedOfLightRequiredSharedMemory<
         FaceSoLType::ReadCell,
         HostKernelConfiguration,
         VectorSpace2DEqual >::value == 0,
      "Vector default DirectGlobal face reads should not require shared read-side storage." );
   static_assert(
      FaceSpeedOfLightRequiredSharedMemory<
         FaceSoLType::ReadCell,
         FullSharedSerialKernelConfiguration,
         VectorSpace2DEqual >::value == 0,
      "Register-only FullShared vector face reads should not require shared arena storage." );

   constexpr GlobalIndex num_elements = 2;
   constexpr GlobalIndex element_index = 1;
   constexpr size_t component_num_dofs = Product( ComponentShape{} );

   std::array< Real, component_num_dofs * num_elements > global_data_0{};
   std::array< Real, component_num_dofs * num_elements > global_data_1{};

   auto component_0 =
      MakeFIFOView(
         global_data_0.data(),
         GlobalIndex{ 4 },
         GlobalIndex{ 4 },
         num_elements );
   auto component_1 =
      MakeFIFOView(
         global_data_1.data(),
         GlobalIndex{ 4 },
         GlobalIndex{ 4 },
         num_elements );

   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< ComponentShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, 2 > native_indices{
            static_cast< GlobalIndex >( k )... };
         component_0( k..., element ) =
            VectorIndexEncodedValue( 0, native_indices, element );
         component_1( k..., element ) =
            VectorIndexEncodedValue( 1, native_indices, element );
      });
   }

   auto global_dofs = std::make_tuple( component_0, component_1 );
   VectorSpace2DEqual fe_space{};
   Real * no_shared_memory = nullptr;
   KernelContext< FullSharedSerialKernelConfiguration, 0 >
      full_shared_context( no_shared_memory );
   KernelContext< HostKernelConfiguration, 0 >
      direct_global_context( no_shared_memory );

   const std::array< Permutation< 2 >, 5 > orientations{
      Permutation< 2 >{ { 1, 2 } },
      Permutation< 2 >{ { -1, 2 } },
      Permutation< 2 >{ { 1, -2 } },
      Permutation< 2 >{ { 2, 1 } },
      Permutation< 2 >{ { 2, -1 } } };

   bool success = true;
   Integer num_failures_reported = 0;

   for ( const auto & orientation : orientations )
   {
      const TestFaceView< 2 > face{ element_index, orientation };
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

      UnitLoop< ComponentShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, 2 > reference_indices{
            static_cast< GlobalIndex >( k )... };

         const Real full_value_0 = std::get< 0 >( full_shared )( k... );
         const Real direct_value_0 = std::get< 0 >( direct_global )( k... );
         const Real full_value_1 = std::get< 1 >( full_shared )( k... );
         const Real direct_value_1 = std::get< 1 >( direct_global )( k... );

         if ( std::abs( full_value_0 - direct_value_0 ) > 1e-12 ||
              std::abs( full_value_1 - direct_value_1 ) > 1e-12 )
         {
            success = false;
            if ( num_failures_reported < 8 )
            {
               ++num_failures_reported;
               std::cout
                  << "Vector ReadDofs smoke mismatch: orientation=";
               PrintOrientation( orientation );
               std::cout << ", reference=";
               PrintIndex( reference_indices );
               std::cout << ", FullShared=(" << full_value_0
                         << "," << full_value_1
                         << "), DirectGlobal=(" << direct_value_0
                         << "," << direct_value_1 << ")\n";
            }
         }
      });
   }

   if ( success )
   {
      std::cout
         << "PASS vector ReadDofs policy equivalence smoke: "
         << "2D equal-extent component-wise scalar orientation\n";
   }

   return success;
}

bool RunVectorFaceReadAudit()
{
   std::cout
      << "Vector face ReadDofs DirectGlobal audit: current vector face reads "
      << "apply the scalar orientation independently per component; no "
      << "component permutation/sign transform is implemented in the current "
      << "FullShared read path. DirectGlobal vector reads are therefore "
      << "validated as component-wise scalar reads in this test.\n";
   return true;
}

} // namespace

int main()
{
   return RunVectorFaceReadAudit() && RunVectorReadDofsPolicySmoke()
      ? 0
      : 1;
}
