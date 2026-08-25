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

template < Integer Dim, class Orientation = Permutation< Dim > >
struct TestFaceView
{
   using orientation_type = Orientation;

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
   using face_write_dofs_policy = FullSharedFaceWriteDofsPolicy;
};

using HeterogeneousComponent0Shape =
   TensorShapeFunctions<
      GaussLegendreShapeFunctions< 1 >,
      GaussLegendreShapeFunctions< 1 >,
      GaussLegendreShapeFunctions< 1 > >;
using HeterogeneousComponent1Shape =
   TensorShapeFunctions<
      GaussLegendreShapeFunctions< 3 >,
      GaussLegendreShapeFunctions< 2 >,
      GaussLegendreShapeFunctions< 2 > >;
using HeterogeneousVectorFE =
   FiniteElement<
      HyperCube< 3 >,
      VectorShapeFunctions<
         HeterogeneousComponent0Shape,
         HeterogeneousComponent1Shape > >;
using HeterogeneousVectorSpace =
   TestFiniteElementSpaceFromFiniteElement< 3, HeterogeneousVectorFE >;
using HeterogeneousDofShapes =
   typename HeterogeneousVectorFE::shape_functions::dof_shape;

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

Real VectorWriteBaselineValue(
   const Integer component,
   const size_t linear_index )
{
   return static_cast< Real >( -500000 + 1000 * ( component + 1 ) ) +
      static_cast< Real >( linear_index + 1 ) / 10.0;
}

template < WriteOp Op,
           typename KernelContext,
           typename FiniteElementSpace,
           typename Face,
           typename LocalDofs,
           typename GlobalDofs >
void ApplyFaceWriteOp(
   KernelContext & context,
   const FiniteElementSpace & fe_space,
   const Face & face,
   const LocalDofs & local_dofs,
   GlobalDofs & global_dofs )
{
   if constexpr ( Op == WriteAdd )
   {
      WriteAddDofs( context, fe_space, face, local_dofs, global_dofs );
   }
   else if constexpr ( Op == WriteSub )
   {
      WriteSubDofs( context, fe_space, face, local_dofs, global_dofs );
   }
   else
   {
      WriteDofs( context, fe_space, face, local_dofs, global_dofs );
   }
}

template < WriteOp Op >
const char * WriteOpName()
{
   if constexpr ( Op == WriteAdd )
   {
      return "WriteAdd";
   }
   else if constexpr ( Op == WriteSub )
   {
      return "WriteSub";
   }
   else
   {
      return "Write";
   }
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

template < WriteOp Op >
bool RunVectorWriteDofsPolicySmokeForOp()
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

   constexpr GlobalIndex num_elements = 2;
   constexpr GlobalIndex element_index = 1;
   constexpr size_t component_num_dofs = Product( ComponentShape{} );

   VectorSpace2DEqual fe_space{};
   Real * no_shared_memory = nullptr;

   const std::array< Permutation< 2 >, 5 > orientations{
      Permutation< 2 >{ { 1, 2 } },
      Permutation< 2 >{ { -1, 2 } },
      Permutation< 2 >{ { 1, -2 } },
      Permutation< 2 >{ { 2, 1 } },
      Permutation< 2 >{ { 2, -1 } } };

   auto local_dofs =
      MakeVectorDofs( DofShapes{}, std::make_index_sequence< 2 >{} );

   ConstexprLoop< 2 >( [&] ( auto component )
   {
      using ComponentDofShape = std::tuple_element_t< component, DofShapes >;
      auto & local_component = std::get< component >( local_dofs );
      UnitLoop< ComponentDofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, 2 > native_indices{
            static_cast< GlobalIndex >( k )... };
         local_component( k... ) =
            VectorIndexEncodedValue( component, native_indices, 0 ) / 13.0;
      });
   });

   bool success = true;
   Integer num_failures_reported = 0;

   for ( const auto & orientation : orientations )
   {
      std::array< Real, component_num_dofs * num_elements >
         full_shared_data_0{};
      std::array< Real, component_num_dofs * num_elements >
         full_shared_data_1{};
      std::array< Real, component_num_dofs * num_elements >
         direct_global_data_0{};
      std::array< Real, component_num_dofs * num_elements >
         direct_global_data_1{};
      std::array< Real, component_num_dofs * num_elements >
         direct_helper_data_0{};
      std::array< Real, component_num_dofs * num_elements >
         direct_helper_data_1{};

      for ( size_t i = 0; i < component_num_dofs * num_elements; ++i )
      {
         full_shared_data_0[ i ] = VectorWriteBaselineValue( 0, i );
         full_shared_data_1[ i ] = VectorWriteBaselineValue( 1, i );
         direct_global_data_0[ i ] = full_shared_data_0[ i ];
         direct_global_data_1[ i ] = full_shared_data_1[ i ];
         direct_helper_data_0[ i ] = full_shared_data_0[ i ];
         direct_helper_data_1[ i ] = full_shared_data_1[ i ];
      }

      auto full_shared_component_0 =
         MakeFIFOView(
            full_shared_data_0.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );
      auto full_shared_component_1 =
         MakeFIFOView(
            full_shared_data_1.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );
      auto direct_global_component_0 =
         MakeFIFOView(
            direct_global_data_0.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );
      auto direct_global_component_1 =
         MakeFIFOView(
            direct_global_data_1.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );
      auto direct_helper_component_0 =
         MakeFIFOView(
            direct_helper_data_0.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );
      auto direct_helper_component_1 =
         MakeFIFOView(
            direct_helper_data_1.data(),
            GlobalIndex{ 4 },
            GlobalIndex{ 4 },
            num_elements );

      auto full_shared_global_dofs =
         std::make_tuple( full_shared_component_0, full_shared_component_1 );
      auto direct_global_dofs =
         std::make_tuple( direct_global_component_0, direct_global_component_1 );
      auto direct_helper_dofs =
         std::make_tuple( direct_helper_component_0, direct_helper_component_1 );

      KernelContext< FullSharedSerialKernelConfiguration, 0 >
         full_shared_context( no_shared_memory );
      KernelContext< HostKernelConfiguration, 0 >
         direct_global_context( no_shared_memory );
      KernelContext< HostKernelConfiguration, 0 >
         direct_helper_context( no_shared_memory );
      const TestFaceView< 2 > face{ element_index, orientation };

      ApplyFaceWriteOp< Op >(
         full_shared_context,
         fe_space,
         face,
         local_dofs,
         full_shared_global_dofs );
      ApplyFaceWriteOp< Op >(
         direct_global_context,
         fe_space,
         face,
         local_dofs,
         direct_global_dofs );
      DirectGlobalSerialWriteVectorFaceDofs< Op >(
         direct_helper_context,
         fe_space,
         face,
         local_dofs,
         direct_helper_dofs );

      for ( size_t i = 0; i < component_num_dofs * num_elements; ++i )
      {
         const bool mismatch =
            std::abs( full_shared_data_0[ i ] - direct_global_data_0[ i ] ) >
               1e-12 ||
            std::abs( full_shared_data_1[ i ] - direct_global_data_1[ i ] ) >
               1e-12 ||
            std::abs( full_shared_data_0[ i ] - direct_helper_data_0[ i ] ) >
               1e-12 ||
            std::abs( full_shared_data_1[ i ] - direct_helper_data_1[ i ] ) >
               1e-12;

         if ( mismatch )
         {
            success = false;
            if ( num_failures_reported < 8 )
            {
               ++num_failures_reported;
               std::cout
                  << "Vector " << WriteOpName< Op >()
                  << "Dofs smoke mismatch: orientation=";
               PrintOrientation( orientation );
               std::cout
                  << ", linear index=" << i
                  << ", FullShared=(" << full_shared_data_0[ i ]
                  << "," << full_shared_data_1[ i ]
                  << "), DirectGlobal=(" << direct_global_data_0[ i ]
                  << "," << direct_global_data_1[ i ]
                  << "), DirectHelper=(" << direct_helper_data_0[ i ]
                  << "," << direct_helper_data_1[ i ] << ")\n";
            }
         }
      }
   }

   if ( success )
   {
      std::cout
         << "PASS vector " << WriteOpName< Op >()
         << "Dofs policy equivalence smoke: "
         << "FullShared vs public/default DirectGlobal vs direct helper\n";
   }

   return success;
}

bool RunVectorWriteDofsPolicySmoke()
{
   return RunVectorWriteDofsPolicySmokeForOp< Write >() &&
      RunVectorWriteDofsPolicySmokeForOp< WriteAdd >() &&
      RunVectorWriteDofsPolicySmokeForOp< WriteSub >();
}

template < class Orientation >
bool RunHeterogeneousVectorReadForOrientation(
   const Orientation & orientation )
{
   constexpr GlobalIndex num_elements = 2;
   constexpr GlobalIndex element_index = 1;
   constexpr size_t NumDofs0 = Product( HeterogeneousComponent0Shape::dof_shape{} );
   constexpr size_t NumDofs1 = Product( HeterogeneousComponent1Shape::dof_shape{} );

   std::array< Real, NumDofs0 * num_elements > global_data_0{};
   std::array< Real, NumDofs1 * num_elements > global_data_1{};
   for ( size_t i = 0; i < global_data_0.size(); ++i )
   {
      global_data_0[i] = static_cast< Real >( 1000 + i );
   }
   for ( size_t i = 0; i < global_data_1.size(); ++i )
   {
      global_data_1[i] = static_cast< Real >( 2000 + i );
   }

   auto component_0 = MakeFIFOView(
      global_data_0.data(),
      GlobalIndex{2},
      GlobalIndex{2},
      GlobalIndex{2},
      num_elements );
   auto component_1 = MakeFIFOView(
      global_data_1.data(),
      GlobalIndex{4},
      GlobalIndex{3},
      GlobalIndex{3},
      num_elements );
   auto global_dofs = std::make_tuple( component_0, component_1 );

   HeterogeneousVectorSpace fe_space{};
   Real * no_shared_memory = nullptr;
   KernelContext< FullSharedSerialKernelConfiguration, 0 >
      full_shared_context( no_shared_memory );
   KernelContext< HostKernelConfiguration, 0 >
      direct_global_context( no_shared_memory );
   const TestFaceView< 3, Orientation > face{
      element_index,
      orientation };

   const auto full_shared = ReadDofs(
      full_shared_context,
      fe_space,
      face,
      global_dofs );
   const auto direct_global = ReadDofs(
      direct_global_context,
      fe_space,
      face,
      global_dofs );

   bool success = true;
   ConstexprLoop< 2 >( [&] ( auto component )
   {
      using ComponentDofShape =
         std::tuple_element_t< component, HeterogeneousDofShapes >;
      UnitLoop< ComponentDofShape >( [&] ( auto... k )
      {
         success =
            std::get< component >( full_shared )( k... ) ==
               std::get< component >( direct_global )( k... ) &&
            success;
      });
   });
   if ( !success )
   {
      std::cerr
         << "Heterogeneous vector FullShared and DirectGlobal reads differ.\n";
   }
   return success;
}

template < WriteOp Op, class Orientation >
bool RunHeterogeneousVectorWriteForOrientation(
   const Orientation & orientation )
{
   constexpr GlobalIndex num_elements = 2;
   constexpr GlobalIndex element_index = 1;
   constexpr size_t NumDofs0 = Product( HeterogeneousComponent0Shape::dof_shape{} );
   constexpr size_t NumDofs1 = Product( HeterogeneousComponent1Shape::dof_shape{} );

   auto local_dofs = MakeVectorDofs(
      HeterogeneousDofShapes{},
      std::make_index_sequence< 2 >{} );
   ConstexprLoop< 2 >( [&] ( auto component )
   {
      using ComponentDofShape =
         std::tuple_element_t< component, HeterogeneousDofShapes >;
      auto & local_component = std::get< component >( local_dofs );
      UnitLoop< ComponentDofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, 3 > indices{
            static_cast< GlobalIndex >( k )... };
         local_component( k... ) = static_cast< Real >(
            10000 * ( component + 1 ) +
            FlattenMultiIndex< ComponentDofShape >( indices ) );
      });
   });

   std::array< Real, NumDofs0 * num_elements > full_shared_data_0{};
   std::array< Real, NumDofs1 * num_elements > full_shared_data_1{};
   std::array< Real, NumDofs0 * num_elements > direct_global_data_0{};
   std::array< Real, NumDofs1 * num_elements > direct_global_data_1{};
   for ( size_t i = 0; i < full_shared_data_0.size(); ++i )
   {
      full_shared_data_0[i] = static_cast< Real >( -1000 - i );
      direct_global_data_0[i] = full_shared_data_0[i];
   }
   for ( size_t i = 0; i < full_shared_data_1.size(); ++i )
   {
      full_shared_data_1[i] = static_cast< Real >( -2000 - i );
      direct_global_data_1[i] = full_shared_data_1[i];
   }

   auto full_shared_global_dofs = std::make_tuple(
      MakeFIFOView(
         full_shared_data_0.data(),
         GlobalIndex{2},
         GlobalIndex{2},
         GlobalIndex{2},
         num_elements ),
      MakeFIFOView(
         full_shared_data_1.data(),
         GlobalIndex{4},
         GlobalIndex{3},
         GlobalIndex{3},
         num_elements ) );
   auto direct_global_dofs = std::make_tuple(
      MakeFIFOView(
         direct_global_data_0.data(),
         GlobalIndex{2},
         GlobalIndex{2},
         GlobalIndex{2},
         num_elements ),
      MakeFIFOView(
         direct_global_data_1.data(),
         GlobalIndex{4},
         GlobalIndex{3},
         GlobalIndex{3},
         num_elements ) );

   HeterogeneousVectorSpace fe_space{};
   Real * no_shared_memory = nullptr;
   KernelContext< FullSharedSerialKernelConfiguration, 0 >
      full_shared_context( no_shared_memory );
   KernelContext< HostKernelConfiguration, 0 >
      direct_global_context( no_shared_memory );
   const TestFaceView< 3, Orientation > face{
      element_index,
      orientation };

   ApplyFaceWriteOp< Op >(
      full_shared_context,
      fe_space,
      face,
      local_dofs,
      full_shared_global_dofs );
   ApplyFaceWriteOp< Op >(
      direct_global_context,
      fe_space,
      face,
      local_dofs,
      direct_global_dofs );

   const bool success =
      full_shared_data_0 == direct_global_data_0 &&
      full_shared_data_1 == direct_global_data_1;
   if ( !success )
   {
      std::cerr
         << "Heterogeneous vector FullShared and DirectGlobal writes differ.\n";
   }
   return success;
}

template < class Orientation >
bool RunHeterogeneousVectorPoliciesForOrientation(
   const Orientation & orientation )
{
   bool success = true;
   success = RunHeterogeneousVectorReadForOrientation(
      orientation ) && success;
   success = RunHeterogeneousVectorWriteForOrientation< Write >(
      orientation ) && success;
   success = RunHeterogeneousVectorWriteForOrientation< WriteAdd >(
      orientation ) && success;
   success = RunHeterogeneousVectorWriteForOrientation< WriteSub >(
      orientation ) && success;
   return success;
}

bool RunHeterogeneousVectorPolicyCoverage()
{
   bool success = true;
   for ( Integer swap_equal_axes = 0;
         swap_equal_axes < 2;
         ++swap_equal_axes )
   {
      for ( Integer reversal_mask = 0;
            reversal_mask < 8;
            ++reversal_mask )
      {
         Permutation< 3 > orientation = swap_equal_axes == 0
            ? Permutation< 3 >{ { 1, 2, 3 } }
            : Permutation< 3 >{ { 1, 3, 2 } };
         for ( Integer axis = 0; axis < 3; ++axis )
         {
            if ( reversal_mask & ( 1 << axis ) )
            {
               orientation( axis ) = -orientation( axis );
            }
         }
         success = RunHeterogeneousVectorPoliciesForOrientation(
            orientation ) && success;
      }
   }

   success = RunHeterogeneousVectorPoliciesForOrientation(
      IdentityOrientation< 3 >{} ) && success;
   success = RunHeterogeneousVectorPoliciesForOrientation(
      MakeTensorProductOrientation(
         IdentityOrientation< 1 >{},
         Permutation< 2 >{ { 2, -1 } } ) ) && success;

   if ( success )
   {
      std::cout
         << "PASS heterogeneous vector face I/O policy coverage: all "
         << "supported flat, static, and mixed structured orientations\n";
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
   return RunVectorFaceReadAudit() &&
         RunVectorReadDofsPolicySmoke() &&
         RunVectorWriteDofsPolicySmoke() &&
         RunHeterogeneousVectorPolicyCoverage()
      ? 0
      : 1;
}
