// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace raw_coo_weight_test {

struct WeightedRestriction
{
   using dof_shape_type = std::index_sequence< 1 >;
   static constexpr gendil::Integer tensor_dim = 1;
};

inline gendil::GlobalIndex GetNumberOfLocalDofs(
   const WeightedRestriction & )
{
   return 1;
}

inline gendil::GlobalIndex GetNumberOfGlobalDofs(
   const WeightedRestriction & )
{
   return 1;
}

inline gendil::GlobalIndex GetAlgebraicDofExtent(
   const WeightedRestriction & )
{
   return 1;
}

template < typename Visitor >
void ForEachRestrictionEntry(
   const WeightedRestriction &,
   const gendil::GlobalIndex,
   const std::array< gendil::GlobalIndex, 1 > &,
   Visitor && visitor )
{
   std::forward< Visitor >( visitor )(
      0,
      gendil::Real{ 2 } );
}

struct WeightedSpace
{
   using restriction_type = WeightedRestriction;
};

} // namespace raw_coo_weight_test

namespace gendil {

template <>
inline constexpr size_t static_restriction_entry_count_v<
   raw_coo_weight_test::WeightedRestriction > = 1;

} // namespace gendil

using namespace gendil;

static_assert(
   TensorElementDoFRestriction<
      raw_coo_weight_test::WeightedRestriction > );
static_assert(
   !IsRawCOOCellAssemblySpace<
      raw_coo_weight_test::WeightedSpace >::value,
   "RawCOO coordinate insertion must reject a one-entry weighted row until weight handling is implemented." );

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

bool CheckRawTripletRangesAndFinite(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer )
{
   bool success = true;
   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      success = Check(
         ReadHost( buffer.rows )[i] < buffer.num_rows,
         "Raw COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         ReadHost( buffer.cols )[i] < buffer.num_cols,
         "Raw COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( ReadHost( buffer.values )[i] ),
         "Raw COO emitted a non-finite value." ) && success;
   }
   return success;
}

template < typename FESpace >
bool CheckScalar1DRawCellSlotCoordinates(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer,
   const FESpace & fe_space )
{
   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );

   bool success = true;
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      for ( GlobalIndex local_col = 0; local_col < local_dofs; ++local_col )
      {
         const std::array< GlobalIndex, 1 > col_indices{ local_col };
         const GlobalIndex expected_col =
            GetGlobalDofIndex( fe_space, element, col_indices );

         for ( GlobalIndex local_row = 0; local_row < local_dofs; ++local_row )
         {
            const std::array< GlobalIndex, 1 > row_indices{ local_row };
            const GlobalIndex expected_row =
               GetGlobalDofIndex( fe_space, element, row_indices );
            const GlobalIndex raw_index =
               element * local_dofs * local_dofs +
               local_col * local_dofs +
               local_row;

            success = Check(
               raw_index < buffer.nnz_raw,
               "Raw COO slot coordinate test exceeded the triplet buffer." ) && success;
            success = Check(
               ReadHost( buffer.rows )[raw_index] == expected_row,
               "Raw COO row slot does not match element-local test DoF order." ) && success;
            success = Check(
               ReadHost( buffer.cols )[raw_index] == expected_col,
               "Raw COO column slot does not match element-local trial DoF order." ) && success;
         }
      }
   }

   return success;
}

template < typename FESpace >
bool CheckVectorRawCellSlotCoordinates(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer,
   const FESpace & fe_space )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   static_assert(
      is_vector_shape_functions_v< ShapeFunctions >,
      "CheckVectorRawCellSlotCoordinates requires a vector FE space." );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   const GlobalIndex block_entries = local_dofs * local_dofs;

   bool success = true;
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      std::vector< GlobalIndex > expected_globals(
         static_cast< size_t >( local_dofs ),
         0 );
      std::vector< bool > seen(
         static_cast< size_t >( local_dofs ),
         false );

      ForEachLocalDof(
         fe_space,
         [&] ( const auto component, const auto & indices )
         {
            const GlobalIndex local_id =
               FlattenLocalDof( fe_space, component, indices );
            success = Check(
               local_id < local_dofs,
               "Vector Raw COO local DoF flattening exceeded the block size." ) && success;

            if ( local_id < local_dofs )
            {
               expected_globals[static_cast< size_t >( local_id )] =
                  GetGlobalDofIndex(
                     fe_space,
                     component,
                     element,
                     indices );
               seen[static_cast< size_t >( local_id )] = true;
            }
         });

      for ( const bool was_seen : seen )
      {
         success = Check(
            was_seen,
            "Vector Raw COO component-major local DoF traversal skipped a slot." ) && success;
      }

      for ( GlobalIndex local_col = 0; local_col < local_dofs; ++local_col )
      {
         for ( GlobalIndex local_row = 0; local_row < local_dofs; ++local_row )
         {
            const GlobalIndex raw_index =
               element * block_entries +
               local_col * local_dofs +
               local_row;

            success = Check(
               raw_index < buffer.nnz_raw,
               "Vector Raw COO slot coordinate test exceeded the triplet buffer." ) && success;
            success = Check(
               ReadHost( buffer.rows )[raw_index] ==
                  expected_globals[static_cast< size_t >( local_row )],
               "Vector Raw COO row slot does not match component-major test DoF order." ) && success;
            success = Check(
               ReadHost( buffer.cols )[raw_index] ==
                  expected_globals[static_cast< size_t >( local_col )],
               "Vector Raw COO column slot does not match component-major trial DoF order." ) && success;
         }
      }
   }

   return success;
}

bool HasDuplicateCoordinate(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer )
{
   std::vector< std::pair< GlobalIndex, GlobalIndex > > coordinates;
   coordinates.reserve( static_cast< size_t >( buffer.nnz_raw ) );

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      coordinates.emplace_back( ReadHost( buffer.rows )[i], ReadHost( buffer.cols )[i] );
   }

   std::sort( coordinates.begin(), coordinates.end() );
   return std::adjacent_find( coordinates.begin(), coordinates.end() ) !=
      coordinates.end();
}

template < typename Backend >
bool CheckCanonicalCOOSortedUnique(
   const COOMatrix< Real, GlobalIndex, Backend > & matrix )
{
   bool success = true;
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      success = Check(
         ReadHost( matrix.rows )[i] < matrix.num_rows,
         "Canonical COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         ReadHost( matrix.cols )[i] < matrix.num_cols,
         "Canonical COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( ReadHost( matrix.values )[i] ),
         "Canonical COO emitted a non-finite value." ) && success;
   }

   for ( GlobalIndex i = 1; i < matrix.nnz; ++i )
   {
      const bool ordered =
         ReadHost( matrix.rows )[i - 1] < ReadHost( matrix.rows )[i] ||
         ( ReadHost( matrix.rows )[i - 1] == ReadHost( matrix.rows )[i] &&
           ReadHost( matrix.cols )[i - 1] < ReadHost( matrix.cols )[i] );

      success = Check(
         ordered,
         "Canonical COO entries are not strictly sorted and unique." ) && success;
   }

   return success;
}

template < typename ActualBackend, typename ExpectedBackend >
bool CheckCOOMatricesEqual(
   const COOMatrix< Real, GlobalIndex, ActualBackend > & actual,
   const COOMatrix< Real, GlobalIndex, ExpectedBackend > & expected,
   const char * message )
{
   bool success = true;
   success = Check(
      actual.num_rows == expected.num_rows,
      message ) && success;
   success = Check(
      actual.num_cols == expected.num_cols,
      message ) && success;
   success = Check(
      actual.nnz == expected.nnz,
      message ) && success;

   const GlobalIndex nnz =
      actual.nnz < expected.nnz ? actual.nnz : expected.nnz;
   for ( GlobalIndex i = 0; i < nnz; ++i )
   {
      success = Check(
         ReadHost( actual.rows )[i] == ReadHost( expected.rows )[i],
         message ) && success;
      success = Check(
         ReadHost( actual.cols )[i] == ReadHost( expected.cols )[i],
         message ) && success;
      success = Check(
         Near( ReadHost( actual.values )[i], ReadHost( expected.values )[i] ),
         message ) && success;
   }

   return success;
}

template <
   bool IncludeCellTerms,
   bool IncludeBoundaryFaceTerms,
   bool IncludeInteriorFaceTerms,
   typename FESpace >
GlobalIndex ExpectedRawCOONNZ( const FESpace & fe_space )
{
   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex block_entries = local_dofs * local_dofs;
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   std::vector< bool > active_diagonal(
      static_cast< size_t >( num_elements ),
      false );
   GlobalIndex offdiag_blocks = 0;

   if constexpr ( IncludeCellTerms )
   {
      std::fill(
         active_diagonal.begin(),
         active_diagonal.end(),
         true );
   }

   if constexpr ( IncludeBoundaryFaceTerms )
   {
      for ( GlobalIndex element = 0; element < num_elements; ++element )
      {
         BoundaryFaceLoop(
            fe_space,
            element,
            [&] ( const auto & )
            {
               active_diagonal[static_cast< size_t >( element )] = true;
            });
      }
   }

   if constexpr ( IncludeInteriorFaceTerms )
   {
      for ( GlobalIndex element = 0; element < num_elements; ++element )
      {
         InteriorFaceLoop(
            fe_space,
            element,
            [&] ( const auto & )
            {
               active_diagonal[static_cast< size_t >( element )] = true;
               ++offdiag_blocks;
            });
      }
   }

   GlobalIndex block_count = offdiag_blocks;
   for ( const bool is_active : active_diagonal )
   {
      if ( is_active )
      {
         ++block_count;
      }
   }

   return block_count * block_entries;
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

void FillDeterministicInput( Vector & x )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] = 0.75 + 0.125 * static_cast< Real >( i );
   }
}

void ApplyTwoCellVectorH1P1MassReference(
   const Vector & x,
   Vector & y )
{
   // Two uniform 1D p1 elements with h = 0.5 give local scalar mass
   // (h / 6) * [[2, 1], [1, 2]]. Assembling the shared middle node produces
   // the component-major dense reference below for each independent component.
   const Real h = 0.5;
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

bool TestRawCOOBufferAllocation()
{
   auto buffer = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 4, 5, 6 );

   bool success = true;
   success = Check( buffer.num_rows == 4, "Raw COO row count is wrong." ) && success;
   success = Check( buffer.num_cols == 5, "Raw COO column count is wrong." ) && success;
   success = Check( buffer.nnz_raw == 6, "Raw COO triplet count is wrong." ) && success;
   success = Check( buffer.rows.data.host_pointer != nullptr, "Raw COO rows were not allocated on host." ) && success;
   success = Check( buffer.cols.data.host_pointer != nullptr, "Raw COO cols were not allocated on host." ) && success;
   success = Check( buffer.values.data.host_pointer != nullptr, "Raw COO values were not allocated on host." ) && success;

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      success = Check( ReadHost( buffer.rows )[i] == 0, "Raw COO rows were not initialized." ) && success;
      success = Check( ReadHost( buffer.cols )[i] == 0, "Raw COO cols were not initialized." ) && success;
      success = Check( Near( ReadHost( buffer.values )[i], 0.0 ), "Raw COO values were not initialized." ) && success;
   }

   return success;
}

bool TestRawCOOToCOOFinalization()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 2, 3, 6 );
   auto raw_data = GetHostReadWriteView( raw );

   raw_data.rows[0] = 1;
   raw_data.cols[0] = 2;
   raw_data.values[0] = 3.0;
   raw_data.rows[1] = 0;
   raw_data.cols[1] = 0;
   raw_data.values[1] = 1.0;
   raw_data.rows[2] = 1;
   raw_data.cols[2] = 2;
   raw_data.values[2] = 4.0;
   raw_data.rows[3] = 0;
   raw_data.cols[3] = 1;
   raw_data.values[3] = 5.0;
   raw_data.rows[4] = 1;
   raw_data.cols[4] = 1;
   raw_data.values[4] = 2.0;
   raw_data.rows[5] = 1;
   raw_data.cols[5] = 1;
   raw_data.values[5] = -2.0;

   auto coo =
      FinalizeRawCOOToCOOHost( raw );

   bool success = true;
   success = Check( coo.num_rows == 2, "Canonical COO row count is wrong." ) && success;
   success = Check( coo.num_cols == 3, "Canonical COO column count is wrong." ) && success;
   success = Check( coo.nnz == 4, "Canonical COO reduced nnz is wrong." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   success = Check( ReadHost( coo.rows )[0] == 0 && ReadHost( coo.cols )[0] == 0, "Canonical COO entry 0 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[0], 1.0 ), "Canonical COO entry 0 value is wrong." ) && success;
   success = Check( ReadHost( coo.rows )[1] == 0 && ReadHost( coo.cols )[1] == 1, "Canonical COO entry 1 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[1], 5.0 ), "Canonical COO entry 1 value is wrong." ) && success;
   success = Check( ReadHost( coo.rows )[2] == 1 && ReadHost( coo.cols )[2] == 1, "Canonical COO entry 2 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[2], 0.0 ), "Canonical COO should retain exact reduced zeros." ) && success;
   success = Check( ReadHost( coo.rows )[3] == 1 && ReadHost( coo.cols )[3] == 2, "Canonical COO entry 3 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[3], 7.0 ), "Canonical COO entry 3 value is wrong." ) && success;

   return success;
}

bool TestScalarL2CellMassRawCOOAgainstBSR()
{
   const Integer n = 4;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 2;
   FiniteElementOrders< order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto rho =
      MakeCoefficient< "density", PhysicalCoordinates >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + x_phys[0] * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      coo.nnz == raw_coo.nnz_raw,
      "Scalar L2 cell-only RawCOO should not create duplicate triplets." ) && success;
   success = Check(
      direct_coo.nnz == coo.nnz,
      "Direct scalar L2 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.25 + 0.5 * static_cast< Real >( i );
   }

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_bsr( GetAlgebraicDofExtent( fe_space ) );
   direct_coo( x, y_coo );
   y_bsr = 0.0;
   bsr( x, y_bsr );

   const Real * coo_data = y_coo.ReadHostData();
   const Real * bsr_data = y_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < y_coo.Size(); ++i )
   {
      success = Check(
         Near( coo_data[i], bsr_data[i] ),
         "Raw COO action disagrees with BSR action." ) && success;
   }

   return success;
}

bool TestVectorL2CellMassRawCOOAgainstSGBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   Cells< "mesh" > domain;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( domain, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Vector L2 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Vector L2 Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Vector L2 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 8 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 }
   };

   success = Check(
      raw_coo.nnz_raw == raw_coords.size(),
      "Vector p0 RawCOO test expected exactly two 2x2 cell blocks." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Vector p0 RawCOO coordinates are not component-major." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      coo.nnz == raw_coo.nnz_raw,
      "Vector L2 cell-only RawCOO should not create duplicate triplets." ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector L2 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_sgbsr( GetAlgebraicDofExtent( fe_space ) );
   direct_coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector L2 COO action disagrees with SGBSR action." ) && success;

   return success;
}

bool TestVectorBoundaryFaceMassCOOAgainstSGBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   BoundaryFacets< "mesh" > boundary_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( boundary_facets, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, true, false >( fe_space ),
      "Vector boundary face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector boundary COO assembly disagrees with explicit RawCOO finalization." ) && success;

   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      const bool cross_component =
         ( ReadHost( raw_coo.rows )[i] < num_elements ) !=
         ( ReadHost( raw_coo.cols )[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( ReadHost( raw_coo.values )[i], 0.0 ),
            "Vector boundary RawCOO should retain zero cross-component entries." ) && success;
      }
   }

   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      const bool cross_component =
         ( ReadHost( coo.rows )[i] < num_elements ) !=
         ( ReadHost( coo.cols )[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( ReadHost( coo.values )[i], 0.0 ),
            "Vector boundary canonical COO should retain zero cross-component entries." ) && success;
      }
   }

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_sgbsr( GetAlgebraicDofExtent( fe_space ) );
   direct_coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector boundary COO action disagrees with SGBSR action." ) && success;

   return success;
}

bool TestScalarH1CellMassRawCOOPreservesDuplicatesAgainstSGBSR()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto fe = MakeLobattoFiniteElement( orders );

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   IndirectH1RestrictionSpecification restriction{ restriction_indices, 3 };
   auto fe_space = MakeFiniteElementSpace( mesh, fe, restriction );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto rho =
      MakeCoefficient< "density", PhysicalCoordinates >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + 0.25 * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule,
         HostBSRBackend<>{} );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "H1 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "H1 Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "H1 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = Check(
      HasDuplicateCoordinate( raw_coo ),
      "Scalar H1 RawCOO should preserve duplicate raw triplet coordinates." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      raw_coo.nnz_raw > coo.nnz,
      "Scalar H1 RawCOO should have fewer canonical entries after reduction." ) && success;
   success = Check(
      direct_coo.nnz == coo.nnz,
      "Direct scalar H1 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   for ( Integer vector_case = 0; vector_case < 2; ++vector_case )
   {
      Vector x( GetAlgebraicDofExtent( fe_space ) );
      Real * x_data = x.WriteHostData();
      for ( GlobalIndex i = 0; i < x.Size(); ++i )
      {
         x_data[i] =
            vector_case == 0
               ? 0.5 + 0.25 * static_cast< Real >( i )
               : 1.0 - 0.2 * static_cast< Real >( i );
      }

      Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
      Vector y_sgbsr( GetAlgebraicDofExtent( fe_space ) );
      direct_coo( x, y_coo );
      y_sgbsr = 0.0;
      sgbsr( x, y_sgbsr );

      const Real * coo_data = y_coo.ReadHostData();
      const Real * sgbsr_data = y_sgbsr.ReadHostData();
      for ( GlobalIndex i = 0; i < y_coo.Size(); ++i )
      {
         success = Check(
            Near( coo_data[i], sgbsr_data[i] ),
            "H1 Raw COO action disagrees with SGBSR action." ) && success;
      }
   }

   return success;
}

bool TestVectorH1CellMassRawCOOAgainstDenseReference()
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

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   IndirectH1RestrictionSpecification scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorIndirectH1RestrictionSpecification< 2 >( scalar_restriction );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > domain;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( domain, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Vector H1 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( GetAlgebraicDofExtent( fe_space ) ),
      "Vector H1 Raw COO column dimension is wrong." ) && success;
   success = Check(
      GetAlgebraicDofExtent( fe_space ) == 6,
      "Vector H1 test expected two component-major copies of three scalar true DoFs." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Vector H1 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = Check(
      HasDuplicateCoordinate( raw_coo ),
      "Vector H1 RawCOO should preserve duplicate coordinates from shared true DoFs." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      raw_coo.nnz_raw > coo.nnz,
      "Vector H1 RawCOO should have fewer canonical entries after reduction." ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector H1 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_expected( GetAlgebraicDofExtent( fe_space ) );
   direct_coo( x, y_coo );
   ApplyTwoCellVectorH1P1MassReference( x, y_expected );

   success = CheckVectorNear(
      y_coo,
      y_expected,
      "Vector H1 COO action disagrees with the dense p1 mass reference." ) && success;

   return success;
}

bool TestScalarP0InteriorJumpAnalyticRawCOO()
{
   // Analytic p0 assumptions for this mesh/form:
   // - each element basis value is 1 everywhere on the element;
   // - the single interior vertical face has physical measure 1.
   // Therefore integrate(interior_facets, jump(u) * jump(v)) on two cells is
   // [[1, -1], [-1, 1]].
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw == 4,
      "Analytic p0 interior jump RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 }
   };
   const std::array< Real, 4 > raw_values{ 1.0, -1.0, 1.0, -1.0 };

   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Analytic p0 RawCOO did not preserve the directed interior face traversal." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
         "Analytic p0 RawCOO value is wrong." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check( coo.nnz == 4, "Analytic p0 canonical COO nnz is wrong." ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 }
   };
   const std::array< Real, 4 > coo_values{ 1.0, -1.0, -1.0, 1.0 };

   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
         "Analytic p0 canonical COO value is wrong." ) && success;
   }

   Vector x( 2 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;

   Vector y( 2 );
   coo( x, y );
   const Real * y_data = y.ReadHostData();
   success = Check( Near( y_data[0], -3.0 ), "Analytic p0 COO action row 0 is wrong." ) && success;
   success = Check( Near( y_data[1], 3.0 ), "Analytic p0 COO action row 1 is wrong." ) && success;

   return success;
}

bool TestVectorP0InteriorJumpAnalyticRawCOO()
{
   // Analytic p0 assumptions for this mesh/form:
   // - each component basis value is 1 on its owning element;
   // - the single interior vertical face has physical measure 1;
   // - directed traversal emits current-row self and neighbor vector blocks.
   // Therefore dot(jump(u), jump(v)) gives the scalar jump matrix independently
   // for each component, with exact zero cross-component entries retained.
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   InteriorFacets< "mesh" > interior_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, dot( jump( u ), jump( v ) ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, false, true >( fe_space ),
      "Vector analytic p0 interior jump RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 16 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 2 }
   };
   const std::array< Real, 16 > raw_values{
      1.0, 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0, -1.0,
      1.0, 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0, -1.0
   };

   success = Check(
      raw_coo.nnz_raw == raw_values.size(),
      "Vector analytic p0 RawCOO should have four compact 2x2 blocks." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Vector analytic p0 RawCOO compact directed coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
         "Vector analytic p0 RawCOO value is wrong." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector analytic p0 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 16 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 }
   };
   const std::array< Real, 16 > coo_values{
      1.0, -1.0, 0.0, 0.0,
      -1.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, -1.0,
      0.0, 0.0, -1.0, 1.0
   };

   success = Check(
      coo.nnz == coo_values.size(),
      "Vector analytic p0 canonical COO nnz is wrong." ) && success;
   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Vector analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
         "Vector analytic p0 canonical COO value is wrong." ) && success;
   }

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;
   x_data[2] = 7.0;
   x_data[3] = 11.0;

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_sgbsr( GetAlgebraicDofExtent( fe_space ) );
   coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   Vector y_expected( GetAlgebraicDofExtent( fe_space ) );
   Real * expected_data = y_expected.WriteHostData();
   expected_data[0] = -3.0;
   expected_data[1] = 3.0;
   expected_data[2] = -4.0;
   expected_data[3] = 4.0;

   success = CheckVectorNear(
      y_coo,
      y_expected,
      "Vector analytic p0 COO action is wrong." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector analytic p0 COO action disagrees with SGBSR action." ) && success;

   return success;
}

bool TestScalarBoundaryFaceMassCOOAgainstGenericAndBSR()
{
   Cartesian2DMesh mesh( 0.5, 2, 2 );

   constexpr Integer order = 1;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   BoundaryFacets< "mesh" > boundary_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( boundary_facets, u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< order + 2, order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, true, false >( fe_space ),
      "Boundary face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_bsr( GetAlgebraicDofExtent( fe_space ) );
   Vector y_generic( GetAlgebraicDofExtent( fe_space ) );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Boundary face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Boundary face COO action disagrees with matrix-free action." ) && success;

   return success;
}

bool TestScalarInteriorJumpCOOAgainstGenericAndBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 1;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< order + 2, order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, false, true >( fe_space ),
      "Interior face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_bsr( GetAlgebraicDofExtent( fe_space ) );
   Vector y_generic( GetAlgebraicDofExtent( fe_space ) );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Interior face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Interior face COO action disagrees with matrix-free action." ) && success;

   return success;
}

bool TestScalarCombinedFaceCOOOffsetsAndAccumulation()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > cells;
   BoundaryFacets< "mesh" > boundary_facets;
   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;

   auto weak_form =
      integrate( cells, u * v ) +
      integrate( boundary_facets, u * v ) +
      integrate( boundary_facets, 2.0 * u * v ) +
      integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< true, true, true >( fe_space ),
      "Combined face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 }
   };
   const std::array< Real, 4 > raw_values{ 11.0, -1.0, 11.0, -1.0 };

   success = Check(
      raw_coo.nnz_raw == raw_values.size(),
      "Combined p0 RawCOO should have compact diagonal/offdiag blocks only." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Combined p0 RawCOO compact block coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
         "Combined p0 RawCOO did not accumulate self contributions into diagonal slots." ) && success;
   }

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 }
   };
   const std::array< Real, 4 > coo_values{ 11.0, -1.0, -1.0, 11.0 };

   success = Check(
      coo.nnz == coo_values.size(),
      "Combined p0 canonical COO nnz is wrong." ) && success;
   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Combined p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
         "Combined p0 canonical COO value is wrong." ) && success;
   }

   Vector x( GetAlgebraicDofExtent( fe_space ) );
   FillDeterministicInput( x );

   Vector y_coo( GetAlgebraicDofExtent( fe_space ) );
   Vector y_bsr( GetAlgebraicDofExtent( fe_space ) );
   Vector y_generic( GetAlgebraicDofExtent( fe_space ) );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Combined face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Combined face COO action disagrees with matrix-free action." ) && success;

   return success;
}

template < typename FESpace >
bool CheckTensorProductFacetAssembly(
   const FESpace & fe_space,
   const bool expect_duplicate_coordinates )
{
   using KernelPolicy = SerialKernelConfiguration;
   using Space = std::remove_cvref_t< FESpace >;
   const auto & mesh =
      static_cast< const typename Space::mesh_type & >( fe_space );

   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   const auto weak_form =
      integrate(
         InteriorFacets< "mesh" >{},
         jump( u ) * jump( v ) ) +
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v );
   const auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         context,
         integration_rule,
         HostBSRBackend<>{} );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         context,
         integration_rule );

   const GlobalIndex num_dofs =
      static_cast< GlobalIndex >(
         GetAlgebraicDofExtent( fe_space ) );
   bool success = true;
   success = Check(
      raw.num_rows == num_dofs && raw.num_cols == num_dofs,
      "Tensor-product facet RawCOO has the wrong matrix dimensions." ) && success;
   success = Check(
      csr.num_rows == num_dofs && csr.num_cols == num_dofs,
      "Tensor-product facet CSR has the wrong matrix dimensions." ) && success;
   success = Check(
      raw.nnz_raw ==
         ExpectedRawCOONNZ< false, true, true >( fe_space ),
      "Tensor-product facet RawCOO compact block count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = Check(
      HasDuplicateCoordinate( raw ) == expect_duplicate_coordinates,
      "Tensor-product facet RawCOO duplicate-coordinate behavior is wrong." ) && success;
   if ( expect_duplicate_coordinates )
   {
      success = Check(
         raw.nnz_raw > finalized.nnz,
         "Tensor-product H1-factor RawCOO coordinates were not reduced." ) && success;
   }

   Vector input( num_dofs );
   FillDeterministicInput( input );
   Vector expected( num_dofs );
   Vector raw_result( num_dofs );
   Vector csr_result( num_dofs );
   Vector sgbsr_result( num_dofs );
   expected = 0.0;
   raw_result = 0.0;
   csr_result = 0.0;
   sgbsr_result = 0.0;
   generic( input, expected );
   finalized( input, raw_result );
   csr( input, csr_result );
   sgbsr( input, sgbsr_result );

   success = CheckVectorNear(
      raw_result,
      expected,
      "Finalized tensor-product facet RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckVectorNear(
      csr_result,
      expected,
      "Tensor-product facet CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckVectorNear(
      sgbsr_result,
      expected,
      "Tensor-product facet SGBSR action disagrees with MakeGenericOperator." ) && success;

   return success;
}

template < typename FESpace >
bool CheckTensorProductCellSGBSR( const FESpace & fe_space )
{
   using KernelPolicy = SerialKernelConfiguration;
   using Space = std::remove_cvref_t< FESpace >;
   static_assert( DefaultBsrMappingSpace< Space > );
   const auto & mesh =
      static_cast< const typename Space::mesh_type & >( fe_space );

   TrialSpace< "tensor_product_cell" > u;
   TestSpace< "tensor_product_cell" > v;
   const auto form =
      integrate( Cells< "mesh" >{}, u * v );
   const auto context =
      MakeWeakFormContext(
         MakeTrialField< "tensor_product_cell" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );
   const auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule,
         HostBSRBackend<>{} );

   const GlobalIndex num_dofs =
      GetAlgebraicDofExtent( fe_space );
   Vector input( num_dofs );
   Vector expected( num_dofs );
   Vector actual( num_dofs );
   FillDeterministicInput( input );
   expected = 0.0;
   actual = 0.0;
   generic( input, expected );
   sgbsr( input, actual );
   return CheckVectorNear(
      actual,
      expected,
      "Tensor-product cell SGBSR action disagrees with MakeGenericOperator." );
}

bool TestTensorProductFacetRawCOOAndCSRAgainstGeneric()
{
   Cartesian1DMesh spatial_mesh( 0.5, 2 );
   Cartesian1DMesh velocity_mesh( 0.5, 2 );
   const auto product_mesh =
      MakeCartesianProductMesh( spatial_mesh, velocity_mesh );

   const auto spatial_l2_space =
      MakeFiniteElementSpace(
         spatial_mesh,
         GLFiniteElement< 1 >{},
         ContiguousL2RestrictionSpecification{} );
   const auto velocity_dg_space =
      MakeFiniteElementSpace(
         velocity_mesh,
         GLFiniteElement< 1 >{},
         ContiguousL2RestrictionSpecification{} );
   const auto l2_dg_restriction =
      MakeTensorProductRestriction(
         spatial_l2_space,
         velocity_dg_space );
   const auto l2_dg_space =
      MakeFiniteElementSpace(
         product_mesh,
         GLFiniteElement< 1, 1 >{},
         l2_dg_restriction );

   std::array< int, 4 > spatial_h1_indices{ 0, 1, 1, 2 };
   HostDevicePointer< const int > spatial_h1_pointer{};
   spatial_h1_pointer.host_pointer = spatial_h1_indices.data();
   const auto spatial_h1_space =
      MakeFiniteElementSpace(
         spatial_mesh,
         GLLFiniteElement< 1 >{},
         IndirectH1RestrictionSpecification{ spatial_h1_pointer, 3 } );
   const auto h1_dg_restriction =
      MakeTensorProductRestriction(
         spatial_h1_space,
         velocity_dg_space );
   using H1DGFiniteElement = FiniteElement<
      HyperCube< 2 >,
      TensorShapeFunctions<
         GaussLobattoLegendreShapeFunctions< 1 >,
         GaussLegendreShapeFunctions< 1 > > >;
   const auto h1_dg_space =
      MakeFiniteElementSpace(
         product_mesh,
         H1DGFiniteElement{},
         h1_dg_restriction );

   bool success = true;
   success =
      CheckTensorProductCellSGBSR( l2_dg_space ) && success;
   success =
      CheckTensorProductFacetAssembly(
         l2_dg_space,
         false ) && success;
   success =
      CheckTensorProductFacetAssembly(
         h1_dg_space,
         true ) && success;
   return success;
}

template < typename Operator >
Vector ApplyRectangularOperator(
   const Operator & op,
   const Vector & x,
   const GlobalIndex output_size )
{
   Vector y( output_size );
   y = 0.0;
   op( x, y );
   return y;
}

template < typename Operator >
bool CheckRectangularAction(
   const Operator & op,
   const Vector & x,
   const Vector & expected,
   const char * message )
{
   const auto actual =
      ApplyRectangularOperator( op, x, expected.Size() );
   return CheckVectorNear( actual, expected, message );
}

template <
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule >
bool CheckFacetAssemblyCase(
   const WeakForm & weak_form,
   const WeakFormContext & context,
   const IntegrationRule & integration_rule,
   const GlobalIndex num_rows,
   const GlobalIndex num_cols,
   const GlobalIndex expected_raw_nnz,
   const bool require_duplicate_reduction,
   const bool expect_zero_action,
   const char * case_name )
{
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         context,
         integration_rule,
         HostBSRBackend<>{} );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      raw.num_rows == num_rows && raw.num_cols == num_cols,
      "Facet RawCOO has the wrong matrix dimensions." ) && success;
   success = Check(
      csr.num_rows == num_rows && csr.num_cols == num_cols,
      "Facet CSR has the wrong matrix dimensions." ) && success;
   success = Check(
      raw.nnz_raw == expected_raw_nnz,
      "Facet RawCOO compact block count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;

   if ( require_duplicate_reduction )
   {
      success = Check(
         HasDuplicateCoordinate( raw ),
         "Facet RawCOO should contain duplicate shared-DoF coordinates." ) && success;
      success = Check(
         raw.nnz_raw > finalized.nnz,
         "Facet RawCOO coordinates were not reduced." ) && success;
   }

   Vector input( num_cols );
   FillDeterministicInput( input );
   const auto expected =
      ApplyRectangularOperator( generic, input, num_rows );
   const auto raw_result =
      ApplyRectangularOperator( finalized, input, num_rows );
   const auto csr_result =
      ApplyRectangularOperator( csr, input, num_rows );
   const auto sgbsr_result =
      ApplyRectangularOperator( sgbsr, input, num_rows );

   success = CheckVectorNear(
      raw_result,
      expected,
      "Finalized facet RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckVectorNear(
      csr_result,
      expected,
      "Facet CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckVectorNear(
      sgbsr_result,
      expected,
      "Facet SGBSR action disagrees with MakeGenericOperator." ) && success;

   const Real * expected_data = expected.ReadHostData();
   bool has_nonzero = false;
   for ( GlobalIndex row = 0; row < num_rows; ++row )
   {
      has_nonzero =
         has_nonzero || std::abs( expected_data[row] ) >= tolerance;
      if ( expect_zero_action )
      {
         success = Check(
            Near( expected_data[row], 0.0 ),
            "Continuous H1 jump action should be zero." ) && success;
      }
   }

   if ( expect_zero_action )
   {
      for ( GlobalIndex entry = 0; entry < finalized.nnz; ++entry )
      {
         success = Check(
            Near( ReadHost( finalized.values )[entry], 0.0 ),
            "Canonical H1 jump matrix should contain only zero values." ) && success;
      }
   }
   else
   {
      if ( !has_nonzero )
      {
         std::cout << "Facet case was unexpectedly zero: "
                   << case_name << '\n';
      }
      success = Check(
         has_nonzero,
         "Facet validation expected a nonzero operator action." ) && success;
   }

   return success;
}

bool TestScalarH1FacetRawCOOAndCSRAgainstGeneric()
{
   Cartesian2DMesh mesh( 0.5, 2, 2 );
   const auto h1_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   const auto l2_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );

   const std::array< int, 16 > h1_map{
      0, 1, 3, 4,
      1, 2, 4, 5,
      3, 4, 6, 7,
      4, 5, 7, 8 };
   HostDevicePointer< const int > h1_indices{};
   h1_indices.host_pointer = h1_map.data();
   const auto h1_space =
      MakeFiniteElementSpace(
         mesh,
         h1_fe,
         IndirectH1RestrictionSpecification{ h1_indices, 9 } );
   const auto l2_space = MakeFiniteElementSpace( mesh, l2_fe );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );

   const GlobalIndex h1_dofs =
      static_cast< GlobalIndex >(
         GetAlgebraicDofExtent( h1_space ) );
   const GlobalIndex l2_dofs =
      static_cast< GlobalIndex >(
         GetAlgebraicDofExtent( l2_space ) );
   bool success = true;

   {
      TrialSpace< "h1" > u;
      TestSpace< "h1" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            average( u ) * average( v ) ) +
         integrate(
            BoundaryFacets< "mesh" >{},
            u * v );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "h1" >( h1_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            h1_dofs,
            h1_dofs,
            ExpectedRawCOONNZ< false, true, true >( h1_space ),
            true,
            false,
            "H1/H1 skeleton and boundary mass" ) && success;
   }

   {
      TrialSpace< "h1_jump" > u;
      TestSpace< "h1_jump" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            jump( u ) * jump( v ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "h1_jump" >( h1_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            h1_dofs,
            h1_dofs,
            ExpectedRawCOONNZ< false, false, true >( h1_space ),
            true,
            true,
            "H1/H1 jump cancellation" ) && success;
   }

   {
      TrialSpace< "h1_trial" > u;
      TestSpace< "l2_test" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            average( u ) * jump( v ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "h1_trial" >( h1_space ),
            MakeTestField< "l2_test" >( l2_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            l2_dofs,
            h1_dofs,
            ExpectedRawCOONNZ< false, false, true >( l2_space ),
            false,
            false,
            "H1 trial to L2 test" ) && success;
   }

   {
      TrialSpace< "l2_trial" > u;
      TestSpace< "h1_test" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            jump( u ) * average( v ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "l2_trial" >( l2_space ),
            MakeTestField< "h1_test" >( h1_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            h1_dofs,
            l2_dofs,
            ExpectedRawCOONNZ< false, false, true >( l2_space ),
            false,
            true,
            "L2 trial to H1 test" ) && success;
   }

   {
      TrialSpace< "l2_average_trial" > u;
      TestSpace< "h1_average_test" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            average( u ) * average( v ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "l2_average_trial" >( l2_space ),
            MakeTestField< "h1_average_test" >( h1_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            h1_dofs,
            l2_dofs,
            ExpectedRawCOONNZ< false, false, true >( l2_space ),
            false,
            false,
            "L2 trial to H1 test side-symmetric mass" ) && success;
   }

   return success;
}

bool TestVectorH1FacetRawCOOAndCSRAgainstGeneric()
{
   Cartesian2DMesh mesh( 0.5, 2, 2 );
   const auto scalar_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   const auto vector_fe =
      MakeVectorFiniteElement( scalar_fe, scalar_fe );

   const std::array< int, 16 > h1_map{
      0, 1, 3, 4,
      1, 2, 4, 5,
      3, 4, 6, 7,
      4, 5, 7, 8 };
   HostDevicePointer< const int > h1_indices{};
   h1_indices.host_pointer = h1_map.data();
   const auto scalar_specification =
      IndirectH1RestrictionSpecification{ h1_indices, 9 };
   const auto vector_space =
      MakeFiniteElementSpace(
         mesh,
         vector_fe,
         MakeVectorIndirectH1RestrictionSpecification< 2 >(
            scalar_specification ) );
   const auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} );
   const GlobalIndex vector_dofs =
      GetAlgebraicDofExtent( vector_space );
   bool success = true;

   {
      VectorTrialSpace< "vector_h1_boundary" > u;
      VectorTestSpace< "vector_h1_boundary" > v;
      const auto form =
         integrate(
            BoundaryFacets< "mesh" >{},
            dot( u, v ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "vector_h1_boundary" >( vector_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            vector_dofs,
            vector_dofs,
            ExpectedRawCOONNZ< false, true, false >( vector_space ),
            true,
            false,
            "conforming vector H1 boundary mass" ) && success;
   }

   {
      VectorTrialSpace< "vector_h1_interior" > u;
      VectorTestSpace< "vector_h1_interior" > v;
      const auto form =
         integrate(
            InteriorFacets< "mesh" >{},
            dot( average( u ), average( v ) ) );
      const auto context =
         MakeWeakFormContext(
            MakeTrialField< "vector_h1_interior" >( vector_space ),
            MakeIntegrationDomain< "mesh" >( mesh ) );
      success =
         CheckFacetAssemblyCase(
            form,
            context,
            integration_rule,
            vector_dofs,
            vector_dofs,
            ExpectedRawCOONNZ< false, false, true >( vector_space ),
            true,
            false,
            "conforming vector H1 interior average mass" ) && success;
   }

   return success;
}

template < size_t NumNodes >
Real IndependentLagrangeValue(
   const std::array< Real, NumNodes > & nodes,
   const size_t basis,
   const Real point )
{
   Real value = 1.0;
   for ( size_t other = 0; other < NumNodes; ++other )
   {
      if ( other != basis )
      {
         value *=
            ( point - nodes[other] ) /
            ( nodes[basis] - nodes[other] );
      }
   }
   return value;
}

std::array< Real, 36 > IndependentP2TestP1TrialCellMassBlock()
{
   // Test-owned values on [0, 1]. These deliberately do not use GenDiL's
   // point, weight, interpolation, qdata, or DofToQuad implementations.
   constexpr std::array< Real, 2 > trial_nodes{
      0.21132486540518711775,
      0.78867513459481288225 };
   constexpr std::array< Real, 3 > test_nodes{
      0.11270166537925831148,
      0.5,
      0.88729833462074168852 };
   constexpr std::array< Real, 5 > points{
      0.0469100770306680,
      0.230765344947158,
      0.500000000000000,
      0.769234655052842,
      0.953089922969332 };
   constexpr std::array< Real, 5 > weights{
      0.118463442528095,
      0.239314335249683,
      0.284444444444444,
      0.239314335249683,
      0.118463442528095 };

   std::array< Real, 36 > block{};
   for ( size_t trial_y = 0; trial_y < trial_nodes.size(); ++trial_y )
   {
      for ( size_t trial_x = 0; trial_x < trial_nodes.size(); ++trial_x )
      {
         const size_t trial = trial_x + 2 * trial_y;
         for ( size_t test_y = 0; test_y < test_nodes.size(); ++test_y )
         {
            for ( size_t test_x = 0; test_x < test_nodes.size(); ++test_x )
            {
               const size_t test = test_x + 3 * test_y;
               Real value = 0.0;
               for ( size_t qy = 0; qy < points.size(); ++qy )
               {
                  for ( size_t qx = 0; qx < points.size(); ++qx )
                  {
                     value +=
                        weights[qx] * weights[qy] *
                        IndependentLagrangeValue(
                           test_nodes,
                           test_x,
                           points[qx] ) *
                        IndependentLagrangeValue(
                           test_nodes,
                           test_y,
                           points[qy] ) *
                        IndependentLagrangeValue(
                           trial_nodes,
                           trial_x,
                           points[qx] ) *
                        IndependentLagrangeValue(
                           trial_nodes,
                           trial_y,
                           points[qy] );
                  }
               }
               // Cartesian2DMesh(1.0, ...) has unit affine-cell Jacobian.
               block[trial * 9 + test] = value;
            }
         }
      }
   }
   return block;
}

bool TestRectangularP1TrialP2TestCellAssembly()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form = integrate( Cells< "mesh" >{}, u * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      raw.num_rows == 18 && raw.num_cols == 8,
      "Rectangular cell RawCOO dimensions are not 18 x 8." ) && success;
   success = Check(
      raw.nnz_raw == 72,
      "Rectangular cell RawCOO must contain two 9 x 4 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct rectangular COO disagrees with finalized RawCOO." ) && success;
   success = Check(
      coo.num_rows == 18 && coo.num_cols == 8 &&
      csr.num_rows == 18 && csr.num_cols == 8 &&
      csc.num_rows == 18 && csc.num_cols == 8,
      "A derived cell format lost the 18 x 8 dimensions." ) && success;
   success = Check(
      bsr.block_rows == 9 && bsr.block_cols == 4 &&
      bsr.num_row_blocks == 2 && bsr.num_col_blocks == 2,
      "Rectangular cell BSR has the wrong block dimensions." ) && success;

   const auto independent_block =
      IndependentP2TestP1TrialCellMassBlock();
   const auto raw_data = GetHostReadView( raw );
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex trial = 0; trial < 4; ++trial )
      {
         const std::array< GlobalIndex, 2 > trial_indices{
            trial % 2,
            trial / 2 };
         const GlobalIndex expected_col =
            GetGlobalDofIndex( trial_space, element, trial_indices );
         for ( GlobalIndex test = 0; test < 9; ++test )
         {
            const std::array< GlobalIndex, 2 > test_indices{
               test % 3,
               test / 3 };
            const GlobalIndex expected_row =
               GetGlobalDofIndex( test_space, element, test_indices );
            const GlobalIndex entry =
               element * 36 + trial * 9 + test;
            success = Check(
               raw_data.rows[entry] == expected_row,
               "Rectangular cell RawCOO row is not derived from the p2 test space." ) && success;
            success = Check(
               raw_data.cols[entry] == expected_col,
               "Rectangular cell RawCOO column is not derived from the p1 trial space." ) && success;
         }
      }
   }
   for ( GlobalIndex entry = 0; entry < 36; ++entry )
   {
      success = Check(
         Near( raw_data.values[entry], independent_block[entry] ),
         "Rectangular 9 x 4 cell block disagrees with independent nodal integration." ) && success;
   }

   Vector x( 8 );
   FillDeterministicInput( x );
   const auto expected = ApplyRectangularOperator( generic, x, 18 );
   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Finalized rectangular RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Rectangular COO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Rectangular CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Rectangular CSC action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      bsr,
      x,
      expected,
      "Rectangular cell BSR action disagrees with MakeGenericOperator." ) && success;

   return success;
}

bool TestRectangularP1TrialP2TestAffineBoundaryAssembly()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto source = MakeCoefficient< "source" >(
      [] GENDIL_HOST_DEVICE () -> Real { return 1.375; } );
   auto form =
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v + source * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      raw.num_rows == 18 && raw.num_cols == 8 && raw.nnz_raw == 72,
      "Rectangular affine-boundary RawCOO does not contain two 9 x 4 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct affine-boundary COO disagrees with finalized RawCOO." ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check(
      coo.num_rows == 18 && coo.num_cols == 8 &&
      csr.num_rows == 18 && csr.num_cols == 8 &&
      csc.num_rows == 18 && csc.num_cols == 8,
      "A derived affine-boundary format lost the 18 x 8 dimensions." ) && success;
   success = Check(
      bsr.block_rows == 9 && bsr.block_cols == 4 &&
      bsr.num_row_blocks == 2 && bsr.num_col_blocks == 2,
      "Rectangular affine-boundary BSR has the wrong block dimensions." ) && success;

   Vector x( 8 );
   FillDeterministicInput( x );
   Vector zero( 8 );
   zero = 0.0;
   const auto fx = ApplyRectangularOperator( generic, x, 18 );
   const auto fzero = ApplyRectangularOperator( generic, zero, 18 );
   Vector expected( 18 );
   Real * expected_data = expected.WriteHostData();
   const Real * fx_data = fx.ReadHostData();
   const Real * fzero_data = fzero.ReadHostData();
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      expected_data[i] = fx_data[i] - fzero_data[i];
   }

   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Affine-boundary RawCOO action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Affine-boundary COO action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Affine-boundary CSR action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Affine-boundary CSC action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      bsr,
      x,
      expected,
      "Affine-boundary BSR action disagrees with F(x) - F(0)." ) && success;

   return success;
}

struct OrientedTwoQuadFixture
{
   // Cell 0 is axis-aligned. Cell 1 is rotated by 180 degrees, so both cells
   // are positively oriented but their shared-face tangential coordinates run
   // in opposite directions.
   std::array< Real, 12 > node_data{
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 0.0,
      2.0, 1.0 };
   std::array< int, 8 > restriction_data{
      0, 1, 2, 3,
      5, 3, 4, 1 };
   HostDevicePointer< const int > restriction_pointer{};
   UnstructuredConformingConnectivity< HyperCube< 2 > > connectivity{ 2 };

   OrientedTwoQuadFixture()
   {
      restriction_pointer.host_pointer = restriction_data.data();
      for ( GlobalIndex element = 0; element < 2; ++element )
      {
         for ( Integer face = 0; face < HyperCube< 2 >::num_faces; ++face )
         {
            connectivity[element].faces[face] =
               { 0,
                 {},
                 MakeReferencePermutation< 2 >(),
                 {},
                 {},
                 true };
         }
      }

      const Permutation< 2 > rotated_neighbor{ { -1, -2 } };
      connectivity[0].faces[2] =
         { 1, {}, rotated_neighbor, {}, {}, false };
      connectivity[1].faces[2] =
         { 0, {}, rotated_neighbor, {}, {}, false };
   }

   auto MakeMesh() const
   {
      return QuadMesh< 1 >{
         MakeFIFOView( node_data.data(), GlobalIndex( 2 ), GlobalIndex( 6 ) ),
         MakeFIFOView(
            restriction_pointer,
            GlobalIndex( 2 ),
            GlobalIndex( 2 ),
            GlobalIndex( 2 ) ),
         connectivity,
         2 };
   }
};

Real IndependentP2P1TangentialMass(
   const size_t trial,
   const size_t test )
{
   constexpr std::array< Real, 3 > trial_nodes{
      0.11270166537925831148,
      0.5,
      0.88729833462074168852 };
   constexpr std::array< Real, 2 > test_nodes{
      0.21132486540518711775,
      0.78867513459481288225 };
   constexpr std::array< Real, 5 > points{
      0.0469100770306680,
      0.230765344947158,
      0.500000000000000,
      0.769234655052842,
      0.953089922969332 };
   constexpr std::array< Real, 5 > weights{
      0.118463442528095,
      0.239314335249683,
      0.284444444444444,
      0.239314335249683,
      0.118463442528095 };

   Real value = 0.0;
   for ( size_t q = 0; q < points.size(); ++q )
   {
      value +=
         weights[q] *
         IndependentLagrangeValue( trial_nodes, trial, points[q] ) *
         IndependentLagrangeValue( test_nodes, test, points[q] );
   }
   return value;
}

bool TestRectangularP2TrialP1TestOrientedInteriorAssembly()
{
   OrientedTwoQuadFixture fixture;
   auto mesh = fixture.MakeMesh();
   const auto fixture_face =
      mesh.GetLocalFaceInfo(
         0,
         std::integral_constant< Integer, 2 >{} );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form =
      integrate(
         InteriorFacets< "mesh" >{},
         jump( u ) * jump( v ) );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      fixture_face.PlusSide().orientation ==
         Permutation< 2 >{ { -1, -2 } },
      "Two-quad fixture did not retain its nonidentity face orientation." ) && success;
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      const int node00 = fixture.restriction_data[4 * element];
      const int node10 = fixture.restriction_data[4 * element + 1];
      const int node01 = fixture.restriction_data[4 * element + 2];
      const Real dx_x =
         fixture.node_data[2 * node10] - fixture.node_data[2 * node00];
      const Real dx_y =
         fixture.node_data[2 * node10 + 1] -
         fixture.node_data[2 * node00 + 1];
      const Real dy_x =
         fixture.node_data[2 * node01] - fixture.node_data[2 * node00];
      const Real dy_y =
         fixture.node_data[2 * node01 + 1] -
         fixture.node_data[2 * node00 + 1];
      const Real determinant =
         dx_x * dy_y - dx_y * dy_x;
      success = Check(
         determinant > 0.0,
         "Two-quad orientation fixture contains a non-positive cell." ) && success;
   }
   success = Check(
      raw.num_rows == 8 && raw.num_cols == 18,
      "Reverse rectangular interior RawCOO dimensions are not 8 x 18." ) && success;
   success = Check(
      raw.nnz_raw == 144,
      "Reverse rectangular interior RawCOO must contain four 4 x 9 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct reverse-interior COO disagrees with finalized RawCOO." ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check(
      coo.num_rows == 8 && coo.num_cols == 18 &&
      csr.num_rows == 8 && csr.num_cols == 18 &&
      csc.num_rows == 8 && csc.num_cols == 18,
      "A derived interior format lost the 8 x 18 dimensions." ) && success;
   success = Check(
      bsr.block_rows == 4 && bsr.block_cols == 9 &&
      bsr.num_row_blocks == 2 && bsr.num_col_blocks == 2 &&
      bsr.num_blocks == 4,
      "Reverse-interior BSR has the wrong domain-derived block pattern." ) && success;
   const auto bsr_data = GetHostReadView( bsr );
   success = Check(
      bsr_data.row_offsets[0] == 0 &&
      bsr_data.row_offsets[1] == 2 &&
      bsr_data.row_offsets[2] == 4 &&
      bsr_data.col_indices[0] == 0 &&
      bsr_data.col_indices[1] == 1 &&
      bsr_data.col_indices[2] == 0 &&
      bsr_data.col_indices[3] == 1,
      "Reverse-interior BSR adjacency is not derived from the domain mesh." ) && success;

   const auto raw_data = GetHostReadView( raw );
   constexpr GlobalIndex first_neighbor_offset = 36;
   constexpr std::array< GlobalIndex, 4 > selected_trial_dofs{
      0, 1, 5, 8 };
   for ( const GlobalIndex trial : selected_trial_dofs )
   {
      const GlobalIndex trial_x = trial % 3;
      const GlobalIndex trial_y = trial / 3;
      const std::array< GlobalIndex, 2 > native_trial_indices{
         trial_x,
         trial_y };
      const GlobalIndex expected_col =
         GetGlobalDofIndex( trial_space, 1, native_trial_indices );

      for ( GlobalIndex test = 0; test < 4; ++test )
      {
         const GlobalIndex test_x = test % 2;
         const GlobalIndex test_y = test / 2;
         const std::array< GlobalIndex, 2 > test_indices{
            test_x,
            test_y };
         const GlobalIndex expected_row =
            GetGlobalDofIndex( test_space, 0, test_indices );
         const GlobalIndex entry =
            first_neighbor_offset + trial * 4 + test;
         success = Check(
            raw_data.rows[entry] == expected_row,
            "Oriented neighbor RawCOO row is not from the p1 minus-side test space." ) && success;
         success = Check(
            raw_data.cols[entry] == expected_col,
            "Oriented neighbor RawCOO column does not apply the p2 plus-side reversal." ) && success;

         constexpr std::array< Real, 2 > p1_nodes{
            0.21132486540518711775,
            0.78867513459481288225 };
         constexpr std::array< Real, 3 > p2_nodes{
            0.11270166537925831148,
            0.5,
            0.88729833462074168852 };
         // RawCOO slots are indexed by native neighbor columns. The {-1,-2}
         // plus orientation maps the reversed reference basis into this slot.
         const GlobalIndex reference_trial_x = 2 - trial_x;
         const GlobalIndex reference_trial_y = 2 - trial_y;
         const Real expected_value =
            -IndependentLagrangeValue(
               p1_nodes,
               static_cast<size_t>( test_x ),
               1.0 ) *
            IndependentLagrangeValue(
               p2_nodes,
               static_cast<size_t>( reference_trial_x ),
               0.0 ) *
            IndependentP2P1TangentialMass(
               static_cast<size_t>( reference_trial_y ),
               static_cast<size_t>( test_y ) );
         success = Check(
            Near( raw_data.values[entry], expected_value ),
            "Oriented neighbor RawCOO value disagrees with the independent face block." ) && success;
      }
   }

   Vector x( 18 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] =
         0.35 + 0.11 * static_cast< Real >( i ) +
         0.017 * static_cast< Real >( i * i );
   }
   const auto expected = ApplyRectangularOperator( generic, x, 8 );
   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Reverse-interior RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Reverse-interior COO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Reverse-interior CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Reverse-interior CSC action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      bsr,
      x,
      expected,
      "Reverse-interior BSR action disagrees with MakeGenericOperator." ) && success;

   return success;
}

bool TestPartitionScalarCellRawCOOUsesAlgebraicExtents()
{
   Cartesian1DMesh mesh0(0.5, 2);
   Cartesian1DMesh mesh1(1.0, 1);
   auto partition = MakePartition(
      MakeCellPart(mesh0),
      MakeCellPart(mesh1));

   auto p0 = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto p1 = MakeLegendreFiniteElement(FiniteElementOrders<1>{});
   auto trial_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{p0, p1},
      DGDirectSumNumbering{});
   auto test_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{p1, p0},
      DGDirectSumNumbering{});

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto weak_form = integrate(Cells<"mesh">{}, u * v);
   auto context = MakeWeakFormContext(
      MakeTrialField<"u">(trial_space),
      MakeTestField<"v">(test_space),
      MakeIntegrationDomain<"mesh">(partition));
   auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3>{});

   auto raw = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      SerialKernelConfiguration>(
         weak_form,
         context,
         rule);
   auto coo = FinalizeRawCOOToCOOHost(raw);

   bool success = true;
   success = Check(
      GetNumberOfGlobalDofs(trial_space) == 4 &&
         GetAlgebraicDofExtent(trial_space) == 4 &&
         GetNumberOfGlobalDofs(test_space) == 5 &&
         GetAlgebraicDofExtent(test_space) == 5,
      "Compact mixed L2 logical counts and algebraic extents are wrong.") && success;
   const auto& trial0 =
      trial_space.template GetCellFiniteElementSpace<0>();
   const auto& trial1 =
      trial_space.template GetCellFiniteElementSpace<1>();
   success = Check(
      GetNumberOfGlobalDofs(trial0) == 2 &&
         GetNumberOfGlobalDofs(trial1) == 2 &&
         GetAlgebraicDofExtent(trial0) == 4 &&
         GetAlgebraicDofExtent(trial1) == 4 &&
         trial1.GetNumberOfFiniteElementDofs() == 2,
      "Partition L2 leaves do not separate logical counts from the common algebraic extent.") && success;
   success = Check(
      raw.num_rows == 5 && raw.num_cols == 4 && raw.nnz_raw == 6,
      "Partition RawCOO dimensions or segmented capacity are wrong.") && success;

   const auto raw_data = GetHostReadView(raw);
   for (GlobalIndex i = 0; i < 4; ++i)
   {
      success = Check(
         raw_data.rows[i] < 4 && raw_data.cols[i] < 2,
         "Partition RawCOO part-0 coordinate escaped its direct-sum range.") && success;
   }
   for (GlobalIndex i = 4; i < 6; ++i)
   {
      success = Check(
         raw_data.rows[i] >= 4 && raw_data.rows[i] < 5 &&
            raw_data.cols[i] >= 2 && raw_data.cols[i] < 4,
         "Partition RawCOO part-1 coordinate missed its shifted algebraic range.") && success;
   }

   auto matrix_free = MakeGenericOperator<SerialKernelConfiguration>(
      weak_form,
      context,
      rule);
   Vector x(4);
   auto* x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < 4; ++i)
   {
      x_data[i] = Real(0.25) + Real(0.5) * i;
   }
   const auto expected = ApplyRectangularOperator(matrix_free, x, 5);
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Partition RawCOO action disagrees with the matrix-free operator.") && success;
   return success;
}

bool TestPartitionSharedCoordinateCellRawCOOReduction()
{
   Cartesian1DMesh mesh0( 0.5, 2 );
   Cartesian1DMesh mesh1( 1.0, 1 );
   auto partition = MakePartition(
      MakeCellPart( mesh0 ),
      MakeCellPart( mesh1 ) );

   const auto finite_element =
      MakeLobattoFiniteElement( FiniteElementOrders< 1 >{} );
   const std::array< int, 4 > part0_map{ 0, 1, 1, 2 };
   const std::array< int, 2 > part1_map{ 0, 1 };
   HostDevicePointer< const int > part0_indices{};
   HostDevicePointer< const int > part1_indices{};
   part0_indices.host_pointer = part0_map.data();
   part1_indices.host_pointer = part1_map.data();
   using Restriction =
      IndirectH1Restriction< std::index_sequence< 2 > >;
   const Restriction part0_restriction{
      part0_indices,
      0,
      4,
      3,
      3 };
   const Restriction part1_restriction{
      part1_indices,
      1,
      2,
      2,
      3 };
   auto space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ finite_element, finite_element },
      std::tuple{ part0_restriction, part1_restriction } );

   TrialSpace< "shared" > u;
   TestSpace< "shared" > v;
   const auto weak_form = integrate( Cells< "mesh" >{}, u * v );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "shared" >( space ),
      MakeIntegrationDomain< "mesh" >( partition ) );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3 >{} );

   auto raw = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule );
   auto coo = FinalizeRawCOOToCOOHost( raw );
   auto csr = FinalizeRawCOOToCSRHost( raw );
   auto csc = FinalizeRawCOOToCSCHost( raw );
   auto matrix_free = MakeGenericOperator< SerialKernelConfiguration >(
      weak_form,
      context,
      rule );

   bool success = true;
   success = Check(
      GetNumberOfGlobalDofs( space ) == 5 &&
         GetAlgebraicDofExtent( space ) == 3,
      "Shared mixed space did not preserve logical counts and common algebraic extent." ) && success;
   success = Check(
      raw.num_rows == 3 && raw.num_cols == 3 && raw.nnz_raw == 12,
      "Shared mixed RawCOO dimensions or exact capacity are wrong." ) && success;
   success = Check(
      HasDuplicateCoordinate( raw ) && raw.nnz_raw > coo.nnz,
      "Shared mixed RawCOO coordinates were not canonically reducible." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( 3 );
   FillDeterministicInput( x );
   const auto expected = ApplyRectangularOperator( matrix_free, x, 3 );
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Shared mixed COO action disagrees with the matrix-free operator." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Shared mixed CSR action disagrees with the matrix-free operator." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Shared mixed CSC action disagrees with the matrix-free operator." ) && success;
   return success;
}

bool TestPartitionHeterogeneousGlobalFacetRawCOO()
{
   Cartesian1DMesh left_mesh( 0.5, 2 );
   Cartesian1DMesh right_mesh( 1.0, 1 );
   const auto left_boundary_faces =
      MakeCartesianBoundaryFaceConnectivity< 1 >( { 2 } );
   const auto right_boundary_faces =
      MakeCartesianBoundaryFaceConnectivity< 1 >( { 1 } );
   const auto left_interior_faces =
      MakeCartesianInteriorFaceConnectivity< 1 >( { 2 } );
   const auto right_interior_faces =
      MakeCartesianInteriorFaceConnectivity< 1 >( { 1 } );
   const CartesianIntermeshFaceConnectivity< 1, 1 > interface_faces(
      { 2 },
      { 1 } );
   const auto partition = MakePartition(
      MakeCellPart( left_mesh ),
      MakeCellPart( right_mesh ),
      MakeBoundaryFacePart< 1 >( right_boundary_faces ),
      MakeInteriorFacePart< 0, 0 >( left_interior_faces ),
      MakeInteriorFacePart< 1, 1 >( right_interior_faces ),
      MakeInteriorFacePart< 0, 1 >( interface_faces ),
      MakeBoundaryFacePart< 0 >( left_boundary_faces ) );

   const auto p0 =
      MakeLegendreFiniteElement( FiniteElementOrders< 0 >{} );
   const auto p1 =
      MakeLegendreFiniteElement( FiniteElementOrders< 1 >{} );
   const auto p2 =
      MakeLegendreFiniteElement( FiniteElementOrders< 2 >{} );
   const auto trial_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ p1, p2 },
      DGDirectSumNumbering{} );
   const auto test_space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ p2, p0 },
      DGDirectSumNumbering{} );

   TrialSpace< "partition_facet_u" > u;
   TestSpace< "partition_facet_v" > v;
   const auto weak_form =
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v ) +
      integrate(
         InteriorFacets< "mesh" >{},
         jump( u ) * jump( v ) );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "partition_facet_u" >( trial_space ),
      MakeTestField< "partition_facet_v" >( test_space ),
      MakeIntegrationDomain< "mesh" >( partition ) );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4 >{} );

   const auto raw = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule );
   const auto coo = FinalizeRawCOOToCOOHost( raw );
   const auto csr = GenericAssembly<
      MatrixAssemblyType::CSR,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule );
   const auto csc = GenericAssembly<
      MatrixAssemblyType::CSC,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule );
#ifdef GENDIL_USE_HYPRE
   const auto hypre = GenericAssembly<
      MatrixAssemblyType::HypreCSR,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule,
         HypreCSRHostBackend{} );
#endif
   const auto matrix_free =
      MakeGenericOperator< SerialKernelConfiguration >(
         weak_form,
         context,
         rule );

   constexpr GlobalIndex expected_boundary_entries = 18;
   constexpr GlobalIndex expected_left_interior_entries = 24;
   constexpr GlobalIndex expected_cross_entries = 20;
   constexpr GlobalIndex expected_raw_entries =
      expected_boundary_entries +
      expected_left_interior_entries +
      expected_cross_entries;
   bool success = true;
   success = Check(
      raw.num_rows == 7 && raw.num_cols == 7,
      "Heterogeneous partition facet RawCOO dimensions are wrong." ) && success;
   success = Check(
      raw.nnz_raw == expected_raw_entries,
      "Heterogeneous partition facet RawCOO segment capacity is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;

   const auto raw_data = GetHostReadView( raw );
   const auto check_segment = [&] (
      const GlobalIndex begin,
      const GlobalIndex end,
      const GlobalIndex row_begin,
      const GlobalIndex row_end,
      const GlobalIndex col_begin,
      const GlobalIndex col_end,
      const char* message )
   {
      bool segment_success = true;
      for ( GlobalIndex i = begin; i < end; ++i )
      {
         segment_success = Check(
            raw_data.rows[i] >= row_begin &&
               raw_data.rows[i] < row_end &&
               raw_data.cols[i] >= col_begin &&
               raw_data.cols[i] < col_end,
            message ) && segment_success;
      }
      return segment_success;
   };

   // Boundary parts are ordered right/right/left/left by MakePartition.
   success = check_segment(
      0, 6, 6, 7, 4, 7,
      "Right boundary segment escaped the right direct-sum ranges." ) && success;
   success = check_segment(
      6, 18, 0, 6, 0, 4,
      "Left boundary segment escaped the left direct-sum ranges." ) && success;
   success = check_segment(
      18, 42, 0, 6, 0, 4,
      "Same-part interior segment escaped the left direct-sum ranges." ) && success;

   // Cross-part storage is --, -+, +-, ++, with exact heterogeneous sizes.
   success = check_segment(
      42, 48, 0, 6, 0, 4,
      "Cross-part -- segment has the wrong algebraic ranges." ) && success;
   success = check_segment(
      48, 57, 0, 6, 4, 7,
      "Cross-part -+ segment has the wrong algebraic ranges." ) && success;
   success = check_segment(
      57, 59, 6, 7, 0, 4,
      "Cross-part +- segment has the wrong algebraic ranges." ) && success;
   success = check_segment(
      59, 62, 6, 7, 4, 7,
      "Cross-part ++ segment has the wrong algebraic ranges." ) && success;
   success = Check(
      HasDuplicateCoordinate( raw ) && raw.nnz_raw > coo.nnz,
      "Partition facet RawCOO duplicates were not reduced." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( raw.num_cols );
   FillDeterministicInput( x );
   const auto expected =
      ApplyRectangularOperator( matrix_free, x, raw.num_rows );
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Partition facet COO action disagrees with matrix-free." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Partition facet CSR action disagrees with matrix-free." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Partition facet CSC action disagrees with matrix-free." ) && success;
#ifdef GENDIL_USE_HYPRE
   success = CheckRectangularAction(
      hypre,
      x,
      expected,
      "Partition facet process-local HypreCSR action disagrees with matrix-free." ) && success;
#endif
   return success;
}

bool TestPartitionH1GlobalFacetAliasesAndExplicitPlacement()
{
   Cartesian1DMesh mesh( 0.5, 2 );
   const auto partition = MakePartition(
      MakeCellPart( mesh ),
      MakeInteriorFacePart< 0, 0 >(
         MakeCartesianInteriorFaceConnectivity< 1 >( { 2 } ) ),
      MakeBoundaryFacePart< 0 >(
         MakeCartesianBoundaryFaceConnectivity< 1 >( { 2 } ) ) );
   const auto finite_element =
      MakeLobattoFiniteElement( FiniteElementOrders< 1 >{} );
   const std::array< int, 4 > h1_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > indices{};
   indices.host_pointer = h1_map.data();
   using Restriction =
      IndirectH1Restriction< std::index_sequence< 2 > >;
   const Restriction restriction{
      indices,
      2,
      4,
      3,
      5 };
   const auto space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ finite_element },
      std::tuple{ restriction } );

   TrialSpace< "partition_h1" > u;
   TestSpace< "partition_h1" > v;
   const auto weak_form =
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v ) +
      integrate(
         InteriorFacets< "mesh" >{},
         average( u ) * average( v ) );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "partition_h1" >( space ),
      MakeIntegrationDomain< "mesh" >( partition ) );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 3 >{} );
   const auto raw = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      SerialKernelConfiguration >(
         weak_form,
         context,
         rule );
   const auto coo = FinalizeRawCOOToCOOHost( raw );
   const auto matrix_free =
      MakeGenericOperator< SerialKernelConfiguration >(
         weak_form,
         context,
         rule );

   bool success = true;
   success = Check(
      raw.num_rows == 5 && raw.num_cols == 5 && raw.nnz_raw == 24,
      "Partition H1 facet dimensions or segment capacity are wrong." ) && success;
   const auto raw_data = GetHostReadView( raw );
   for ( GlobalIndex i = 0; i < raw.nnz_raw; ++i )
   {
      success = Check(
         raw_data.rows[i] >= 2 && raw_data.rows[i] < 5 &&
            raw_data.cols[i] >= 2 && raw_data.cols[i] < 5,
         "Partition H1 facet coordinate ignored explicit placement." ) && success;
   }
   success = Check(
      HasDuplicateCoordinate( raw ) && raw.nnz_raw > coo.nnz,
      "Partition H1 facet aliases were not retained as RawCOO duplicates." ) && success;

   Vector input( 5 );
   FillDeterministicInput( input );
   const auto expected = ApplyRectangularOperator( matrix_free, input, 5 );
   success = CheckRectangularAction(
      coo,
      input,
      expected,
      "Explicitly placed partition H1 facet COO disagrees with matrix-free." ) && success;
   return success;
}

bool TestPartitionVectorTensorProductRawCOOAndCSRAgainstGeneric()
{
   constexpr GlobalIndex nx = 2;
   constexpr GlobalIndex nv_part = 2;
   const Cartesian1DMesh spatial_mesh( Real{1} / nx, nx );
   const Cartesian1DMesh velocity_minus(
      Real{0.25}, nv_part, Real{0} );
   const Cartesian1DMesh velocity_plus(
      Real{0.25}, nv_part, Real{0.5} );

   const auto spatial_partition = MakePartition(
      MakeCellPart( spatial_mesh ),
      MakeInteriorFacePart< 0, 0 >(
         MakeCartesianInteriorFaceConnectivity< 1 >( { nx } ) ),
      MakeBoundaryFacePart< 0 >(
         MakeCartesianBoundaryFaceConnectivity< 1 >( { nx } ) ) );
   const auto velocity_partition = MakePartition(
      MakeCellPart( velocity_minus ),
      MakeCellPart( velocity_plus ),
      MakeInteriorFacePart< 0, 0 >(
         MakeCartesianInteriorFaceConnectivity< 1 >( { nv_part } ) ),
      MakeInteriorFacePart< 1, 1 >(
         MakeCartesianInteriorFaceConnectivity< 1 >( { nv_part } ) ),
      MakeInteriorFacePart< 0, 1 >(
         CartesianIntermeshFaceConnectivity< 1, 1 >(
            { nv_part },
            { nv_part } ) ),
      MakeBoundaryFacePart< 0 >(
         CartesianBoundaryFaceConnectivity< 1, 0 >( { nv_part } ) ),
      MakeBoundaryFacePart< 1 >(
         CartesianBoundaryFaceConnectivity< 1, 1 >( { nv_part } ) ) );
   const auto phase_partition =
      MakeCartesianProductPartition(
         spatial_partition,
         velocity_partition );
   static_assert( decltype( phase_partition )::num_cell_parts == 2 );

   const auto spatial_scalar_h1 =
      MakeLobattoFiniteElement( FiniteElementOrders< 1 >{} );
   const auto spatial_vector_h1 =
      MakeVectorFiniteElement(
         spatial_scalar_h1,
         spatial_scalar_h1 );
   const std::array< int, 4 > spatial_h1_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > spatial_h1_indices{};
   spatial_h1_indices.host_pointer = spatial_h1_map.data();
   const auto spatial_vector_space = MakeFiniteElementSpace(
      spatial_mesh,
      spatial_vector_h1,
      MakeVectorIndirectH1RestrictionSpecification< 2 >(
         IndirectH1RestrictionSpecification{
            spatial_h1_indices,
            3 } ) );

   const auto velocity_dg =
      MakeLegendreFiniteElement( FiniteElementOrders< 1 >{} );
   constexpr GlobalIndex velocity_dofs_per_part = 2 * nv_part;
   constexpr GlobalIndex velocity_extent = 2 * velocity_dofs_per_part;
   const auto velocity_minus_space = MakeFiniteElementSpace(
      velocity_minus,
      velocity_dg,
      ContiguousL2RestrictionSpecification{
         0,
         velocity_extent } );
   const auto velocity_plus_space = MakeFiniteElementSpace(
      velocity_plus,
      velocity_dg,
      ContiguousL2RestrictionSpecification{
         velocity_dofs_per_part,
         velocity_extent } );
   const auto f1_minus_restriction = MakeTensorProductRestriction(
      spatial_vector_space,
      velocity_minus_space );
   const auto f1_plus_restriction = MakeTensorProductRestriction(
      spatial_vector_space,
      velocity_plus_space );

   using ScalarF1Shape = TensorShapeFunctions<
      GaussLobattoLegendreShapeFunctions< 1 >,
      GaussLegendreShapeFunctions< 1 > >;
   const auto scalar_f1_finite_element =
      FiniteElement< HyperCube< 2 >, ScalarF1Shape >{};
   const auto vector_f1_finite_element = MakeVectorFiniteElement(
      scalar_f1_finite_element,
      scalar_f1_finite_element );
   const auto f1_space = MakeMixedFiniteElementSpace(
      phase_partition,
      std::tuple{
         vector_f1_finite_element,
         vector_f1_finite_element },
      std::tuple{
         f1_minus_restriction,
         f1_plus_restriction } );

   const auto f0_finite_element =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   const auto f0_space = MakeMixedFiniteElementSpace(
      phase_partition,
      std::tuple{ f0_finite_element, f0_finite_element },
      DGDirectSumNumbering{} );

   Cells< "phase" > cells;
   InteriorFacets< "phase" > interior_facets;
   BoundaryFacets< "phase" > boundary_facets;
   VectorTrialSpace< "f1" > f1;
   VectorTestSpace< "f1" > g1;
   TrialSpace< "f0" > f0;
   TestSpace< "f0" > g0;
   const auto beta = MakeVectorCoefficient< "partition_vector_beta" >(
      [] GENDIL_HOST_DEVICE ()
      {
         return std::array< Real, 2 >{ Real{0}, Real{1} };
      } );
   const auto one = MakeCoefficient< "partition_vector_one" >(
      [] GENDIL_HOST_DEVICE () { return Real{1}; } );
   const auto spatial_projector =
      MakeMatrixCoefficient< "partition_spatial_projector" >(
         [] GENDIL_HOST_DEVICE ()
         {
            SerialRecursiveArray< Real, 2, 2 > value{};
            value( 0, 0 ) = Real{1};
            value( 1, 1 ) = Real{1};
            return value;
         } );

   const auto diagonal_form =
      integrate(
         cells,
         -inner( outer( f1, beta ), grad( g1 ) ) +
            one * dot( f1, g1 ) ) +
      integrate(
         interior_facets,
         dot( upwind( beta, f1 ), jump( g1 ) ) ) +
      integrate(
         boundary_facets,
         dot( f1, g1 ) * dot( beta, Normal{} ) );
   const auto divergence_form = integrate(
      cells,
      inner( spatial_projector, grad( f1 ) ) * g0 );
   const auto gradient_form = integrate(
      cells,
      -f0 * inner( spatial_projector, grad( g1 ) ) );

   const auto diagonal_context = MakeWeakFormContext(
      MakeTrialField< "f1" >( f1_space ),
      MakeIntegrationDomain< "phase" >( phase_partition ) );
   const auto divergence_context = MakeWeakFormContext(
      MakeTrialField< "f1" >( f1_space ),
      MakeTestField< "f0" >( f0_space ),
      MakeIntegrationDomain< "phase" >( phase_partition ) );
   const auto gradient_context = MakeWeakFormContext(
      MakeTrialField< "f0" >( f0_space ),
      MakeTestField< "f1" >( f1_space ),
      MakeIntegrationDomain< "phase" >( phase_partition ) );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );

   bool success = true;
   const auto check_form = [&] (
      const auto & form,
      const auto & context,
      const GlobalIndex num_rows,
      const GlobalIndex num_cols,
      const bool expect_duplicates,
      const char * dimension_message,
      const char * coo_message,
      const char * csr_message )
   {
      const auto raw = GenericAssembly<
         MatrixAssemblyType::RawCOO,
         SerialKernelConfiguration >( form, context, rule );
      const auto coo = FinalizeRawCOOToCOOHost( raw );
      const auto csr = GenericAssembly<
         MatrixAssemblyType::CSR,
         SerialKernelConfiguration >( form, context, rule );
      const auto matrix_free =
         MakeGenericOperator< SerialKernelConfiguration >(
            form,
            context,
            rule );

      bool form_success = true;
      form_success = Check(
         raw.num_rows == num_rows && raw.num_cols == num_cols,
         dimension_message ) && form_success;
      form_success =
         CheckRawTripletRangesAndFinite( raw ) && form_success;
      form_success =
         CheckCanonicalCOOSortedUnique( coo ) && form_success;
      if ( expect_duplicates )
      {
         form_success = Check(
            HasDuplicateCoordinate( raw ) && raw.nnz_raw > coo.nnz,
            "Partition vector tensor-product duplicates were not reduced." ) &&
            form_success;
      }

      Vector input( num_cols );
      FillDeterministicInput( input );
      const auto expected =
         ApplyRectangularOperator( matrix_free, input, num_rows );
      form_success = Check(
         Dot( expected, expected ) > Real{1.0e-16},
         "Partition vector tensor-product action is unexpectedly zero." ) &&
         form_success;
      form_success = CheckRectangularAction(
         coo,
         input,
         expected,
         coo_message ) && form_success;
      form_success = CheckRectangularAction(
         csr,
         input,
         expected,
         csr_message ) && form_success;
      return form_success;
   };

   const GlobalIndex f0_extent = GetAlgebraicDofExtent( f0_space );
   const GlobalIndex f1_extent = GetAlgebraicDofExtent( f1_space );
   success = Check(
      f0_extent == 32 && f1_extent == 48,
      "Partition vector tensor-product algebraic extents are wrong." ) &&
      success;
   success = check_form(
      diagonal_form,
      diagonal_context,
      f1_extent,
      f1_extent,
      true,
      "Partition vector diagonal RawCOO dimensions are wrong.",
      "Partition vector diagonal COO action disagrees with matrix-free.",
      "Partition vector diagonal CSR action disagrees with matrix-free." ) &&
      success;
   success = check_form(
      divergence_form,
      divergence_context,
      f0_extent,
      f1_extent,
      false,
      "Partition vector divergence RawCOO dimensions are wrong.",
      "Partition vector divergence COO action disagrees with matrix-free.",
      "Partition vector divergence CSR action disagrees with matrix-free." ) &&
      success;
   success = check_form(
      gradient_form,
      gradient_context,
      f1_extent,
      f0_extent,
      false,
      "Partition vector gradient RawCOO dimensions are wrong.",
      "Partition vector gradient COO action disagrees with matrix-free.",
      "Partition vector gradient CSR action disagrees with matrix-free." ) &&
      success;
   return success;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef GENDIL_USE_HYPRE
   hypre_MPI_Init(&argc, &argv);
#else
   (void)argc;
   (void)argv;
#endif
   bool success = true;
#ifdef GENDIL_USE_HYPRE
   {
      HypreSession hypre;
#endif
   success = TestRawCOOBufferAllocation() && success;
   success = TestRawCOOToCOOFinalization() && success;
   success = TestScalarL2CellMassRawCOOAgainstBSR() && success;
   success = TestVectorL2CellMassRawCOOAgainstSGBSR() && success;
   success = TestVectorBoundaryFaceMassCOOAgainstSGBSR() && success;
   success = TestScalarH1CellMassRawCOOPreservesDuplicatesAgainstSGBSR() && success;
   success = TestVectorH1CellMassRawCOOAgainstDenseReference() && success;
   success = TestScalarP0InteriorJumpAnalyticRawCOO() && success;
   success = TestVectorP0InteriorJumpAnalyticRawCOO() && success;
   success = TestScalarBoundaryFaceMassCOOAgainstGenericAndBSR() && success;
   success = TestScalarInteriorJumpCOOAgainstGenericAndBSR() && success;
   success = TestScalarCombinedFaceCOOOffsetsAndAccumulation() && success;
   success = TestTensorProductFacetRawCOOAndCSRAgainstGeneric() && success;
   success = TestScalarH1FacetRawCOOAndCSRAgainstGeneric() && success;
   success = TestVectorH1FacetRawCOOAndCSRAgainstGeneric() && success;
   success = TestRectangularP1TrialP2TestCellAssembly() && success;
   success = TestRectangularP1TrialP2TestAffineBoundaryAssembly() && success;
   success = TestRectangularP2TrialP1TestOrientedInteriorAssembly() && success;
   success = TestPartitionScalarCellRawCOOUsesAlgebraicExtents() && success;
   success = TestPartitionSharedCoordinateCellRawCOOReduction() && success;
   success = TestPartitionHeterogeneousGlobalFacetRawCOO() && success;
   success = TestPartitionH1GlobalFacetAliasesAndExplicitPlacement() && success;
   success = TestPartitionVectorTensorProductRawCOOAndCSRAgainstGeneric() && success;
#ifdef GENDIL_USE_HYPRE
   }
   hypre_MPI_Finalize();
#endif

   return success ? 0 : 1;
}
