// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Interfaces/Hypre/hypreerror.hpp"
#include "gendil/Interfaces/Hypre/hypretypes.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil
{
namespace details
{

inline void DetachHypreDiagAlias( HYPRE_ParCSRMatrix parcsr )
{
   if ( parcsr == nullptr )
   {
      return;
   }

   auto * matrix = reinterpret_cast< hypre_ParCSRMatrix * >( parcsr );
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( matrix );

   if ( diag != nullptr )
   {
      hypre_CSRMatrixI( diag ) = nullptr;
      hypre_CSRMatrixJ( diag ) = nullptr;
      hypre_CSRMatrixData( diag ) = nullptr;
      hypre_CSRMatrixBigJ( diag ) = nullptr;
      hypre_CSRMatrixRownnz( diag ) = nullptr;
      hypre_CSRMatrixOwnsData( diag ) = 0;
   }
}

inline void DestroyHypreParCSRView( HYPRE_ParCSRMatrix & parcsr )
{
   if ( parcsr == nullptr )
   {
      return;
   }

   DetachHypreDiagAlias( parcsr );
   CheckHypreError(
      HYPRE_ParCSRMatrixDestroy( parcsr ),
      "HYPRE_ParCSRMatrixDestroy failed" );
   parcsr = nullptr;
}

template < typename HypreMatrix >
HYPRE_ParCSRMatrix CreateHostHypreParCSRView(
   const HypreMatrix & matrix )
{
   RequireHypreInitialized(
      "HypreCSRMatrix conversion to HYPRE_ParCSRMatrix requires an active HypreSession or prior HYPRE_Initialize()." );

   HYPRE_ParCSRMatrix parcsr = nullptr;
   auto row_starts = const_cast< HYPRE_BigInt * >( matrix.metadata.row_starts );
   auto col_starts = const_cast< HYPRE_BigInt * >( matrix.metadata.col_starts );

   CheckHypreError(
      HYPRE_ParCSRMatrixCreate(
         matrix.metadata.comm,
         matrix.metadata.global_num_rows,
         matrix.metadata.global_num_cols,
         row_starts,
         col_starts,
         0,
         matrix.csr.nnz,
         0,
         &parcsr ),
      "HYPRE_ParCSRMatrixCreate failed" );

   auto * internal = reinterpret_cast< hypre_ParCSRMatrix * >( parcsr );
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( internal );

   hypre_CSRMatrixI( diag ) = matrix.csr.row_ptr.host_pointer;
   hypre_CSRMatrixJ( diag ) = matrix.csr.col_ind.host_pointer;
   hypre_CSRMatrixData( diag ) = matrix.csr.values.host_pointer;
   hypre_CSRMatrixOwnsData( diag ) = 0;
   hypre_CSRMatrixMemoryLocation( diag ) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( internal );
   if ( offd != nullptr )
   {
      hypre_CSRMatrixMemoryLocation( offd ) = HYPRE_MEMORY_HOST;
   }

   CheckHypreError(
      hypre_ParCSRMatrixInitialize_v2( internal, HYPRE_MEMORY_HOST ),
      "hypre_ParCSRMatrixInitialize_v2 failed" );

   return parcsr;
}

template < typename HypreMatrix >
HYPRE_ParCSRMatrix CreateDeviceHypreParCSRView(
   const HypreMatrix & matrix )
{
#ifdef GENDIL_USE_HYPRE_DEVICE
   RequireHypreInitialized(
      "HypreCSRMatrix device conversion to HYPRE_ParCSRMatrix requires an active HypreSession or prior HYPRE_Initialize()." );

   HYPRE_ParCSRMatrix parcsr = nullptr;
   auto row_starts = const_cast< HYPRE_BigInt * >( matrix.metadata.row_starts );
   auto col_starts = const_cast< HYPRE_BigInt * >( matrix.metadata.col_starts );

   CheckHypreError(
      HYPRE_ParCSRMatrixCreate(
         matrix.metadata.comm,
         matrix.metadata.global_num_rows,
         matrix.metadata.global_num_cols,
         row_starts,
         col_starts,
         0,
         matrix.csr.nnz,
         0,
         &parcsr ),
      "HYPRE_ParCSRMatrixCreate failed" );

   auto * internal = reinterpret_cast< hypre_ParCSRMatrix * >( parcsr );
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( internal );

   hypre_CSRMatrixI( diag ) = matrix.csr.row_ptr.device_pointer;
   hypre_CSRMatrixJ( diag ) = matrix.csr.col_ind.device_pointer;
   hypre_CSRMatrixData( diag ) = matrix.csr.values.device_pointer;
   hypre_CSRMatrixOwnsData( diag ) = 0;
   hypre_CSRMatrixMemoryLocation( diag ) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( internal );
   if ( offd != nullptr )
   {
      hypre_CSRMatrixMemoryLocation( offd ) = HYPRE_MEMORY_DEVICE;
   }

   CheckHypreError(
      hypre_ParCSRMatrixInitialize_v2( internal, HYPRE_MEMORY_DEVICE ),
      "hypre_ParCSRMatrixInitialize_v2 failed" );

   return parcsr;
#else
   (void) matrix;
   static_assert(
      dependent_false_v< HypreMatrix >,
      "CreateDeviceHypreParCSRView requires GENDIL_USE_HYPRE_DEVICE. Configure "
      "GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return nullptr;
#endif
}

} // namespace details
} // namespace gendil
