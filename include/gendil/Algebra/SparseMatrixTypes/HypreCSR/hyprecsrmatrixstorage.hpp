// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixstorage.hpp"
#include "gendil/Interfaces/Hypre/hypreparcsrview.hpp"
#include "gendil/Interfaces/Hypre/hypretypes.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * @brief Serial Hypre ParCSR metadata carried alongside GenDiL-owned CSR data.
 *
 * `HypreCSRMatrix` owns ordinary GenDiL CSR storage, but its lazy
 * `HYPRE_ParCSRMatrix` shell still needs the global dimensions, partition
 * arrays, and communicator expected by Hypre. In the v1 interface these values
 * describe a rank-1, host-memory matrix on `hypre_MPI_COMM_SELF`: local rows and
 * columns are the full global matrix, and the two-entry start arrays are
 * `[0, global_size]`.
 *
 * The diagonal fields are GenDiL-side bookkeeping produced by
 * `FinalizeRawCOOToHypreCSR`. They let matvec-oriented code accept rectangular
 * or missing-diagonal matrices while allowing future solver wrappers to reject
 * matrices that are not suitable for Hypre preconditioners/solvers.
 */
struct HypreCSRMetadata
{
   /// Global row count passed to `HYPRE_ParCSRMatrixCreate`.
   HYPRE_BigInt global_num_rows = 0;

   /// Global column count passed to `HYPRE_ParCSRMatrixCreate`.
   HYPRE_BigInt global_num_cols = 0;

   /// Row ownership range for this rank; serial v1 uses `{0, global_num_rows}`.
   HYPRE_BigInt row_starts[2] = { 0, 0 };

   /// Column ownership range for this rank; serial v1 uses `{0, global_num_cols}`.
   HYPRE_BigInt col_starts[2] = { 0, 0 };

   /// Communicator associated with the Hypre shell; v1 uses `hypre_MPI_COMM_SELF`.
   MPI_Comm comm = hypre_MPI_COMM_SELF;

   /// True when the local/global matrix dimensions are square.
   bool is_square = false;

   /// True when every eligible diagonal row has an explicit diagonal entry.
   bool has_explicit_diagonal = false;

   /// Number of rows that can have a diagonal entry, `min(num_rows, num_cols)`.
   HYPRE_Int diagonal_rows = 0;

   /// Count of eligible rows where an explicit diagonal entry was found.
   HYPRE_Int explicit_diagonal_count = 0;

   /// Count of eligible rows where no explicit diagonal entry was found.
   HYPRE_Int missing_diagonal_count = 0;

   /// First eligible row missing an explicit diagonal, or `-1` when none is missing.
   HYPRE_Int first_missing_diagonal = -1;
};

template <
   typename Backend =
#ifdef GENDIL_USE_HYPRE_DEVICE
      HypreCSRDeviceBackend
#else
      HypreCSRHostBackend
#endif
   >
struct HypreCSRMatrix
{
   using value_type = HYPRE_Complex;
   using index_type = HYPRE_Int;
   using backend_type = Backend;
   using csr_type = CSRMatrix< HYPRE_Complex, HYPRE_Int, HostCSRBackend<> >;

   csr_type csr;
   HypreCSRMetadata metadata;
   backend_type backend{};
   mutable HYPRE_ParCSRMatrix host_parcsr = nullptr;
   mutable HYPRE_ParCSRMatrix device_parcsr = nullptr;

   HypreCSRMatrix() = default;

   HypreCSRMatrix(
      csr_type csr_,
      HypreCSRMetadata metadata_,
      Backend backend_ = Backend{} )
   : csr( std::move( csr_ ) ),
     metadata( metadata_ ),
     backend( backend_ )
   { }

   HypreCSRMatrix( const HypreCSRMatrix & ) = delete;
   HypreCSRMatrix & operator=( const HypreCSRMatrix & ) = delete;

   HypreCSRMatrix( HypreCSRMatrix && other ) noexcept
   : csr( std::move( other.csr ) ),
     metadata( other.metadata ),
     backend( other.backend ),
     host_parcsr( other.host_parcsr ),
     device_parcsr( other.device_parcsr )
   {
      other.host_parcsr = nullptr;
      other.device_parcsr = nullptr;
   }

   HypreCSRMatrix & operator=( HypreCSRMatrix && other ) noexcept
   {
      if ( this != &other )
      {
         details::DestroyHypreParCSRView( host_parcsr );
         details::DestroyHypreParCSRView( device_parcsr );
         FreeCSRMatrix( csr );

         csr = std::move( other.csr );
         metadata = other.metadata;
         backend = other.backend;
         host_parcsr = other.host_parcsr;
         device_parcsr = other.device_parcsr;
         other.host_parcsr = nullptr;
         other.device_parcsr = nullptr;
      }
      return *this;
   }

   ~HypreCSRMatrix()
   {
      details::DestroyHypreParCSRView( host_parcsr );
      details::DestroyHypreParCSRView( device_parcsr );
      FreeCSRMatrix( csr );
   }

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const;

   HYPRE_ParCSRMatrix GetHostHypreParCSR() const
   {
      if ( host_parcsr == nullptr )
      {
         host_parcsr =
            details::CreateHostHypreParCSRView( *this );
      }
      return host_parcsr;
   }

   HYPRE_ParCSRMatrix GetDeviceHypreParCSR() const
   {
#ifdef GENDIL_USE_HYPRE_DEVICE
      if ( device_parcsr == nullptr )
      {
         device_parcsr =
            details::CreateDeviceHypreParCSRView( *this );
      }
      return device_parcsr;
#else
      GENDIL_VERIFY(
         false,
         "HypreCSRDeviceBackend requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
      return nullptr;
#endif
   }

   HYPRE_ParCSRMatrix GetHypreParCSR() const
   {
      if constexpr ( is_host_matvec_backend_v< Backend > )
      {
         return GetHostHypreParCSR();
      }
      else if constexpr ( is_device_matvec_backend_v< Backend > )
      {
         return GetDeviceHypreParCSR();
      }
      else
      {
         static_assert(
            dependent_false_v< Backend >,
            "HypreCSRMatrix default Hypre view dispatch requires a backend inheriting HostMatVecBackend or DeviceMatVecBackend." );
      }
   }

   explicit operator HYPRE_ParCSRMatrix() const
   {
      return GetHypreParCSR();
   }
};

} // namespace gendil

#endif // GENDIL_USE_HYPRE
