// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check(const bool condition, const char * message)
{
   if (!condition)
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near(const Real lhs, const Real rhs)
{
   return std::abs(lhs - rhs) < tolerance;
}

bool CheckVectorNear(
   const Vector& actual,
   const Vector& expected,
   const char * message)
{
   bool success = true;
   success = Check(actual.Size() == expected.Size(), message) && success;
   const GlobalIndex size =
      actual.Size() < expected.Size() ? actual.Size() : expected.Size();
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();
   for (GlobalIndex i = 0; i < size; ++i)
   {
      success = Check(Near(actual_data[i], expected_data[i]), message) && success;
   }
   return success;
}

template<class Matrix>
Vector ApplyOperator(
   const Matrix& matrix,
   const Vector& x,
   const GlobalIndex output_size)
{
   Vector y(output_size);
   y = 0.0;
   matrix(x, y);
   return y;
}

template<class NativeCSR>
bool CheckOwnedCSRLogicalEntries(
   const HypreCSRMatrix<HypreCSRHostBackend>& hypre,
   const NativeCSR& native)
{
   const auto hypre_csr = GetHostReadView(hypre.csr);
   const auto native_csr = GetHostReadView(native);

   bool success = true;
   success =
      Check(
         hypre_csr.num_rows == native_csr.num_rows &&
         hypre_csr.num_cols == native_csr.num_cols &&
         hypre_csr.nnz == native_csr.nnz,
         "HypreCSR owned CSR dimensions or reduced nnz disagree with native CSR.") &&
      success;

   const HYPRE_Int num_rows = hypre_csr.num_rows;
   for (HYPRE_Int row = 0; row <= num_rows; ++row)
   {
      success =
         Check(
            hypre_csr.row_ptr[row] ==
               static_cast<HYPRE_Int>(native_csr.row_ptr[row]),
            "HypreCSR owned CSR row pointers disagree with native CSR.") &&
         success;
   }

   for (HYPRE_Int row = 0; row < num_rows; ++row)
   {
      std::vector<std::pair<HYPRE_Int, Real>> hypre_entries;
      std::vector<std::pair<HYPRE_Int, Real>> native_entries;
      for (HYPRE_Int i = hypre_csr.row_ptr[row];
           i < hypre_csr.row_ptr[row + 1];
           ++i)
      {
         hypre_entries.emplace_back(
            hypre_csr.col_ind[i],
            static_cast<Real>(hypre_csr.values[i]));
      }
      for (GlobalIndex i = native_csr.row_ptr[row];
           i < native_csr.row_ptr[row + 1];
           ++i)
      {
         native_entries.emplace_back(
            static_cast<HYPRE_Int>(native_csr.col_ind[i]),
            native_csr.values[i]);
      }
      std::sort(hypre_entries.begin(), hypre_entries.end());
      std::sort(native_entries.begin(), native_entries.end());
      success =
         Check(
            hypre_entries.size() == native_entries.size(),
            "HypreCSR owned CSR row nnz disagrees with native CSR.") &&
         success;
      const size_t count =
         std::min(hypre_entries.size(), native_entries.size());
      for (size_t i = 0; i < count; ++i)
      {
         success =
            Check(
               hypre_entries[i].first == native_entries[i].first &&
               Near(hypre_entries[i].second, native_entries[i].second),
               "HypreCSR owned CSR logical entries disagree with native CSR.") &&
            success;
      }
   }

   return success;
}

bool CheckSerialMetadata(
   const HypreCSRMatrix<HypreCSRHostBackend>& matrix,
   const HYPRE_Int expected_rows,
   const HYPRE_Int expected_cols)
{
   const auto& metadata = matrix.metadata;
   const auto csr = GetHostReadView(matrix.csr);
   const HYPRE_Int eligible = std::min(expected_rows, expected_cols);
   HYPRE_Int explicit_count = 0;
   HYPRE_Int first_missing = -1;
   for (HYPRE_Int row = 0; row < eligible; ++row)
   {
      bool found = false;
      for (HYPRE_Int i = csr.row_ptr[row]; i < csr.row_ptr[row + 1]; ++i)
      {
         found = found || csr.col_ind[i] == row;
      }
      if (found)
      {
         ++explicit_count;
      }
      else if (first_missing < 0)
      {
         first_missing = row;
      }
   }

   bool success = true;
   success =
      Check(
         metadata.global_num_rows == expected_rows &&
         metadata.global_num_cols == expected_cols,
         "HypreCSR did not preserve independent global dimensions.") &&
      success;
   success =
      Check(
         metadata.row_starts[0] == 0 &&
         metadata.row_starts[1] == expected_rows &&
         metadata.col_starts[0] == 0 &&
         metadata.col_starts[1] == expected_cols,
         "HypreCSR rank-one ownership ranges are incorrect.") &&
      success;
   success =
      Check(
         metadata.comm == hypre_MPI_COMM_SELF,
         "HypreCSR Phase 0 metadata must use the rank-one communicator.") &&
      success;
   success =
      Check(!metadata.is_square, "Rectangular HypreCSR is marked square.") &&
      success;
   success =
      Check(
         metadata.diagonal_rows == eligible &&
         metadata.explicit_diagonal_count == explicit_count &&
         metadata.missing_diagonal_count == eligible - explicit_count &&
         metadata.first_missing_diagonal == first_missing &&
         metadata.has_explicit_diagonal == (first_missing < 0),
         "HypreCSR rank-one diagonal bookkeeping is inconsistent with owned CSR.") &&
      success;
   return success;
}

template<Integer TrialOrder, Integer TestOrder>
bool TestRectangularDirection(
   const HYPRE_Int expected_rows,
   const HYPRE_Int expected_cols)
{
   Cartesian2DMesh mesh(1.0, 2, 1);
   auto trial_fe =
      MakeLegendreFiniteElement(
         FiniteElementOrders<TrialOrder, TrialOrder>{});
   auto test_fe =
      MakeLegendreFiniteElement(
         FiniteElementOrders<TestOrder, TestOrder>{});
   auto trial_space = MakeFiniteElementSpace(mesh, trial_fe);
   auto test_space = MakeFiniteElementSpace(mesh, test_fe);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form =
      integrate(Cells<"mesh">{}, u * v) +
      integrate(Cells<"mesh">{}, 0.375 * u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(trial_space),
         MakeTestField<"v">(test_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   using KernelPolicy = SerialKernelConfiguration;

   auto native =
      GenericAssembly<MatrixAssemblyType::CSR, KernelPolicy>(
         form,
         context,
         integration_rule,
         HostCSRBackend<>{});
   auto raw =
      GenericAssembly<MatrixAssemblyType::RawCOO, KernelPolicy>(
         form,
         context,
         integration_rule);
   auto hypre =
      GenericAssembly<MatrixAssemblyType::HypreCSR, KernelPolicy>(
         form,
         context,
         integration_rule,
         HypreCSRHostBackend{});
   auto generic =
      MakeGenericOperator<KernelPolicy>(
         form,
         context,
         integration_rule);

   bool success = true;
   success =
      CheckOwnedCSRLogicalEntries(hypre, native) && success;
   success =
      CheckSerialMetadata(hypre, expected_rows, expected_cols) && success;

   // Duplicate every FE triplet and split its value between the copies. The
   // owned HypreCSR must reduce back to exactly the native logical matrix.
   auto duplicated =
      MakeRawCOOTripletBuffer<Real, GlobalIndex>(
         raw.num_rows,
         raw.num_cols,
         2 * raw.nnz_raw);
   const auto raw_data = GetHostReadView(raw);
   auto duplicated_data = GetHostReadWriteView(duplicated);
   for (GlobalIndex i = 0; i < raw.nnz_raw; ++i)
   {
      for (GlobalIndex copy = 0; copy < 2; ++copy)
      {
         const GlobalIndex target = 2 * i + copy;
         duplicated_data.rows[target] = raw_data.rows[i];
         duplicated_data.cols[target] = raw_data.cols[i];
         duplicated_data.values[target] = 0.5 * raw_data.values[i];
      }
   }
   auto duplicate_reduced =
      FinalizeRawCOOToHypreCSRHost(
         duplicated,
         HypreCSRHostBackend{});
   success =
      Check(
         duplicate_reduced.csr.nnz < duplicated.nnz_raw,
         "HypreCSR did not reduce duplicated FE triplets.") &&
      success;
   success =
      CheckOwnedCSRLogicalEntries(duplicate_reduced, native) && success;

   Vector x(expected_cols);
   Real * x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < x.Size(); ++i)
   {
      x_data[i] =
         0.45 + 0.13 * static_cast<Real>(i) +
         0.009 * static_cast<Real>(i * i);
   }
   const auto expected = ApplyOperator(generic, x, expected_rows);
   const auto native_result = ApplyOperator(native, x, expected_rows);
   const auto hypre_result = ApplyOperator(hypre, x, expected_rows);
   success =
      CheckVectorNear(
         native_result,
         expected,
         "Native rectangular CSR action disagrees with MakeGenericOperator.") &&
      success;
   success =
      CheckVectorNear(
         hypre_result,
         expected,
         "Rank-one rectangular HypreCSR action disagrees with MakeGenericOperator.") &&
      success;

#ifdef GENDIL_USE_HYPRE_DEVICE
   auto device_hypre =
      GenericAssembly<MatrixAssemblyType::HypreCSR, KernelPolicy>(
         form,
         context,
         integration_rule,
         HypreCSRDeviceBackend{});
   const auto device_result =
      ApplyOperator(device_hypre, x, expected_rows);
   success =
      CheckVectorNear(
         device_result,
         expected,
         "Device rectangular HypreCSR action disagrees with MakeGenericOperator.") &&
      success;
#endif

   return success;
}

} // namespace

int main(int argc, char ** argv)
{
   hypre_MPI_Init(&argc, &argv);

   bool success = true;
   {
      HypreSession hypre;
      success = TestRectangularDirection<1, 2>(18, 8) && success;
      success = TestRectangularDirection<2, 1>(8, 18) && success;
   }

   hypre_MPI_Finalize();
   return success ? 0 : 1;
}
