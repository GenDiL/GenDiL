// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

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

template<class Matrix>
bool CheckAction(
   const Matrix& matrix,
   const Vector& x,
   const Vector& expected,
   const char * message)
{
   Vector result(expected.Size());
   result = 0.0;
   matrix(x, result);

   bool success = true;
   const Real * result_data = result.ReadHostData();
   const Real * expected_data = expected.ReadHostData();
   for (GlobalIndex i = 0; i < expected.Size(); ++i)
   {
      success =
         Check(Near(result_data[i], expected_data[i]), message) &&
         success;
   }
   return success;
}

bool TestElementBlockDiagonalFormats()
{
   constexpr Integer num_elements = 3;
   Cartesian1DMesh mesh(
      1.0 / static_cast<Real>(num_elements),
      num_elements);
   auto fe =
      MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto fe_space = MakeFiniteElementSpace(mesh, fe);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto full_form =
      integrate(Cells<"mesh">{}, u * v) +
      integrate(
         InteriorFacets<"mesh">{},
         jump(u) * jump(v)) +
      integrate(BoundaryFacets<"mesh">{}, u * v);
   auto cell_form =
      integrate(Cells<"mesh">{}, u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeTestField<"v">(fe_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<2>{});

   using KernelPolicy = SerialKernelConfiguration;
   auto full_bsr =
      GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
         full_form,
         context,
         integration_rule);
   auto element_bsr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
   auto explicit_bsr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy>(
            full_form,
            context,
            integration_rule,
            DefaultBackendFor_t<MatrixAssemblyType::BSR>{});
   auto cell_bsr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy>(
            cell_form,
            context,
            integration_rule);
   const auto element_bsr_data = GetHostReadView( element_bsr );
   const auto full_bsr_data = GetHostReadView( full_bsr );
   const auto cell_bsr_data = GetHostReadView( cell_bsr );

   bool success = true;
   success =
      Check(
         element_bsr.num_blocks == num_elements,
         "Element BSR must contain exactly one block per element.") &&
      success;
   success =
      Check(
         element_bsr.block_rows == 1 &&
         element_bsr.block_cols == 1,
         "Element BSR has the wrong scalar p0 block dimensions.") &&
      success;

   bool facet_terms_changed_diagonal = false;
   for (GlobalIndex element = 0;
        element < num_elements;
        ++element)
   {
      success =
         Check(
            element_bsr_data.row_offsets[element] == element &&
            element_bsr_data.row_offsets[element + 1] == element + 1 &&
            element_bsr_data.col_indices[element] == element,
            "Element BSR contains a neighbor block.") &&
         success;

      const auto full_diagonal_block =
         FindBSRBlockIndex(
            full_bsr_data,
            element,
            element);
      success =
         Check(
            full_diagonal_block !=
               std::numeric_limits<GlobalIndex>::max(),
            "Full BSR is missing an element diagonal block.") &&
         success;

      const Real element_value =
         element_bsr_data.GetBlockEntry(element, 0, 0);
      const Real full_value =
         full_bsr_data.GetBlockEntry(
            full_diagonal_block,
            0,
            0);
      success =
         Check(
            Near(element_value, full_value),
            "Element BSR disagrees with the full BSR diagonal.") &&
         success;
      facet_terms_changed_diagonal =
         facet_terms_changed_diagonal ||
         !Near(
            element_value,
            cell_bsr_data.GetBlockEntry(element, 0, 0));
   }
   success =
      Check(
         facet_terms_changed_diagonal,
         "Element BSR silently omitted all facet self contributions.") &&
      success;

   auto raw =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::RawCOO,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
   const auto raw_data = GetHostReadView( raw );
   success =
      Check(
         raw.nnz_raw == num_elements,
         "Element RawCOO allocated neighbor entries.") &&
      success;
   for (GlobalIndex i = 0; i < raw.nnz_raw; ++i)
   {
      success =
         Check(
            raw_data.rows[i] == i && raw_data.cols[i] == i,
            "Element RawCOO emitted a cross-element entry.") &&
         success;
      success =
         Check(
            Near(
               raw_data.values[i],
               element_bsr_data.GetBlockEntry(i, 0, 0)),
            "Element RawCOO disagrees with element BSR.") &&
         success;
   }

   auto sgbsr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
   auto coo =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::COO,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
   auto explicit_coo =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::COO,
         KernelPolicy>(
            full_form,
            context,
            integration_rule,
            DefaultBackendFor_t<MatrixAssemblyType::COO>{});
   auto csr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::CSR,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
   auto csc =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::CSC,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
#ifdef GENDIL_USE_HYPRE
   auto hypre_csr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::HypreCSR,
         KernelPolicy>(
            full_form,
            context,
            integration_rule);
#endif

   Vector x(fe_space.GetNumberOfFiniteElementDofs());
   Real * x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < x.Size(); ++i)
   {
      x_data[i] = 0.5 + static_cast<Real>(i);
   }
   Vector expected(fe_space.GetNumberOfFiniteElementDofs());
   expected = 0.0;
   element_bsr(x, expected);

   success =
      CheckAction(
         explicit_bsr,
         x,
         expected,
         "Explicit-backend element BSR action disagrees with element BSR.") &&
      success;
   success =
      CheckAction(
         sgbsr,
         x,
         expected,
         "Element SGBSR action disagrees with element BSR.") &&
      success;
   success =
      CheckAction(
         coo,
         x,
         expected,
         "Element COO action disagrees with element BSR.") &&
      success;
   success =
      CheckAction(
         explicit_coo,
         x,
         expected,
         "Explicit-backend element COO action disagrees with element BSR.") &&
      success;
   success =
      CheckAction(
         csr,
         x,
         expected,
         "Element CSR action disagrees with element BSR.") &&
      success;
   success =
      CheckAction(
         csc,
         x,
         expected,
         "Element CSC action disagrees with element BSR.") &&
      success;
#ifdef GENDIL_USE_HYPRE
   success =
      CheckAction(
         hypre_csr,
         x,
         expected,
         "Element HypreCSR action disagrees with element BSR.") &&
      success;
#endif


   return success;
}

bool TestH1SGBSRElementBlockDiagonal()
{
   Cartesian1DMesh mesh(0.5, 2);
   auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const std::array<int, 4> restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer<const int> restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   IndirectH1RestrictionSpecification restriction{restriction_indices, 3};
   auto fe_space =
      MakeFiniteElementSpace(mesh, fe, restriction);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form =
      integrate(Cells<"mesh">{}, u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeTestField<"v">(fe_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<3>{});

   using KernelPolicy = SerialKernelConfiguration;
   auto element_matrix =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy>(
            form,
            context,
            integration_rule,
            HostBSRBackend<>{});
   auto full_matrix =
      GenericAssembly<MatrixAssemblyType::SGBSR, KernelPolicy>(
         form,
         context,
         integration_rule,
         HostBSRBackend<>{});

   Vector x(fe_space.GetNumberOfFiniteElementDofs());
   Real * x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < x.Size(); ++i)
   {
      x_data[i] = 0.25 + static_cast<Real>(i);
   }
   Vector expected(fe_space.GetNumberOfFiniteElementDofs());
   expected = 0.0;
   full_matrix(x, expected);

   return CheckAction(
      element_matrix,
      x,
      expected,
      "H1 element SGBSR action disagrees with full cell-only SGBSR.");
}

bool TestFacetOnlyRawCOOActivatesTouchedElements()
{
   constexpr Integer num_elements = 3;
   Cartesian1DMesh mesh(
      1.0 / static_cast<Real>(num_elements),
      num_elements);
   auto fe =
      MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto fe_space = MakeFiniteElementSpace(mesh, fe);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form =
      integrate(BoundaryFacets<"mesh">{}, u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(fe_space),
         MakeTestField<"v">(fe_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<2>{});

   auto raw =
      GenericRawCOOElementBlockDiagonalAssembly<
         SerialKernelConfiguration>(
            form,
            context,
            integration_rule);
   const auto raw_data = GetHostReadView( raw );

   bool success = true;
   success =
      Check(
         raw.nnz_raw == 2,
         "Boundary-only element RawCOO must activate only boundary elements.") &&
      success;
   success =
      Check(
         raw_data.rows[0] == 0 &&
         raw_data.cols[0] == 0 &&
         raw_data.rows[1] == num_elements - 1 &&
         raw_data.cols[1] == num_elements - 1,
         "Boundary-only element RawCOO activated the wrong elements.") &&
      success;

   return success;
}

bool TestRectangularElementBSR()
{
   static_assert(
      details::is_bsr_assembly_backend_compatible_v<HostBSRBackend<>, 1, 2>);
   static_assert(
      details::is_bsr_assembly_backend_compatible_v<NativeDeviceBSRBackend<>, 1, 2>);
   static_assert(
      !details::is_bsr_assembly_backend_compatible_v<CuSparseBSRBackend<>, 1, 2>);
   static_assert(
      !details::is_bsr_assembly_backend_compatible_v<RocSparseBSRBackend<>, 1, 2>);

   Cartesian1DMesh mesh(0.5, 2);
   auto trial_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<1>{});
   auto test_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto trial_space =
      MakeFiniteElementSpace(mesh, trial_fe);
   auto test_space =
      MakeFiniteElementSpace(mesh, test_fe);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form =
      integrate(Cells<"mesh">{}, u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(trial_space),
         MakeTestField<"v">(test_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<2>{});

   auto matrix =
      GenericBSRElementBlockDiagonalAssembly<
         SerialKernelConfiguration>(
            form,
            context,
            integration_rule);
   auto sparse_matrix =
      GenericBSRAssembly<
         SerialKernelConfiguration>(
            form,
            context,
            integration_rule);
   auto generic =
      MakeGenericOperator<SerialKernelConfiguration>(
         form,
         context,
         integration_rule);
   const auto matrix_data = GetHostReadView( matrix );
   const auto sparse_data = GetHostReadView( sparse_matrix );

#if defined(GENDIL_USE_DEVICE)
   static_assert(
      std::is_same_v<
         typename decltype( matrix )::backend_type,
         NativeDeviceBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( sparse_matrix )::backend_type,
         NativeDeviceBSRBackend<> > );
#else
   static_assert(
      std::is_same_v<
         typename decltype( matrix )::backend_type,
         HostBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( sparse_matrix )::backend_type,
         HostBSRBackend<> > );
#endif

   bool success = true;
   success =
      Check(
         matrix.num_blocks == 2 &&
         matrix.block_rows == 1 &&
         matrix.block_cols == 2,
         "Rectangular element BSR has the wrong shape.") &&
      success;
   for (GlobalIndex block = 0;
        block < matrix.num_blocks;
        ++block)
   {
      for (GlobalIndex col = 0;
           col < matrix.block_cols;
           ++col)
      {
         success =
            Check(
               std::isfinite(
                  matrix_data.GetBlockEntry(block, 0, col)),
               "Rectangular element BSR contains a non-finite value.") &&
            success;
      }
   }
   success =
      Check(
         sparse_matrix.block_rows == 1 &&
         sparse_matrix.block_cols == 2,
         "Rectangular GenericBSRAssembly has the wrong block shape.") &&
      success;
   success =
      Check(
         sparse_matrix.num_blocks == 4 &&
         sparse_data.row_offsets[0] == 0 &&
         sparse_data.row_offsets[1] == 2 &&
         sparse_data.row_offsets[2] == 4 &&
         sparse_data.col_indices[0] == 0 &&
         sparse_data.col_indices[1] == 1 &&
         sparse_data.col_indices[2] == 0 &&
         sparse_data.col_indices[3] == 1,
         "Rectangular full BSR does not use the domain adjacency.") &&
      success;

   Vector x(trial_space.GetNumberOfFiniteElementDofs());
   Real * x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < x.Size(); ++i)
   {
      x_data[i] = 0.3 + 0.27 * static_cast<Real>(i);
   }
   Vector expected(test_space.GetNumberOfFiniteElementDofs());
   expected = 0.0;
   generic(x, expected);
   success =
      CheckAction(
         matrix,
         x,
         expected,
         "Rectangular element BSR action disagrees with MakeGenericOperator.") &&
      success;
   success =
      CheckAction(
         sparse_matrix,
         x,
         expected,
         "Rectangular full BSR action disagrees with MakeGenericOperator.") &&
      success;

   return success;
}

bool TestRectangularElementRawCOOCSRAndCSC()
{
   Cartesian2DMesh mesh(1.0, 2, 1);
   auto trial_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<1, 1>{});
   auto test_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<2, 2>{});
   auto trial_space = MakeFiniteElementSpace(mesh, trial_fe);
   auto test_space = MakeFiniteElementSpace(mesh, test_fe);

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto source = MakeCoefficient<"source">(
      [] GENDIL_HOST_DEVICE () -> Real { return 0.625; });
   auto form =
      integrate(Cells<"mesh">{}, u * v) +
      integrate(
         BoundaryFacets<"mesh">{},
         u * v + source * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(trial_space),
         MakeTestField<"v">(test_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::RawCOO,
         KernelPolicy>(
            form,
            context,
            integration_rule);
   auto finalized = FinalizeRawCOOToCOOHost(raw);
   auto coo =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::COO,
         KernelPolicy>(
            form,
            context,
            integration_rule);
   auto csr =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::CSR,
         KernelPolicy>(
            form,
            context,
            integration_rule);
   auto csc =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::CSC,
         KernelPolicy>(
            form,
            context,
            integration_rule);
   auto generic =
      MakeGenericOperator<KernelPolicy>(
         form,
         context,
         integration_rule);

   bool success = true;
   success =
      Check(
         raw.num_rows == 18 &&
         raw.num_cols == 8 &&
         raw.nnz_raw == 72,
         "Rectangular element RawCOO does not contain two 9 x 4 blocks.") &&
      success;
   success =
      Check(
         finalized.num_rows == 18 && finalized.num_cols == 8 &&
         coo.num_rows == 18 && coo.num_cols == 8 &&
         csr.num_rows == 18 && csr.num_cols == 8 &&
         csc.num_rows == 18 && csc.num_cols == 8,
         "A derived element format lost the 18 x 8 dimensions.") &&
      success;

   Vector x(8);
   Real * x_data = x.WriteHostData();
   for (GlobalIndex i = 0; i < x.Size(); ++i)
   {
      x_data[i] = 0.4 + 0.19 * static_cast<Real>(i);
   }
   Vector zero(8);
   zero = 0.0;
   Vector fx(18);
   Vector fzero(18);
   fx = 0.0;
   fzero = 0.0;
   generic(x, fx);
   generic(zero, fzero);

   Vector expected(18);
   Real * expected_data = expected.WriteHostData();
   const Real * fx_data = fx.ReadHostData();
   const Real * fzero_data = fzero.ReadHostData();
   for (GlobalIndex i = 0; i < expected.Size(); ++i)
   {
      expected_data[i] = fx_data[i] - fzero_data[i];
   }

   success =
      CheckAction(
         finalized,
         x,
         expected,
         "Rectangular element RawCOO action disagrees with F(x) - F(0).") &&
      success;
   success =
      CheckAction(
         coo,
         x,
         expected,
         "Rectangular element COO action disagrees with F(x) - F(0).") &&
      success;
   success =
      CheckAction(
         csr,
         x,
         expected,
         "Rectangular element CSR action disagrees with F(x) - F(0).") &&
      success;
   success =
      CheckAction(
         csc,
         x,
         expected,
         "Rectangular element CSC action disagrees with F(x) - F(0).") &&
      success;

   return success;
}

} // namespace

int main(int argc, char ** argv)
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
      success = TestElementBlockDiagonalFormats() && success;
      success = TestH1SGBSRElementBlockDiagonal() && success;
      success =
         TestFacetOnlyRawCOOActivatesTouchedElements() &&
         success;
      success = TestRectangularElementBSR() && success;
      success = TestRectangularElementRawCOOCSRAndCSC() && success;
#ifdef GENDIL_USE_HYPRE
   }
   hypre_MPI_Finalize();
#endif
   return success ? 0 : 1;
}
