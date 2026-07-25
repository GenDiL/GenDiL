// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

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
            element_bsr.row_offsets[element] == element &&
            element_bsr.row_offsets[element + 1] == element + 1 &&
            element_bsr.col_indices[element] == element,
            "Element BSR contains a neighbor block.") &&
         success;

      const auto full_diagonal_block =
         FindBSRBlockIndex(
            full_bsr,
            element,
            element);
      success =
         Check(
            full_diagonal_block !=
               std::numeric_limits<GlobalIndex>::max(),
            "Full BSR is missing an element diagonal block.") &&
         success;

      const Real element_value =
         element_bsr.GetBlockEntry(element, 0, 0);
      const Real full_value =
         full_bsr.GetBlockEntry(
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
            cell_bsr.GetBlockEntry(element, 0, 0));
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
   success =
      Check(
         raw.nnz_raw == num_elements,
         "Element RawCOO allocated neighbor entries.") &&
      success;
   for (GlobalIndex i = 0; i < raw.nnz_raw; ++i)
   {
      success =
         Check(
            raw.rows[i] == i && raw.cols[i] == i,
            "Element RawCOO emitted a cross-element entry.") &&
         success;
      success =
         Check(
            Near(
               raw.values[i],
               element_bsr.GetBlockEntry(i, 0, 0)),
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

   FreeRawCOOTripletBuffer(raw);
   FreeCOOMatrix(coo);
   FreeCOOMatrix(explicit_coo);
   FreeCSRMatrix(csr);
   FreeCSCMatrix(csc);

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
   H1Restriction restriction{restriction_indices, 3};
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
            integration_rule);
   auto full_matrix =
      GenericAssembly<MatrixAssemblyType::SGBSR, KernelPolicy>(
         form,
         context,
         integration_rule);

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

   bool success = true;
   success =
      Check(
         raw.nnz_raw == 2,
         "Boundary-only element RawCOO must activate only boundary elements.") &&
      success;
   success =
      Check(
         raw.rows[0] == 0 &&
         raw.cols[0] == 0 &&
         raw.rows[1] == num_elements - 1 &&
         raw.cols[1] == num_elements - 1,
         "Boundary-only element RawCOO activated the wrong elements.") &&
      success;

   FreeRawCOOTripletBuffer(raw);
   return success;
}

bool TestRectangularElementBSR()
{
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
                  matrix.GetBlockEntry(block, 0, col)),
               "Rectangular element BSR contains a non-finite value.") &&
            success;
      }
   }
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
#ifdef GENDIL_USE_HYPRE
   }
   hypre_MPI_Finalize();
#endif
   return success ? 0 : 1;
}
