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

#if defined(GENDIL_USE_DEVICE)
using GlobalFaceKernelPolicy =
   DeviceKernelConfiguration<ThreadBlockLayout<4>, 1, 2>;
#else
using GlobalFaceKernelPolicy = SerialKernelConfiguration;
#endif

template <typename VectorType>
Real AbsoluteL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");

   Real err_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
   }
   return std::sqrt(err_sq);
}

template <typename VectorType>
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   const Real abs_err = AbsoluteL2Error(a, b);

   Real norm_b_sq = 0.0;
   for (Integer i = 0; i < b.Size(); ++i)
   {
      norm_b_sq += b[i] * b[i];
   }

   const Real norm_b = std::sqrt(norm_b_sq);
   return norm_b == 0.0 ? abs_err : abs_err / norm_b;
}

template <typename VectorType>
void FillPattern(VectorType& x)
{
   x.WriteHostData();
   for (Integer i = 0; i < x.Size(); ++i)
   {
      x[i] = 0.25 + 0.13 * i + 0.07 * ((i % 3) + 1);
   }
}

template<class Mesh, class FiniteElement, class InteriorFaces>
auto MakeInteriorGlobalMixedSpace(
   const Mesh& mesh,
   const FiniteElement& finite_element,
   const InteriorFaces& interior_faces)
{
   auto partition =
      MakePartition(
         MakeCellPart(mesh),
         MakeInteriorFacePart<0, 0>(interior_faces));
   return MakeMixedFiniteElementSpace(
      partition,
      std::tuple{finite_element},
      DGDirectSumNumbering{});
}

template<class Mesh, class FiniteElement, class BoundaryFaces>
auto MakeBoundaryGlobalMixedSpace(
   const Mesh& mesh,
   const FiniteElement& finite_element,
   const BoundaryFaces& boundary_faces)
{
   auto partition =
      MakePartition(
         MakeCellPart(mesh),
         MakeBoundaryFacePart<0>(boundary_faces));
   return MakeMixedFiniteElementSpace(
      partition,
      std::tuple{finite_element},
      DGDirectSumNumbering{});
}

template<class Mesh, class FiniteElement, class InteriorFaces, class BoundaryFaces>
auto MakeInteriorBoundaryGlobalMixedSpace(
   const Mesh& mesh,
   const FiniteElement& finite_element,
   const InteriorFaces& interior_faces,
   const BoundaryFaces& boundary_faces)
{
   auto partition =
      MakePartition(
         MakeCellPart(mesh),
         MakeInteriorFacePart<0, 0>(interior_faces),
         MakeBoundaryFacePart<0>(boundary_faces));
   return MakeMixedFiniteElementSpace(
      partition,
      std::tuple{finite_element},
      DGDirectSumNumbering{});
}

template <typename VectorType>
bool CheckClose(
   const char* label,
   const VectorType& a,
   const VectorType& b,
   const Real tol = 1.0e-11)
{
   const Real abs_err = AbsoluteL2Error(a, b);
   const Real rel_err = RelativeL2Error(a, b);
   std::cout << label
             << " | abs L2 error = " << abs_err
             << ", rel L2 error = " << rel_err << "\n";

   if (rel_err > tol)
   {
      std::cerr << "FAILED: " << label << " exceeded tolerance " << tol << "\n";
      return false;
   }
   return true;
}

template <typename Operator>
Vector ApplyOperator(const Operator& op, const Vector& x)
{
   Vector y(x.Size());
   y = 0.0;
   op(x, y);
   return y;
}

template <typename Operator>
std::vector<Real> BuildDenseMatrix(
   const Operator& op,
   const Integer size)
{
   std::vector<Real> dense(static_cast<size_t>(size * size), 0.0);
   Vector x(size);
   Vector y(size);

   for (Integer col = 0; col < size; ++col)
   {
      // Initialize basis vector entirely on host
      Real* x_data = x.WriteHostData();
      std::fill(x_data, x_data + size, 0.0);
      x_data[col] = 1.0;

      y = 0.0;
      op(x, y);

      for (Integer row = 0; row < size; ++row)
      {
         dense[static_cast<size_t>(row * size + col)] = y[row];
      }
   }

   return dense;
}

void AddOuterProduct(
   std::vector<Real>& ref,
   const std::vector<Real>& row_trace,
   const std::vector<Real>& col_trace,
   const Real scale = 1.0)
{
   GENDIL_VERIFY(
      row_trace.size() == col_trace.size(),
      "Trace vector sizes do not match.");
   const size_t size = row_trace.size();
   GENDIL_VERIFY(ref.size() == size * size, "Dense matrix size mismatch.");

   for (size_t row = 0; row < size; ++row)
   {
      for (size_t col = 0; col < size; ++col)
      {
         ref[row * size + col] +=
            scale * row_trace[row] * col_trace[col];
      }
   }
}

bool CheckDenseClose(
   const char* label,
   const std::vector<Real>& dense,
   const std::vector<Real>& ref,
   const Real tol = 1.0e-11)
{
   GENDIL_VERIFY(dense.size() == ref.size(), "Dense matrix sizes do not match.");

   Real max_err = 0.0;
   for (size_t i = 0; i < dense.size(); ++i)
   {
      max_err = std::max(max_err, std::abs(dense[i] - ref[i]));
   }

   std::cout << label << " | dense max error = " << max_err << "\n";
   if (max_err > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      const size_t size =
         static_cast<size_t>(std::sqrt(static_cast<Real>(dense.size())));
      for (size_t row = 0; row < size; ++row)
      {
         for (size_t col = 0; col < size; ++col)
         {
            const size_t idx = row * size + col;
            if (std::abs(dense[idx] - ref[idx]) > tol)
            {
               std::cerr
                  << "  (" << row << ", " << col << "): got "
                  << dense[idx] << ", expected " << ref[idx] << "\n";
            }
         }
      }
      return false;
   }
   return true;
}

struct FullSharedThreadedFaceKernelPolicy :
   public DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 1>
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
   using face_write_dofs_policy = FullSharedFaceWriteDofsPolicy;
};

bool TestContextStorage()
{
   constexpr Integer Dim = 2;
   constexpr GlobalIndex n = 2;
   const Real h = 1.0 / n;

   CartesianMesh<Dim> mesh({n, n}, {h, h}, {0.0, 0.0}, false);
   FiniteElementOrders<1, 1> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);

   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<Dim>({n, n});
   auto boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<Dim>({n, n});

   auto singleton_global_fes =
      MakeInteriorBoundaryGlobalMixedSpace(
         mesh,
         finite_element,
         interior_faces,
         boundary_faces);

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(singleton_global_fes),
      MakeIntegrationDomain<"mesh">(singleton_global_fes));

   using Ctx = decltype(ctx);
   static_assert(Ctx::template has_domain<"mesh">());
   static_assert(Ctx::template has_interior_face_domain<"mesh">());
   static_assert(Ctx::template has_boundary_face_domain<"mesh">());
   static_assert(Ctx::has_any_interior_face_domain());
   static_assert(Ctx::has_any_boundary_face_domain());
   static_assert(use_global_facets_operator_v<Ctx>);
   static_assert(!Ctx::template has_interior_face_domain<"other">());
   static_assert(!Ctx::template has_boundary_face_domain<"other">());

   (void)ctx.template domain<"mesh">();
   (void)ctx.template interior_face_domain<"mesh">();
   (void)ctx.template boundary_face_domain<"mesh">();

   std::cout << "PASS: weak-form context stores independent face domains\n";
   return true;
}

bool TestInteriorTwoCellSigns()
{
   constexpr Integer order = 2;
   constexpr GlobalIndex n = 2;
   const Real h = 1.0 / n;

   Cartesian1DMesh mesh(h, n);
   FiniteElementOrders<order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

#if defined(GENDIL_USE_DEVICE)
   using LocalKernelPolicy =
      ThreadFirstKernelConfiguration<ThreadBlockLayout<num_quad_1d>, 1>;
#else
   using LocalKernelPolicy = SerialKernelConfiguration;
#endif
   using FESpace = std::remove_cvref_t<decltype(fe_space)>;
   using IntegrationRule = decltype(integration_rule);
   using SmallIntegrationRule =
      decltype(MakeIntegrationRule(IntegrationRuleNumPoints<1>{}));

   static_assert(
      generic_operator_face_read_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         FESpace> ==
      FESpace::finite_element_type::GetNumDofs());
   static_assert(
      generic_operator_face_write_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         FESpace> ==
      FESpace::finite_element_type::GetNumDofs());

   constexpr size_t integrand_scratch =
      generic_operator_integrand_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule>;
   constexpr size_t one_face_read_scratch =
      generic_operator_face_read_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         FESpace>;
   constexpr size_t one_face_write_scratch =
      generic_operator_face_write_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         FESpace>;
   constexpr size_t accurate_global_face_scratch =
      Max(
         integrand_scratch,
         one_face_read_scratch,
         one_face_write_scratch);

   using ScalarJumpForm = decltype(integrate(
      InteriorFacets<"solid">{},
      jump(TrialSpace<"u">{}) * jump(TestSpace<"u">{})));
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace,
         FESpace,
         FESpace,
         FESpace,
         ScalarJumpForm> == accurate_global_face_scratch);
   static_assert(
      global_generic_boundary_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace> == accurate_global_face_scratch);

   constexpr size_t small_integrand_scratch =
      generic_operator_integrand_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule>;
   constexpr size_t one_read_small_rule_formula =
      Max(
         small_integrand_scratch,
         one_face_read_scratch,
         one_face_write_scratch);
   constexpr size_t two_read_small_rule_formula =
      Max(
         small_integrand_scratch,
         2 * one_face_read_scratch,
         one_face_write_scratch);
   static_assert(small_integrand_scratch < one_face_read_scratch);
   static_assert(one_read_small_rule_formula != two_read_small_rule_formula);
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         FESpace,
         FESpace,
         FESpace,
         FESpace,
         ScalarJumpForm> == one_read_small_rule_formula);
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         FESpace,
         FESpace,
         FESpace,
         FESpace,
         ScalarJumpForm> != two_read_small_rule_formula);
   static_assert(
      global_generic_boundary_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         FESpace> == one_read_small_rule_formula);
   static_assert(
      local_generic_cell_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace> ==
      local_generic_interior_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace>);
   static_assert(
      local_generic_cell_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace> ==
      local_generic_boundary_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace>);
   using BatchedKernelPolicy =
      DeviceKernelConfiguration<ThreadBlockLayout<4>, 1, 2>;
   static_assert(
      !is_unbatched_operator_configuration_allowed_v<BatchedKernelPolicy>);

   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(u_h);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"mesh"> interior_facets;

   auto sign_sensitive_form =
      integrate(
         interior_facets,
         jump(u) * jump(v));
   auto canonical_sign_sensitive_form =
      integrate(
         interior_facets,
         jump(u) * jump(v)
         + average(u) * jump(v)
         + average(dot(grad(u), Normal{})) * jump(v));

   auto local_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeIntegrationDomain<"mesh">(fe_space));

   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({n});
   auto singleton_global_fes =
      MakeInteriorGlobalMixedSpace(mesh, finite_element, interior_faces);

   auto global_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(singleton_global_fes),
      MakeIntegrationDomain<"mesh">(singleton_global_fes));

   auto local_op =
      MakeGenericOperator<LocalKernelPolicy>(
         sign_sensitive_form,
         local_ctx,
         integration_rule);
   auto global_op =
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         sign_sensitive_form,
         global_ctx,
         integration_rule);
   auto canonical_global_op =
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         canonical_sign_sensitive_form,
         global_ctx,
         integration_rule);

   const Vector y_local = ApplyOperator(local_op, u_h);
   const Vector y_global = ApplyOperator(global_op, u_h);

   const std::vector<Real> jump_trace{0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
   const std::vector<Real> average_trace{0.0, 0.0, 0.5, 0.5, 0.0, 0.0};
   const std::vector<Real> average_normal_flux_trace{
      -1.0, 4.0, -3.0, 3.0, -4.0, 1.0};
   // Dense reference in the canonical global-face test-row convention. This
   // deliberately keeps the broader expression sign-sensitive instead of
   // falling back to local/current-row parity.
   std::vector<Real> canonical_ref(jump_trace.size() * jump_trace.size(), 0.0);
   AddOuterProduct(canonical_ref, jump_trace, jump_trace);
   AddOuterProduct(canonical_ref, jump_trace, average_trace, -1.0);
   AddOuterProduct(canonical_ref, jump_trace, average_normal_flux_trace, -1.0);

   bool success =
      CheckClose("two-Cell interior local vs global", y_global, y_local);
   success =
      CheckDenseClose(
         "two-Cell canonical global sign-sensitive dense reference",
         BuildDenseMatrix(
            canonical_global_op,
            fe_space.GetNumberOfFiniteElementDofs()),
         canonical_ref) &&
      success;
   return success;
}

bool TestDispatchIndependence()
{
   constexpr Integer Dim = 2;
   constexpr Integer order = 2;
   constexpr GlobalIndex nx = 3;
   constexpr GlobalIndex ny = 2;

   CartesianMesh<Dim> mesh(
      {nx, ny},
      {0.31, 0.47},
      {0.10, -0.20},
      false);

   FiniteElementOrders<order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

#if defined(GENDIL_USE_DEVICE)
   using LocalKernelPolicy =
      ThreadFirstKernelConfiguration<
         ThreadBlockLayout<num_quad_1d, num_quad_1d>,
         Dim>;
#else
   using LocalKernelPolicy = SerialKernelConfiguration;
#endif

   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(u_h);

   Vector mu_h(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(mu_h);
   auto local_mu_view =
      MakeReadOnlyElementTensorView<LocalKernelPolicy>(fe_space, mu_h);
   auto global_mu_view =
      MakeReadOnlyElementTensorView<GlobalFaceKernelPolicy>(fe_space, mu_h);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;
   BoundaryFacets<"mesh"> boundary_facets;

   auto mu = MakeCoefficient<"mu", FieldValue<"mu">>(
      [] GENDIL_HOST_DEVICE (const Real mu_q) -> Real
      {
         return mu_q;
      });

   auto cell_form =
      integrate(cells, u * v);

   auto interior_form =
      integrate(cells, u * v)
      + integrate(
         interior_facets,
         average(mu) * jump(u) * jump(v)
         + average(mu * dot(grad(u), Normal{})) * jump(v));

   auto boundary_form =
      integrate(cells, u * v)
      + integrate(
         boundary_facets,
         mu * u * v
         + mu * dot(grad(u), Normal{}) * v);

   auto form =
      integrate(cells, u * v)
      + integrate(
         interior_facets,
         average(mu) * jump(u) * jump(v)
         + average(mu * dot(grad(u), Normal{})) * jump(v))
      + integrate(
         boundary_facets,
         mu * u * v
         + mu * dot(grad(u), Normal{}) * v);

   static_assert(requires_plus_side_jacobian_v<decltype(form)>);

   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<Dim>({nx, ny});
   auto boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<Dim>({nx, ny});

   auto interior_global_fes =
      MakeInteriorGlobalMixedSpace(mesh, finite_element, interior_faces);
   auto boundary_global_fes =
      MakeBoundaryGlobalMixedSpace(mesh, finite_element, boundary_faces);
   auto both_global_fes =
      MakeInteriorBoundaryGlobalMixedSpace(
         mesh,
         finite_element,
         interior_faces,
         boundary_faces);

   auto local_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, local_mu_view),
      MakeIntegrationDomain<"mesh">(fe_space));

   auto interior_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(interior_global_fes),
      MakeFiniteElementField<"mu">(interior_global_fes, global_mu_view),
      MakeIntegrationDomain<"mesh">(interior_global_fes));

   auto boundary_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(boundary_global_fes),
      MakeFiniteElementField<"mu">(boundary_global_fes, global_mu_view),
      MakeIntegrationDomain<"mesh">(boundary_global_fes));

   auto both_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(both_global_fes),
      MakeFiniteElementField<"mu">(both_global_fes, global_mu_view),
      MakeIntegrationDomain<"mesh">(both_global_fes));

   static_assert(!use_global_facets_operator_v<decltype(local_ctx)>);
   static_assert(use_global_facets_operator_v<decltype(interior_ctx)>);
   static_assert(use_global_facets_operator_v<decltype(boundary_ctx)>);
   static_assert(use_global_facets_operator_v<decltype(both_ctx)>);

   static_assert(
      global_facet_domain_requirements_satisfied_v<
         decltype(interior_form),
         decltype(interior_ctx)>);
   static_assert(
      global_facet_domain_requirements_satisfied_v<
         decltype(boundary_form),
         decltype(boundary_ctx)>);
   static_assert(
      global_facet_domain_requirements_satisfied_v<
         decltype(form),
         decltype(both_ctx)>);
   static_assert(
      !global_facet_domain_requirements_satisfied_v<
         decltype(form),
         decltype(interior_ctx)>);
   static_assert(
      !global_facet_domain_requirements_satisfied_v<
         decltype(form),
         decltype(boundary_ctx)>);

   InteriorFacets<"mesh_a"> interior_facets_a;
   InteriorFacets<"mesh_b"> interior_facets_b;
   auto two_name_interior_form =
      integrate(interior_facets_a, jump(u) * jump(v))
      + integrate(interior_facets_b, jump(u) * jump(v));
   auto mesh_a_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(interior_global_fes),
      MakeFiniteElementField<"mu">(interior_global_fes, global_mu_view),
      MakeIntegrationDomain<"mesh_a">(interior_global_fes));
   static_assert(
      !global_facet_domain_requirements_satisfied_v<
         decltype(two_name_interior_form),
         decltype(mesh_a_ctx)>);

   const Vector y_local = ApplyOperator(
      MakeGenericOperator<LocalKernelPolicy>(form, local_ctx, integration_rule),
      u_h);
   const Vector y_interior = ApplyOperator(
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         interior_form,
         interior_ctx,
         integration_rule),
      u_h);
   const Vector y_interior_ref = ApplyOperator(
      MakeGenericOperator<LocalKernelPolicy>(
         interior_form,
         local_ctx,
         integration_rule),
      u_h);
   const Vector y_boundary = ApplyOperator(
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         boundary_form,
         boundary_ctx,
         integration_rule),
      u_h);
   const Vector y_boundary_ref = ApplyOperator(
      MakeGenericOperator<LocalKernelPolicy>(
         boundary_form,
         local_ctx,
         integration_rule),
      u_h);
   const Vector y_both = ApplyOperator(
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         form,
         both_ctx,
         integration_rule),
      u_h);
   const Vector y_cell = ApplyOperator(
      MakeGenericOperator<GlobalFaceKernelPolicy>(
         cell_form,
         interior_ctx,
         integration_rule),
      u_h);
   const Vector y_cell_ref = ApplyOperator(
      MakeGenericOperator<LocalKernelPolicy>(
         cell_form,
         local_ctx,
         integration_rule),
      u_h);

   bool success = true;
   success = CheckClose("interior-only form with interior domain", y_interior, y_interior_ref) && success;
   success = CheckClose("boundary-only form with boundary domain", y_boundary, y_boundary_ref) && success;
   success = CheckClose("interior+boundary-domain dispatch", y_both, y_local) && success;
   success = CheckClose("cell-only form with global face domain", y_cell, y_cell_ref) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestContextStorage() && success;
   success = TestInteriorTwoCellSigns() && success;
   success = TestDispatchIndependence() && success;

   if (!success)
   {
      return 1;
   }

   std::cout << "\nAll generic global face domain tests passed.\n";
   return 0;
}
