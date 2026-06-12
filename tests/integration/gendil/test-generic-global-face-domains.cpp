// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

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
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   auto interior_faces =
      make_cartesian_interior_face_connectivity<Dim>({n, n});
   auto boundary_faces =
      make_cartesian_boundary_face_connectivity<Dim>({n, n});

   auto interior_face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(fe_space, interior_faces);
   auto boundary_face_fes =
      MakeGlobalBoundaryFaceFiniteElementSpace(fe_space, boundary_faces);

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeDomain<"mesh">(mesh),
      MakeInteriorFaceDomain<"mesh">(interior_face_fes),
      MakeBoundaryFaceDomain<"mesh">(boundary_face_fes));

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

   using KernelPolicy = SerialKernelConfiguration;
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

   static_assert(
      global_generic_interior_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         IntegrationRule,
         FESpace> == accurate_global_face_scratch);
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
      global_generic_interior_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         FESpace> == one_read_small_rule_formula);
   static_assert(
      global_generic_interior_facet_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         FESpace> != two_read_small_rule_formula);
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
      DeviceKernelConfiguration<ThreadBlockLayout<>, 0, 2>;
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
         jump(u) * jump(v)
         + average(u) * jump(v)
         + average(dot(grad(u), Normal{})) * jump(v));

   auto local_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeDomain<"mesh">(mesh));

   auto interior_faces =
      make_cartesian_interior_face_connectivity<1>({n});
   auto interior_face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(fe_space, interior_faces);

   auto global_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeDomain<"mesh">(mesh),
      MakeInteriorFaceDomain<"mesh">(interior_face_fes));

   auto local_op =
      MakeGenericOperator<KernelPolicy>(
         sign_sensitive_form,
         local_ctx,
         integration_rule);
   auto global_op =
      MakeGenericOperator<KernelPolicy>(
         sign_sensitive_form,
         global_ctx,
         integration_rule);

   const Vector y_local = ApplyOperator(local_op, u_h);
   const Vector y_global = ApplyOperator(global_op, u_h);

   return CheckClose("two-Cell interior local vs global", y_global, y_local);
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

   using KernelPolicy = SerialKernelConfiguration;

   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(u_h);

   Vector mu_h(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(mu_h);
   auto mu_view = MakeReadOnlyElementTensorView<KernelPolicy>(fe_space, mu_h);

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
      make_cartesian_interior_face_connectivity<Dim>({nx, ny});
   auto boundary_faces =
      make_cartesian_boundary_face_connectivity<Dim>({nx, ny});

   auto interior_face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(fe_space, interior_faces);
   auto boundary_face_fes =
      MakeGlobalBoundaryFaceFiniteElementSpace(fe_space, boundary_faces);

   auto local_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, mu_view),
      MakeDomain<"mesh">(mesh));

   auto interior_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, mu_view),
      MakeDomain<"mesh">(mesh),
      MakeInteriorFaceDomain<"mesh">(interior_face_fes));

   auto boundary_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, mu_view),
      MakeDomain<"mesh">(mesh),
      MakeBoundaryFaceDomain<"mesh">(boundary_face_fes));

   auto both_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, mu_view),
      MakeDomain<"mesh">(mesh),
      MakeInteriorFaceDomain<"mesh">(interior_face_fes),
      MakeBoundaryFaceDomain<"mesh">(boundary_face_fes));

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
      MakeTrialField<"u">(fe_space),
      MakeFiniteElementField<"mu">(fe_space, mu_view),
      MakeDomain<"mesh">(mesh),
      MakeInteriorFaceDomain<"mesh_a">(interior_face_fes));
   static_assert(
      !global_facet_domain_requirements_satisfied_v<
         decltype(two_name_interior_form),
         decltype(mesh_a_ctx)>);

   const Vector y_local = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(form, local_ctx, integration_rule),
      u_h);
   const Vector y_interior = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(interior_form, interior_ctx, integration_rule),
      u_h);
   const Vector y_interior_ref = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(interior_form, local_ctx, integration_rule),
      u_h);
   const Vector y_boundary = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(boundary_form, boundary_ctx, integration_rule),
      u_h);
   const Vector y_boundary_ref = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(boundary_form, local_ctx, integration_rule),
      u_h);
   const Vector y_both = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(form, both_ctx, integration_rule),
      u_h);
   const Vector y_cell = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(cell_form, interior_ctx, integration_rule),
      u_h);
   const Vector y_cell_ref = ApplyOperator(
      MakeGenericOperator<KernelPolicy>(cell_form, local_ctx, integration_rule),
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
