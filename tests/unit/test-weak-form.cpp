// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

using namespace std;
using namespace gendil;

int main()
{
   // Creating trial and test spaces
   using T = TrialSpace< "trial" >;
   static_assert(T::name.view() == "trial");
   TrialSpace<"temp"> u;
   TestSpace<"temp"> v;
   // Simple coefficient and expression tests
   auto c = MakeCoefficient<"diffusivity", PhysicalCoordinate, Jacobian, FieldValue<"displacement">>(
      []( const auto& x_phys, const auto& J, const auto& u ) {
         return 1.0 + x_phys[0]*x_phys[0] + J[0][0] + u[0];
      }
   );
   auto product = u * v;
   std::cout << product << std::endl;
   std::cout << c << std::endl;
   // auto gradient = c * grad(product);
   auto gradient = c * grad(v);
   std::cout << gradient << std::endl;
   // auto fields_names = GetFieldsNames(gradient);
   // std::cout << fields_names << std::endl;
   // Creating weak forms
   Cells<"mesh1"> domain;
   auto vol_form = integrate(domain, gradient);
   std::cout << vol_form << std::endl;
   auto vol_ops = GetTrialOperators( vol_form );
   std::cout << vol_ops << std::endl;
   InteriorFacets<"mesh1"> facet_domain;
   auto jump_term = jump(u) * average(dot(grad(v), Normal{}));
   auto surf_form = integrate(facet_domain, jump_term);
   std::cout << surf_form << std::endl;
   auto total_form = vol_form + surf_form;
   std::cout << total_form << std::endl;
   // Testing static map
   constexpr auto m = std::tuple{
      Entry<StringKey<"width">, int>{1920},
      Entry<StringKey<"title">, std::string_view>{"demo"}
   };

   static_assert(Get<"width">(m) == 1920);
   constexpr auto keys = std::tuple{
      StaticString{"width"},
      StaticString{"height"},
      StaticString{"depth"}
   };
   std::cout << keys << "\n";
   // Testing weak form context
   const Integer n = 10;
   const Real h = 1.0/n;
   Cartesian1DMesh mesh(h,n);
   constexpr Integer order = 2;
   FiniteElementOrders<order> orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );
   Vector u_h(fe_space.GetNumberOfFiniteElementDofs());
   u_h = 1.0;
   FillRandom( u_h );
   Vector v_h(fe_space.GetNumberOfFiniteElementDofs());
   v_h = 0.0;
   auto weak_form_context = MakeWeakFormContext(
      MakeFiniteElementField<"displacement">(fe_space, u_h),
      MakeDomain<"mesh1">(mesh)
   );

   auto& u_f = weak_form_context.fe_field<"displacement">();
   auto& domain_f = weak_form_context.domain<"mesh1">();
   std::cout << "u_f size: " << u_f.dofs.Size() << ", domain size: " << domain_f.GetNumberOfCells() << "\n";

   ////////////////
   // Mass Operator
   // Testing operator application
   
   TrialSpace<"displacement"> u_mass;
   TestSpace<"displacement"> v_mass;
   // InteriorFacets<"mesh1"> domain_interior_facets;
   auto rho = MakeCoefficient<"density", PhysicalCoordinate>(
      []( const auto& x_phys ) {
         return 1.0 + x_phys[0]*x_phys[0];
      }
   );

   auto mass_weak_form = integrate( domain, rho * u_mass * v_mass );
   // auto mass_weak_form = integrate( domain, dot( grad(u_mass), grad(v_mass) ) ) + integrate( domain_interior_facets, jump(u_mass) * v_mass );//Add( integrate( domain, u_mass * v_mass ) , integrate( domain, u_mass * v_mass ));//integrate( domain, dot( grad(u_mass), grad(v_mass) ) );
   
   std::cout << "\nMass weak form: " << mass_weak_form << "\n";
   
   auto mass_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeDomain<"mesh1">(mesh)
   );
   
   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints<num_quad_1d> nq;
   auto integration_rule = MakeIntegrationRule(nq);
   
   using KernelPolicy = SerialKernelConfiguration;
   
   auto mass_op = MakeGenericOperator<KernelPolicy>( mass_weak_form, mass_wf_context, integration_rule );
   
   mass_op(u_h, v_h);

   // Testing assembly
   auto mass_bsr_matrix = GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      mass_weak_form,
      mass_wf_context,
      integration_rule
   );
   // Comparing matrix-free application to assembled application
   Vector v_h_matrix(fe_space.GetNumberOfFiniteElementDofs());
   v_h_matrix = 0.0;

   mass_bsr_matrix(u_h, v_h_matrix);

   std::cout << "v_h after matrix-free application: " << v_h << "\n";
   std::cout << "v_h after assembled application: " << v_h_matrix << "\n";

   v_h -= v_h_matrix;
   Real error = 0.0;
   for (size_t i = 0; i < v_h.Size(); ++i)   {
      error += std::abs(v_h[i]);
   }
   std::cout << "Error between matrix-free and assembled application: " << error << "\n";

   /////////////////////
   // Advection Operator
   TrialSpace<"displacement"> u_adv;
   TestSpace<"displacement"> v_adv;
   Cells<"mesh1"> cells;
   InteriorFacets<"mesh1"> interior_facets;

   constexpr Integer Dim = 1;
   auto beta_fn = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X)
      -> std::array<Real, Dim>
   {
      const Real x = X[0];
      return { x * (1.0 - x) };
   };
   auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(beta_fn);

   auto advection_dg_wf =
      integrate(cells, -u_adv * dot(beta, grad(v_adv)))
      + integrate(interior_facets, upwind(beta, u_adv) * v_adv);
   
      std::cout << "\nAdvection weak form: " << advection_dg_wf << "\n";

   auto advection_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeDomain<"mesh1">(mesh)
   );
   
   auto advection_op = MakeGenericOperator<KernelPolicy>( advection_dg_wf, advection_wf_context, integration_rule );

   advection_op(u_h, v_h);

   auto advection_matrix = GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      advection_dg_wf,
      advection_wf_context,
      integration_rule
   );

   advection_matrix(u_h, v_h_matrix);

   std::cout << "v_h after matrix-free advection application: " << v_h << "\n";
   std::cout << "v_h after assembled advection application: " << v_h_matrix << "\n";

   v_h -= v_h_matrix;
   error = 0.0;
   for (size_t i = 0; i < v_h.Size(); ++i)   {
      error += std::abs(v_h[i]);
   }
   std::cout << "Error between matrix-free and assembled advection application: " << error << "\n";

   return 0;
}
