// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

int main()
{
   using namespace gendil;

   Cartesian1DMesh mesh(0.5, 2);
   auto trial_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<1>{});
   auto test_fe =
      MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   auto trial_space = MakeFiniteElementSpace(mesh, trial_fe);
   auto test_space = MakeFiniteElementSpace(mesh, test_fe);
   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form = integrate(Cells<"mesh">{}, u * v);
   auto context =
      MakeWeakFormContext(
         MakeTrialField<"u">(trial_space),
         MakeTestField<"v">(test_space),
         MakeIntegrationDomain<"mesh">(mesh));
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<2>{});

#if defined(GENDIL_TEST_ELEMENT_SGBSR)
   auto matrix =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         SerialKernelConfiguration>(
            form,
            context,
            integration_rule);
#else
   auto matrix =
      GenericAssembly<
         MatrixAssemblyType::SGBSR,
         SerialKernelConfiguration>(
            form,
            context,
            integration_rule);
#endif
   (void)matrix;
   return 0;
}
