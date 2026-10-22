// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace validation_test
{
// This vector participates in the public access protocol but cannot expose a
// writable buffer. Every invalid invocation must fail before reaching it.
struct Output
{
   size_t extent;
};
size_t GetVectorSize(const Output& output) { return output.extent; }
[[noreturn]] gendil::Real* MutateOutput()
{
   std::fputs("ERROR: output storage was acquired before validation.\n", stderr);
   std::abort();
}
gendil::Real* WriteHostVector(Output&) { return MutateOutput(); }
gendil::Real* ReadWriteHostVector(Output&) { return MutateOutput(); }
} // namespace validation_test

int main(int argc, char** argv)
{
   using namespace gendil;
   if (argc != 2) { return 2; }
   const std::string_view mode(argv[1]);
   Cartesian1DMesh mesh(Real(0.5), 2);
   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto trial = MakeFiniteElementSpace(mesh, fe, ContiguousL2RestrictionSpecification{2, 8});
   auto test = MakeFiniteElementSpace(mesh, fe, ContiguousL2RestrictionSpecification{7, 14});
   auto context = MakeWeakFormContext(MakeTrialField<"u">(trial),
      MakeTestField<"v">(test), MakeIntegrationDomain<"mesh">(mesh));
   const TrialSpace<"u"> u;
   const TestSpace<"v"> v;
#ifdef GENDIL_TEST_BOUNDARY_INVOCATION
   const BoundaryFacets<"mesh"> cells;
#else
   const Cells<"mesh"> cells;
#endif
   auto form = integrate(cells, u * v);
   auto load_form = integrate(cells, v);
   auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3>{});
   auto op = MakeWeakFormResidualOperator<SerialKernelConfiguration>(form, context, rule);
   auto load = MakeWeakFormResidualOperator<SerialKernelConfiguration>(load_form, context, rule);
   if (op.InputSize() != 8 || op.OutputSize() != 14 ||
       load.InputSize() != 0 || load.OutputSize() != 14) { return 3; }

   if (mode == "domain-topology")
   {
      Cartesian1DMesh other_mesh(Real(0.5), 3);
      auto incompatible = MakeWeakFormContext(MakeTrialField<"u">(trial),
         MakeTestField<"v">(test), MakeIntegrationDomain<"mesh">(other_mesh));
      Vector input(8); input = Real(1);
      validation_test::Output output{14};
      GenericExplicitOperator<SerialKernelConfiguration>(form, incompatible, rule, input, output);
      return 0;
   }

   const bool bad_input = mode.starts_with("input-") || mode == "direct-input";
   Vector input(bad_input ? 7 : 8); input = Real(1);
   validation_test::Output output{bad_input ? size_t(14) : size_t(13)};
   if (mode == "input-replace" || mode == "output-replace") { op(input, output); }
   else if (mode == "input-add" || mode == "output-add") { op.ApplyAdd(input, output); }
   else if (mode == "load-replace") { load(output); }
   else if (mode == "load-add") { load.ApplyAdd(output); }
   else if (mode == "direct-input" || mode == "direct-output")
   { GenericExplicitOperator<SerialKernelConfiguration>(form, context, rule, input, output); }
   else if (mode == "direct-load")
   { GenericExplicitOperator<SerialKernelConfiguration>(load_form, context, rule, output); }
   else { return 2; }
   return 0;
}
