// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include "generalized-boundary-test-utils.hpp"

namespace
{
using namespace gendil;

template<class Policy, bool Owner>
bool CheckSelectedSideTransport()
{
   CartesianMesh<1> left({1},{.4},{0.},false), right({1},{.6},{.4},false);
   auto partition = MakePartition(MakeCellPart(left),MakeCellPart(right),
      MakeInteriorFacePart<0,1>(CartesianIntermeshFaceConnectivity<1,1>({1},{1})));
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto space = MakeMixedFiniteElementSpace(partition,std::tuple{p1,p1},DGDirectSumNumbering{});
   Vector bound(4); auto* field = bound.WriteHostData();
   field[0]=1; field[1]=1.8; field[2]=3; field[3]=4.8;
   auto registration = [&] {
      if constexpr (Owner) return MakeFiniteElementField<"f">(space,bound);
      else return MakeFiniteElementField<"f">(space,
         ReadKernelVector<Policy::is_device_configuration>(bound).data);
   }();
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space),MakeTestField<"v">(space),
      registration,MakeIntegrationDomain<"mesh">(partition));
   const TrialSpace<"u"> u;
   const TestSpace<"v"> v;
   const InteriorFacets<"mesh"> interior;
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4>{});
   Vector x(4),y(7); auto* input=x.WriteHostData();
   input[0]=1; input[1]=2; input[2]=4; input[3]=7;
   auto inverse = MakeCoefficient<"inverse",JacobianInverse>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 1>& invJ) { return 1+invJ[0]; });
   auto bound_derivative = MakeCoefficient<"bound_derivative",FieldGradient<"f">>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 1>& derivative) { return derivative[0]; });
   auto active_derivative = MakeCoefficient<"active_derivative",FieldGradient<"u">>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 1>& derivative) { return derivative[0]; });
   bool ok=true;
   for (const Real speed : {Real(2),Real(-2)})
   {
      auto beta = MakeVectorCoefficient<"beta">(
         [speed] GENDIL_HOST_DEVICE () { return std::array<Real,1>{speed}; });
      auto check = [&](const auto& transported, Real minus_factor, Real plus_factor, const char* label) {
         auto form=integrate(interior,upwind(beta,transported)*jump(v));
         auto op=MakeWeakFormResidualOperator<Policy>(form,context,rule);
         // The only face joins xi=1 of the .4 cell to xi=0 of the .6 cell.
         // Exact P1 traces are 2 and 4; the physical derivatives are 2/3
         // for f and 2.5/5 for u. The chosen side supplies its own inverse J.
         const Real flux=speed*(speed>=0 ? 2*minus_factor : 4*plus_factor);
         const std::array<Real,4> expected{0,flux,-flux,0};
         y=9; op(x,y);
         bool pass=boundary_test::Check(y,expected,label);
         y=-2; op.ApplyAdd(x,y);
         return boundary_test::Check(y,expected,label,1,-2) && pass;
      };
      ok=check(u*inverse,1+1/.4,1+1/.6,"upwind selected inverse Jacobian") && ok;
      ok=check(u*bound_derivative,2,3,"upwind selected bound gradient") && ok;
      // A gradient callback reading the same active name remains valid; only
      // an explicitly transported trial gradient retains its legacy rejection.
      ok=check(u*active_derivative,2.5,5,"upwind selected active gradient callback") && ok;
   }
   return ok;
}
}

int main()
{
   bool ok=CheckSelectedSideTransport<gendil::SerialKernelConfiguration,false>() &&
      CheckSelectedSideTransport<gendil::SerialKernelConfiguration,true>();
#ifdef GENDIL_USE_DEVICE
   using Policy=gendil::DeviceKernelConfiguration<gendil::ThreadBlockLayout<4>,1,4>;
   ok=CheckSelectedSideTransport<Policy,false>() && CheckSelectedSideTransport<Policy,true>() && ok;
#endif
   return ok ? 0 : 1;
}
