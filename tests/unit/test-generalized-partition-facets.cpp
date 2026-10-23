// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include "generalized-interior-test-utils.hpp"

namespace
{
using namespace gendil;
using boundary_test::Check;
using boundary_test::CheckLoad;

template<class Mesh, class FiniteElement>
auto RestrictionAt(const Mesh& mesh, const FiniteElement& fe, GlobalIndex shift, GlobalIndex extent)
{
   return GetRestriction(MakeFiniteElementSpace(mesh,fe,
      ContiguousL2RestrictionSpecification{shift,extent}));
}

template<class Policy, bool OwnerStorage>
bool CheckCanonicalInterface()
{
   constexpr size_t input_extent = 28, output_extent = 24;
   CartesianMesh<1> left({1},{.4},{0.},false);
   CartesianMesh<1> right({1},{.6},{.4},false);
   auto partition = MakePartition(MakeCellPart(left),MakeCellPart(right),
      MakeInteriorFacePart<0,1>(CartesianIntermeshFaceConnectivity<1,1>({1},{1})));
   const auto p0 = MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   const auto u_space = MakeMixedFiniteElementSpace(partition,std::tuple{p1,p2},
      std::tuple{RestrictionAt(left,p1,1,input_extent),RestrictionAt(right,p2,6,input_extent)});
   const auto p_space = MakeMixedFiniteElementSpace(partition,std::tuple{p2,p1},
      std::tuple{RestrictionAt(left,p2,14,input_extent),RestrictionAt(right,p1,20,input_extent)});
   const auto v_space = MakeMixedFiniteElementSpace(partition,std::tuple{p1,p2},
      std::tuple{RestrictionAt(left,p1,2,output_extent),RestrictionAt(right,p2,8,output_extent)});
   const auto q_space = MakeMixedFiniteElementSpace(partition,std::tuple{p0,p1},
      std::tuple{RestrictionAt(left,p0,15,output_extent),RestrictionAt(right,p1,18,output_extent)});
   const auto f_space = MakeMixedFiniteElementSpace(partition,std::tuple{p1,p2},DGDirectSumNumbering{});
   Vector bound(5); auto* data = bound.WriteHostData();
   data[0] = 3; data[1] = 3.4;
   data[2] = -1.2; data[3] = -.6; data[4] = 0;
   auto field = [&]
   {
      if constexpr (OwnerStorage) { return MakeFiniteElementField<"f">(f_space,bound); }
      else { return MakeFiniteElementField<"f">(f_space,
         ReadKernelVector<Policy::is_device_configuration>(bound).data); }
   }();
   auto context = MakeWeakFormContext(MakeTrialField<"u">(u_space),MakeTrialField<"p">(p_space),
      MakeTestField<"v">(v_space),MakeTestField<"q">(q_space),field,
      MakeIntegrationDomain<"mesh">(partition));
   const InteriorFacets<"mesh"> interior;
   const TrialSpace<"u"> u;
   const TrialSpace<"p"> p;
   const TestSpace<"v"> v;
   const TestSpace<"q"> q;
   const FiniteElementField<"f"> f;
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4>{});
   std::array<Real,output_extent> minus_value{},plus_value{},jump_value{},gradient{},bound_value{};
   minus_value[3] = 3;
   plus_value[8] = 4;
   jump_value[3] = 2; jump_value[8] = -2;
   // P2 endpoint derivatives at xi=0 are {-3,4,-1}, divided by the
   // plus cell's length .6. This probes geometry required only by output.
   gradient[8] = -5; gradient[9] = Real(20)/3; gradient[10] = -Real(5)/3;
   bound_value[3] = 1.1; bound_value[8] = -1.1;
   auto unit = MakeVectorCoefficient<"unit">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real,1>{1}; });
   bool ok = CheckLoad(MakeWeakFormResidualOperator<Policy>(integrate(interior,3.0*minus(v)),context,rule),
      minus_value,"canonical minus-only load");
   ok = CheckLoad(MakeWeakFormResidualOperator<Policy>(integrate(interior,4.0*plus(v)),context,rule),
      plus_value,"canonical plus-only load") && ok;
   ok = CheckLoad(MakeWeakFormResidualOperator<Policy>(integrate(interior,2.0*jump(v)),context,rule),
      jump_value,"canonical signed two-row load") && ok;
   ok = CheckLoad(MakeWeakFormResidualOperator<Policy>(integrate(interior,dot(unit,plus(grad(v)))),context,rule),
      gradient,"canonical plus-only output-gradient geometry") && ok;
   ok = CheckLoad(MakeWeakFormResidualOperator<Policy>(integrate(interior,average(f)*jump(v)),context,rule),
      bound_value,"canonical bound-owner/view load") && ok;

   auto read_p = MakeCoefficient<"read_p",FieldValue<"p">>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   auto form = integrate(interior,minus(read_p)*plus(v))
      + integrate(interior,jump(u)*average(p)*jump(v))
      + integrate(interior,jump(p)*jump(q))
      + integrate(interior,average(f)*jump(v));
   auto op = MakeWeakFormResidualOperator<Policy>(form,context,rule);
   Vector x(input_extent),y(output_extent+3); x = 99;
   auto* values = x.ReadWriteHostData();
   values[1] = 1; values[2] = 1.8;
   values[6] = 2.8; values[7] = 1.9; values[8] = 1;
   values[14] = 2; values[15] = 2.04; values[16] = 2.16;
   values[20] = .6; values[21] = 3;
   std::array<Real,output_extent> expected{};
   const Real jump_u = 1.8-2.8, mean_p = (2.16+.6)/2;
   const Real flux = jump_u*mean_p + (3.4-1.2)/2;
   expected[3] = flux;
   expected[8] = -flux+2.16;
   expected[15] = 2.16-.6;
   expected[18] = -(2.16-.6);
   ok = op.InputSize() == input_extent && op.OutputSize() == output_extent && ok;
   observed_facet_vectors::VectorRef observed_x{x}, observed_y{y};
   y = 9; op(observed_x,observed_y);
   ok = observed_x.reads == 1 && observed_x.writes == 0 && observed_x.read_writes == 0 &&
      observed_y.reads == 0 && observed_y.writes == 1 && observed_y.read_writes == 0 && ok;
   ok = Check(y,expected,"canonical heterogeneous nonlinear residual") && ok;
   y = -3; op.ApplyAdd(x,y); ok = Check(y,expected,"canonical additive residual",1,-3) && ok;
   op.ApplyAdd(x,y); ok = Check(y,expected,"canonical repeated additive residual",2,-3) && ok;

   // Assembly probes reuse the exact persistent source binding as application.
   // The independent matrix reference has four nonzero entries: the endpoint
   // jump outer product scaled by (f_minus + f_plus)/2 = 1.1.
   auto linear_form = integrate(interior,average(f)*jump(u)*jump(v));
   auto linear = MakeWeakFormResidualOperator<Policy>(linear_form,context,rule);
   std::array<Real,output_extent> linear_expected{};
   linear_expected[3] = Real(1.1) * (values[2]-values[6]);
   linear_expected[8] = -linear_expected[3];
   y = 0; linear(x,y);
   ok = Check(y,linear_expected,"canonical bound-owner/view linear application") && ok;
   auto raw = GenericAssembly<MatrixAssemblyType::RawCOO,Policy>(linear_form,context,rule);
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   Vector assembled(output_extent+3); assembled = 0;
   auto* result = assembled.ReadWriteHostData();
   const auto* input = x.ReadHostData();
   const auto* rows = ReadHost(raw.rows);
   const auto* columns = ReadHost(raw.cols);
   const auto* entries = ReadHost(raw.values);
   ok = raw.num_rows == output_extent && raw.num_cols == input_extent && ok;
   for (GlobalIndex i = 0; i < raw.nnz_raw; ++i)
   {
      if (rows[i] >= output_extent || columns[i] >= input_extent) { return false; }
      result[rows[i]] += entries[i]*input[columns[i]];
   }
   ok = Check(assembled,linear_expected,"canonical bound-owner/view rectangular RawCOO") && ok;
   return ok;
}

template<class Policy>
bool CheckPartitionBoundaryLoads()
{
   // Two physical endpoints and an empty same-part interior batch per part.
   auto partition = MakeCartesian1DPartition(.5,std::array<GlobalIndex,2>{1,1});
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   const auto space = MakeMixedFiniteElementSpace(partition,std::tuple{p1,p2},DGDirectSumNumbering{});
   auto context = MakeWeakFormContext(MakeTestField<"v">(space),MakeIntegrationDomain<"mesh">(partition));
   auto op = MakeWeakFormResidualOperator<Policy>(
      integrate(BoundaryFacets<"mesh">{},2.0*jump(average(TestSpace<"v">{}))),context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4>{}));
   return CheckLoad(op,std::array<Real,5>{2,0,0,0,2},"heterogeneous partition boundary load");
}
}
int main()
{
   bool ok = CheckCanonicalInterface<SerialKernelConfiguration,false>() &&
      CheckCanonicalInterface<SerialKernelConfiguration,true>() &&
      CheckPartitionBoundaryLoads<SerialKernelConfiguration>();
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<4>,1,4>;
   ok = CheckCanonicalInterface<Policy,false>() && CheckCanonicalInterface<Policy,true>() &&
      CheckPartitionBoundaryLoads<Policy>() && ok;
#endif
   return ok ? 0 : 1;
}
