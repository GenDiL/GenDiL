// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include "generalized-boundary-test-utils.hpp"

namespace
{
using namespace gendil;
using namespace boundary_test;

struct BoundaryPoint
{
   static constexpr bool is_physical_boundary = true;
   std::array<Real, 2> physical_normal{1, 0};
};

bool CheckSelectedStateEvaluation()
{
   Integer velocity_calls = 0, interior_calls = 0, exterior_calls = 0;
   Real speed = 2;
   const auto beta = MakeVectorCoefficient<"beta">([&]
   {
      ++velocity_calls;
      return std::array<Real, 2>{speed, 0};
   });
   const auto interior = MakeCoefficient<"inside">([&]
   {
      ++interior_calls;
      return Real(3);
   });
   const auto exterior = MakeCoefficient<"outside">([&]
   {
      ++exterior_calls;
      return Real(5);
   });
   const auto flux = upwind(beta, interior, exterior);
   const auto evaluate = [] (const auto & expression)
   {
      return expression(Empty{}, Empty{}, Empty{}, Empty{}, BoundaryPoint{}, Empty{});
   };
   bool ok = evaluate(flux) == 6 && velocity_calls == 1 &&
      interior_calls == 1 && exterior_calls == 0;
   speed = -2;
   ok = evaluate(flux) == -10 && velocity_calls == 2 &&
      interior_calls == 1 && exterior_calls == 1 && ok;
   speed = 0;
   ok = evaluate(flux) == 0 && velocity_calls == 3 &&
      interior_calls == 1 && exterior_calls == 1 && ok;

   // Compatible vector states can use distinct fixed-size container types.
   const auto array_state = MakeVectorCoefficient<"array">(
      [] { return std::array<float, 2>{1, 2}; });
   const auto tensor_state = MakeVectorCoefficient<"tensor">([]
   {
      SerialRecursiveArray<Real, 2> value{};
      value(0) = 3;
      value(1) = 4;
      return value;
   });
   const auto vector_flux = upwind(beta, array_state, tensor_state);
   speed = 2;
   const auto positive = evaluate(vector_flux);
   speed = -2;
   const auto negative = evaluate(vector_flux);
   speed = 0;
   const auto zero = evaluate(vector_flux);
   ok = positive(0) == 2 && positive(1) == 4 &&
      negative(0) == -6 && negative(1) == -8 &&
      zero(0) == 0 && zero(1) == 0 && ok;
   if (!ok) { std::cerr << "boundary upwind selected-state evaluation failed\n"; }
   return ok;
}

template<class Policy>
bool CheckScalarFluxAndAffineSplit()
{
   constexpr Real hx = .5, hy = .75, ox = .25, oy = -.5;
   CartesianMesh<2> mesh({2, 1}, {hx, hy}, {ox, oy}, false);
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const BoundaryFacets<"mesh"> boundary;
   const TrialSpace<"u"> u;
   const TestSpace<"v"> v;
   const FiniteElementField<"g"> g;
   const auto beta = MakeVectorCoefficient<"beta">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{2, -.5}; });
   const auto constant = MakeCoefficient<"constant">(
      [] GENDIL_HOST_DEVICE () { return Real(3); });
   Vector x(8), inflow(8);
   auto* input = x.WriteHostData();
   auto* data = inflow.WriteHostData();
   for (size_t cell = 0; cell < 2; ++cell)
   for (size_t j = 0; j < 2; ++j)
   for (size_t i = 0; i < 2; ++i)
   {
      const Real px = ox + hx*(cell+i), py = oy + hy*j;
      input[4*cell+i+2*j] = 1+px+2*py;
      data[4*cell+i+2*j] = 2-.25*px+py;
   }
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeFiniteElementField<"g">(space, inflow),
      MakeIntegrationDomain<"mesh">(mesh));
   // The constant and linear halves must not retain the discarded source.
   const auto lhs_context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(mesh));
   const auto rhs_context = MakeWeakFormContext(MakeTestField<"v">(space),
      MakeFiniteElementField<"g">(space, inflow), MakeIntegrationDomain<"mesh">(mesh));
   const auto form = integrate(boundary, upwind(beta, u, g)*v);
   const auto residual = MakeWeakFormResidualOperator<Policy>(form, context, rule);
   const auto linear = MakeWeakFormResidualOperator<Policy>(lhs(form), lhs_context, rule);
   const auto load = MakeWeakFormResidualOperator<Policy>(rhs(form), rhs_context, rule);
   std::array<Real, 8> expected{}, linear_expected{}, load_expected{}, bound_expected{};
   BoundaryPoints(2, 1, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real speed = 2*point.nx-.5*point.ny;
      const Real inside = 1+point.x+2*point.y, outside = 2-.25*point.x+point.y;
      AddP1(expected, point, hx, hy, speed*(speed >= 0 ? inside : outside));
      AddP1(linear_expected, point, hx, hy, speed >= 0 ? speed*inside : 0);
      AddP1(load_expected, point, hx, hy, speed < 0 ? -speed*outside : 0);
      AddP1(bound_expected, point, hx, hy, speed*(speed >= 0 ? 3 : outside));
   });
   Vector result(11), linear_result(11), load_result(11);
   result = 9;
   residual(x, result);
   bool ok = Check(result, expected, "boundary upwind outward flux");
   result = -2;
   residual.ApplyAdd(x, result);
   ok = Check(result, expected, "boundary upwind additive flux", 1, -2) && ok;
   linear(x, linear_result);
   load(load_result);
   ok = linear.InputSize() == 8 && load.InputSize() == 0 && ok;
   ok = Check(linear_result, linear_expected, "boundary upwind lhs") && ok;
   ok = CheckLoad(load, load_expected, "boundary upwind rhs") && ok;
   for (size_t i = 0; i < expected.size(); ++i)
   {
      ok = std::abs(linear_result.ReadHostData()[i]-load_result.ReadHostData()[i]-expected[i]) < 3e-11 && ok;
   }
   const auto wrapped = MakeWeakFormResidualOperator<Policy>(
      integrate(boundary, average(jump(upwind(beta, jump(u), average(g))))*jump(average(v))),
      context, rule);
   wrapped(x, result);
   ok = Check(result, expected, "boundary upwind nested identity wrappers") && ok;
   const auto bound_load = MakeWeakFormResidualOperator<Policy>(
      integrate(boundary, upwind(beta, constant, g)*v), rhs_context, rule);
   ok = CheckLoad(bound_load, bound_expected, "boundary upwind input-free states") && ok;
   const auto zero = MakeWeakFormResidualOperator<Policy>(
      rhs(integrate(boundary, upwind(beta, u, 2.0*u)*v)),
      MakeWeakFormContext(MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(mesh)), rule);
   ok = CheckLoad(zero, std::array<Real, 8>{}, "boundary upwind pure-linear zero rhs") && ok;

   // Assembly of the affine residual subtracts its complete F(0). Both
   // containers must therefore reproduce the independent outflow reference.
   const auto raw = GenericAssembly<MatrixAssemblyType::RawCOO, Policy>(form, context, rule);
   if constexpr (Policy::is_device_configuration) { GENDIL_DEVICE_SYNC; }
   result = 0;
   auto* assembled = result.ReadWriteHostData();
   const auto* rows = ReadHost(raw.rows);
   const auto* columns = ReadHost(raw.cols);
   const auto* entries = ReadHost(raw.values);
   const auto* values = x.ReadHostData();
   ok = raw.num_rows == 8 && raw.num_cols == 8 && ok;
   for (GlobalIndex i = 0; i < raw.nnz_raw; ++i)
   {
      assembled[rows[i]] += entries[i]*values[columns[i]];
   }
   ok = Check(result, linear_expected, "boundary upwind affine RawCOO") && ok;
   const auto matrix = GenericAssembly<MatrixAssemblyType::BSR, Policy>(lhs(form), lhs_context, rule);
   Vector matrix_result(8);
   matrix(x, matrix_result);
   return Check(matrix_result, linear_expected, "boundary upwind lhs BSR") && ok;
}

template<class Policy>
bool CheckCoupledStates()
{
   constexpr Real hx = .5, hy = .75, ox = .25, oy = -.5;
   CartesianMesh<2> mesh({1, 1}, {hx, hy}, {ox, oy}, false);
   const auto finite_element = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto u_space = MakeFiniteElementSpace(mesh, finite_element,
      ContiguousL2RestrictionSpecification{1, 12});
   const auto p_space = MakeFiniteElementSpace(mesh, finite_element,
      ContiguousL2RestrictionSpecification{7, 12});
   const auto v_space = MakeFiniteElementSpace(mesh, finite_element,
      ContiguousL2RestrictionSpecification{2, 14});
   const auto q_space = MakeFiniteElementSpace(mesh, finite_element,
      ContiguousL2RestrictionSpecification{8, 14});
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(u_space),
      MakeTrialField<"p">(p_space), MakeTestField<"v">(v_space), MakeTestField<"q">(q_space),
      MakeIntegrationDomain<"mesh">(mesh));
   const TrialSpace<"u"> u;
   const TrialSpace<"p"> p;
   const TestSpace<"v"> v;
   const TestSpace<"q"> q;
   const auto beta = MakeVectorCoefficient<"beta">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{2, -.5}; });
   const auto form = integrate(BoundaryFacets<"mesh">{}, upwind(beta, u*u, p)*v)
      + integrate(BoundaryFacets<"mesh">{}, upwind(beta, p, u*p)*q);
   const auto op = MakeWeakFormResidualOperator<Policy>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{}));
   Vector x(12), y(17);
   x = 13;
   auto* input = x.ReadWriteHostData();
   for (size_t j = 0; j < 2; ++j)
   for (size_t i = 0; i < 2; ++i)
   {
      input[1+i+2*j] = 1+ox+hx*i+2*(oy+hy*j);
      input[7+i+2*j] = 2-ox-hx*i+oy+hy*j;
   }
   std::array<Real, 4> v_expected{}, q_expected{};
   BoundaryPoints(1, 1, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real speed = 2*point.nx-.5*point.ny;
      const Real uq = 1+point.x+2*point.y, pq = 2-point.x+point.y;
      AddP1(v_expected, point, hx, hy, speed*(speed >= 0 ? uq*uq : pq));
      AddP1(q_expected, point, hx, hy, speed*(speed >= 0 ? pq : uq*pq));
   });
   std::array<Real, 14> expected{};
   for (size_t i = 0; i < 4; ++i)
   {
      expected[2+i] = v_expected[i];
      expected[8+i] = q_expected[i];
   }
   bool ok = op.InputSize() == 12 && op.OutputSize() == 14;
   y = 9;
   op(x, y);
   ok = Check(y, expected, "coupled nonlinear boundary upwind") && ok;
   y = -2;
   op.ApplyAdd(x, y);
   ok = Check(y, expected, "coupled nonlinear boundary upwind additive", 1, -2) && ok;

   // The callback reads p from the common active vector. The sign decision
   // may depend on the solution during residual evaluation, even though such
   // a selector is intentionally inadmissible to strict affine extraction.
   const auto active_beta = MakeVectorCoefficient<"active_beta", FieldValue<"p">>(
      [] GENDIL_HOST_DEVICE (Real value) { return std::array<Real, 2>{value, 0}; });
   const auto active_velocity = MakeWeakFormResidualOperator<Policy>(
      integrate(BoundaryFacets<"mesh">{}, upwind(active_beta, u, p)*v), context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{}));
   std::array<Real, 4> active_rows{};
   BoundaryPoints(1, 1, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real uq = 1+point.x+2*point.y, pq = 2-point.x+point.y;
      const Real speed = pq*point.nx;
      AddP1(active_rows, point, hx, hy, speed*(speed >= 0 ? uq : pq));
   });
   expected.fill(0);
   for (size_t i = 0; i < 4; ++i) { expected[2+i] = active_rows[i]; }
   y = 9;
   active_velocity(x, y);
   return Check(y, expected, "boundary upwind active-dependent velocity") && ok;
}

template<class Policy>
bool CheckVectorStates()
{
   constexpr Real hx = .5, hy = .75, ox = .25, oy = -.5;
   CartesianMesh<2> mesh({1, 1}, {hx, hy}, {ox, oy}, false);
   const auto finite_element = MakeVectorFiniteElement(
      MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}),
      MakeLegendreFiniteElement(FiniteElementOrders<0, 0>{}));
   const auto space = MakeFiniteElementSpace(mesh, finite_element);
   const VectorTrialSpace<"u"> u;
   const VectorTestSpace<"v"> v;
   const auto beta = MakeVectorCoefficient<"beta">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{2, 0}; });
   const FiniteElementField<"g"> inflow;
   Vector inflow_data(5);
   auto* prescribed = inflow_data.WriteHostData();
   for (size_t i = 0; i < 4; ++i) { prescribed[i] = 4; }
   prescribed[4] = -2;
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeFiniteElementField<"g">(space, inflow_data),
      MakeIntegrationDomain<"mesh">(mesh));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
   const auto form = integrate(BoundaryFacets<"mesh">{}, dot(upwind(beta, u, inflow), v));
   const auto op = MakeWeakFormResidualOperator<Policy>(form, context, rule);
   const auto linear = MakeWeakFormResidualOperator<Policy>(lhs(form),
      MakeWeakFormContext(MakeTrialField<"u">(space), MakeTestField<"v">(space),
         MakeIntegrationDomain<"mesh">(mesh)), rule);
   const auto load = MakeWeakFormResidualOperator<Policy>(rhs(form),
      MakeWeakFormContext(MakeTestField<"v">(space), MakeFiniteElementField<"g">(space, inflow_data),
         MakeIntegrationDomain<"mesh">(mesh)), rule);
   Vector x(5), y(8);
   auto* input = x.WriteHostData();
   for (size_t j = 0; j < 2; ++j)
   for (size_t i = 0; i < 2; ++i)
   {
      input[i+2*j] = 1+ox+hx*i+2*(oy+hy*j);
   }
   input[4] = 3;
   std::array<Real, 4> first_component{}, first_linear{}, first_load{};
   std::array<Real, 5> expected{}, linear_expected{}, load_expected{};
   BoundaryPoints(1, 1, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real speed = 2*point.nx;
      AddP1(first_component, point, hx, hy,
         speed*(speed >= 0 ? 1+point.x+2*point.y : 4));
      AddP1(first_linear, point, hx, hy, speed > 0 ? speed*(1+point.x+2*point.y) : 0);
      AddP1(first_load, point, hx, hy, speed < 0 ? -speed*4 : 0);
      expected[4] += point.weight*speed*(speed >= 0 ? 3 : -2);
      linear_expected[4] += point.weight*(speed > 0 ? speed*3 : 0);
      load_expected[4] += point.weight*(speed < 0 ? speed*2 : 0);
   });
   for (size_t i = 0; i < 4; ++i)
   {
      expected[i] = first_component[i];
      linear_expected[i] = first_linear[i];
      load_expected[i] = first_load[i];
   }
   y = 9;
   op(x, y);
   bool ok = Check(y, expected, "vector boundary upwind with zero-speed faces");
   linear(x, y);
   ok = Check(y, linear_expected, "vector boundary upwind lhs omits bound data") && ok;
   ok = CheckLoad(load, load_expected, "vector boundary upwind rhs retains state shape") && ok;

   // A direct vector FE velocity uses tensor storage, whereas the boundary
   // normal uses std::array. Both must work in the actual device kernel too.
   const auto field_velocity = MakeWeakFormResidualOperator<Policy>(
      integrate(BoundaryFacets<"mesh">{}, dot(upwind(u, u, inflow), v)), context, rule);
   first_component.fill(0);
   expected.fill(0);
   BoundaryPoints(1, 1, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real first_value = 1+point.x+2*point.y;
      const Real speed = first_value*point.nx+3*point.ny;
      AddP1(first_component, point, hx, hy, speed*(speed >= 0 ? first_value : 4));
      expected[4] += point.weight*speed*(speed >= 0 ? 3 : -2);
   });
   for (size_t i = 0; i < 4; ++i) { expected[i] = first_component[i]; }
   field_velocity(x, y);
   return Check(y, expected, "boundary upwind with direct vector-field velocity") && ok;
}

template<class Policy>
bool CheckPartitionBoundary()
{
   const auto partition = MakeCartesian1DPartition(.5, std::array<GlobalIndex, 2>{1, 1});
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   const auto space = MakeMixedFiniteElementSpace(partition, std::tuple{p1, p2}, DGDirectSumNumbering{});
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(space),
      MakeTestField<"v">(space), MakeIntegrationDomain<"mesh">(partition));
   const auto beta = MakeVectorCoefficient<"beta">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real, 1>{2}; });
   const auto inflow = MakeCoefficient<"inflow">(
      [] GENDIL_HOST_DEVICE () { return Real(3); });
   const auto op = MakeWeakFormResidualOperator<Policy>(
      integrate(BoundaryFacets<"mesh">{}, upwind(beta, TrialSpace<"u">{}, inflow)*TestSpace<"v">{}),
      context, MakeIntegrationRule(IntegrationRuleNumPoints<4>{}));
   Vector x(5), y(8);
   auto* input = x.WriteHostData();
   for (size_t i = 0; i < 5; ++i) { input[i] = Real(i+1); }
   y = 9;
   op(x, y);
   // Only the two exterior endpoints contribute. Internal part boundaries
   // must not be treated as physical inflow or outflow.
   return Check(y, std::array<Real, 5>{-6, 0, 0, 0, 10}, "partition boundary upwind");
}
}

int main()
{
   bool ok = CheckSelectedStateEvaluation() &&
      CheckScalarFluxAndAffineSplit<SerialKernelConfiguration>() &&
      CheckCoupledStates<SerialKernelConfiguration>() &&
      CheckVectorStates<SerialKernelConfiguration>() &&
      CheckPartitionBoundary<SerialKernelConfiguration>();
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<4, 4>, 2, 4>;
   using Policy1D = DeviceKernelConfiguration<ThreadBlockLayout<4>, 1, 4>;
   ok = CheckScalarFluxAndAffineSplit<Policy>() && CheckCoupledStates<Policy>() &&
      CheckVectorStates<Policy>() && CheckPartitionBoundary<Policy1D>() && ok;
#endif
   return ok ? 0 : 1;
}
