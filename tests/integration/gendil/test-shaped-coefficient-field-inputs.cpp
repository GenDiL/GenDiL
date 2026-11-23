// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

namespace
{
using namespace gendil;
using Policy = SerialKernelConfiguration;

constexpr std::array<std::array<Real, 2>, 3> gradient{{
   {{1, 2}}, {{3, -4}}, {{5, 6}}
}};
// On [1,3] x [-1,2], these are the exact Q1 weak loads for w_2 and
// w_2 + d_y w_2. Local order is (0,0), (1,0), (0,1), (1,1).
constexpr std::array<Real, 4> third_component_load{{12.5, 17.5, 21.5, 26.5}};
constexpr std::array<Real, 4> combined_load{{21.5, 26.5, 30.5, 35.5}};

template<size_t N>
bool Check(const Vector& result, const std::array<Real, N>& expected,
           const std::string& label, Real initial = 0)
{
   if (result.Size() != N)
   {
      std::cerr << label << ": wrong output size\n";
      return false;
   }
   bool ok = true;
   const auto* actual = result.ReadHostData();
   for (size_t i = 0; i < N; ++i)
   {
      const Real reference = initial + expected[i];
      // These low-order, bounded affine fixtures have only short reductions.
      // Keep an absolute allowance for zero rows and check finiteness explicitly.
      if (!std::isfinite(actual[i]) || !std::isfinite(reference) ||
          std::abs(actual[i] - reference) > 1e-12 * (1 + std::abs(reference)))
      {
         std::cerr << label << " row " << i << ": " << actual[i]
                   << ", expected " << reference << '\n';
         ok = false;
      }
   }
   return ok;
}

template<bool Active, class Operator, size_t N>
bool CheckAction(const Operator& op, const Vector& field,
                 const std::array<Real, N>& expected, const std::string& label)
{
   bool ok = op.OutputSize() == N &&
             op.InputSize() == (Active ? field.Size() : 0);
   Vector result(N);
   result = 9;
   if constexpr (Active) { op(field, result); }
   else { op(result); }
   ok = Check(result, expected, label + " replacement") && ok;
   result = -2;
   if constexpr (Active) { op.ApplyAdd(field, result); }
   else { op.ApplyAdd(result); }
   return Check(result, expected, label + " addition", -2) && ok;
}

template<class Space>
void FillVectorField(const Space& space, Vector& field, bool interior = false)
{
   auto dofs = MakeWriteOnlyElementTensorView<Policy>(space, field);
   for (GlobalIndex cell = 0; cell < space.GetNumberOfCells(); ++cell)
   for (GlobalIndex j = 0; j < 2; ++j)
   for (GlobalIndex i = 0; i < 2; ++i)
   {
      // Explicit Q1 endpoints and affine physical coordinates; no production
      // basis, quadrature, cell-map, or derivative helper forms the fixture.
      const Real x = 1. + 2. * (cell + i), y = -1. + 3. * j;
      if (interior && cell == 0)
      {
         std::get<0>(dofs)(i,j,cell) = 1 - 2*x + 7*y;
         std::get<1>(dofs)(i,j,cell) = -3 + 4*x + y;
         std::get<2>(dofs)(i,j,cell) = 2 - x - 5*y;
      }
      else
      {
         std::get<0>(dofs)(i,j,cell) = x + 2*y;
         std::get<1>(dofs)(i,j,cell) = 3*x - 4*y;
         std::get<2>(dofs)(i,j,cell) = 5*x + 6*y;
      }
   }
}

template<bool Active, class VectorSpace, class ScalarSpace>
auto VectorContext(const VectorSpace& vector_space, const ScalarSpace& scalar_space,
                   const CartesianMesh<2>& mesh, const Vector& field)
{
   if constexpr (Active)
   {
      return MakeWeakFormContext(MakeTrialField<"w">(vector_space),
         MakeTestField<"v">(scalar_space), MakeIntegrationDomain<"mesh">(mesh));
   }
   else
   {
      return MakeWeakFormContext(MakeFiniteElementField<"w">(vector_space,
         MakeReadOnlyElementTensorView<Policy>(vector_space, field)),
         MakeTestField<"v">(scalar_space), MakeIntegrationDomain<"mesh">(mesh));
   }
}

auto ThirdComponent()
{
   return MakeVectorCoefficient<"third_component">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real,3>{0,0,1}; });
}

template<bool Active, class Coefficient>
auto VectorVolumeForm(const Coefficient& coefficient)
{
   const Cells<"mesh"> cells;
   const TestSpace<"v"> v;
   if constexpr (Active)
   {
      const VectorTrialSpace<"w"> w;
      // A real trial occurrence marks w active; its exact load is added below.
      return integrate(cells, (coefficient + dot(w, ThirdComponent())) * v);
   }
   else { return integrate(cells, coefficient * v); }
}

template<bool Active>
bool CheckVectorVolume()
{
   CartesianMesh<2> mesh({1,1}, {2.,3.}, {1.,-1.});
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto scalar_space = MakeFiniteElementSpace(mesh, p1);
   const auto vector_space = MakeFiniteElementSpace(mesh, MakeVectorFiniteElement(p1,p1,p1));
   Vector field(vector_space.GetNumberOfFiniteElementDofs());
   FillVectorField(vector_space, field);
   const auto context = VectorContext<Active>(vector_space, scalar_space, mesh, field);
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<2,2>{});
   const std::string label = Active ? "active vector" : "prescribed vector";

   // Vector input declarations do not change this coefficient's scalar output.
   const auto coefficient = MakeCoefficient<"combined", VectorFieldValue<"w">,
      VectorFieldGradient<"w">>(
      [] GENDIL_HOST_DEVICE (const auto& value, const auto& derivative)
      {
         static_assert(get_rank_v<decltype(value)> == 1);
         static_assert(get_rank_v<decltype(derivative)> == 2);
         static_assert(GetTensorSize<0,std::remove_cvref_t<decltype(value)>>::value == 3);
         static_assert(GetTensorSize<0,std::remove_cvref_t<decltype(derivative)>>::value == 3);
         static_assert(GetTensorSize<1,std::remove_cvref_t<decltype(derivative)>>::value == 2);
         return value(2) + derivative(2,1);
      });
   const auto form = VectorVolumeForm<Active>(coefficient);
   using Signature = form_signature_t<decltype(form)>;
   if constexpr (Active)
   {
      static_assert(contains_named_field_requirement_v<typename Signature::active_inputs,"w">);
      static_assert(!contains_named_field_requirement_v<typename Signature::bound_inputs,"w">);
   }
   else
   {
      static_assert(contains_named_field_requirement_v<typename Signature::bound_inputs,"w">);
   }
   auto expected = combined_load;
   if constexpr (Active)
      for (size_t i = 0; i < expected.size(); ++i) { expected[i] += third_component_load[i]; }
   const auto op = MakeWeakFormResidualOperator<Policy>(form, context, rule);
   bool ok = CheckAction<Active>(op, field, expected, label + " value and gradient");

   // Check every component/direction separately, so entry errors cannot cancel
   // in a weighted scalar contraction. Each constant gradient load is 6/4*g.
   for (Integer component = 0; component < 3; ++component)
   for (Integer direction = 0; direction < 2; ++direction)
   {
      const auto entry = MakeCoefficient<"gradient_entry", VectorFieldGradient<"w">>(
         [component,direction] GENDIL_HOST_DEVICE (const auto& derivative)
         { return derivative(component,direction); });
      const auto entry_op = MakeWeakFormResidualOperator<Policy>(
         VectorVolumeForm<Active>(entry), context, rule);
      std::array<Real,4> entry_expected{};
      for (size_t i = 0; i < entry_expected.size(); ++i)
      {
         entry_expected[i] = 1.5 * gradient[component][direction];
         if constexpr (Active) { entry_expected[i] += third_component_load[i]; }
      }
      ok = CheckAction<Active>(entry_op, field, entry_expected,
         label + " gradient " + std::to_string(component) + "," +
         std::to_string(direction)) && ok;
   }
   if constexpr (Active)
   {
      auto* values = field.ReadWriteHostData();
      for (Integer i = 0; i < field.Size(); ++i) { values[i] *= -2; }
      for (auto& value : expected) { value *= -2; }
      ok = CheckAction<Active>(op, field, expected, label + " updated input") && ok;
   }
   return ok;
}

template<bool Active>
bool CheckScalarVolume()
{
   CartesianMesh<2> mesh({1,1}, {2.,3.}, {1.,-1.});
   const auto space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1,1>{}));
   Vector field(4);
   auto* values = field.WriteHostData();
   const std::array<Real,4> nodes{{2,-2,11,7}}; // s=7-2x+3y at Q1 endpoints.
   for (size_t i = 0; i < nodes.size(); ++i) { values[i] = nodes[i]; }
   const auto context = [&]
   {
      if constexpr (Active)
         return MakeWeakFormContext(MakeTrialField<"s">(space), MakeTestField<"v">(space),
            MakeIntegrationDomain<"mesh">(mesh));
      else
         return MakeWeakFormContext(MakeFiniteElementField<"s">(space,
            MakeReadOnlyElementTensorView<Policy>(space, field)), MakeTestField<"v">(space),
            MakeIntegrationDomain<"mesh">(mesh));
   }();
   const auto coefficient = MakeCoefficient<"scalar", FieldValue<"s",FieldShape::Scalar>,
      FieldGradient<"s",FieldShape::Scalar>>(
      [] GENDIL_HOST_DEVICE (Real value, const auto& derivative)
      { return value + derivative[0] - derivative[1]; });
   const auto form = [&]
   {
      const Cells<"mesh"> cells;
      const TestSpace<"v"> v;
      if constexpr (Active) { return integrate(cells, (coefficient+TrialSpace<"s">{})*v); }
      else { return integrate(cells, coefficient*v); }
   }();
   const auto op = MakeWeakFormResidualOperator<Policy>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<2,2>{}));
   // Integrate the explicit affine polynomials with integral l_i=1/2 and
   // integral t*l_i={1/6,1/3}; no GenDiL integration constructs the reference.
   const std::array<Real,4> expected = Active ? std::array<Real,4>{3.5,-.5,12.5,8.5}
                                            : std::array<Real,4>{-2,-4,2.5,.5};
   return CheckAction<Active>(op, field, expected,
      Active ? "active scalar" : "prescribed scalar");
}

template<bool Active>
bool CheckInteriorSides()
{
   CartesianMesh<2> mesh({2,1}, {2.,3.}, {1.,-1.}, false);
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto scalar_space = MakeFiniteElementSpace(mesh, p1);
   const auto vector_space = MakeFiniteElementSpace(mesh, MakeVectorFiniteElement(p1,p1,p1));
   Vector field(vector_space.GetNumberOfFiniteElementDofs());
   FillVectorField(vector_space, field, true);
   const auto context = VectorContext<Active>(vector_space, scalar_space, mesh, field);
   const auto coefficient = MakeCoefficient<"side", VectorFieldValue<"w">,
      VectorFieldGradient<"w">>(
      [] GENDIL_HOST_DEVICE (const auto& value, const auto& derivative)
      { return value(2)+derivative(2,1); });
   const InteriorFacets<"mesh"> interior;
   const TestSpace<"v"> v;
   const auto form = [&]
   {
      const auto cross = integrate(interior, minus(coefficient)*plus(v)+plus(coefficient)*minus(v));
      if constexpr (Active)
      {
         const VectorTrialSpace<"w"> w;
         return cross + integrate(interior, dot(minus(w),ThirdComponent())*minus(v));
      }
      else { return cross; }
   }();
   const auto op = MakeWeakFormResidualOperator<Policy>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<2,2>{}));
   // Cartesian global faces choose the higher-index cell as minus. At x=3,
   // c_minus=21+6y and c_plus=-6-5y. Integrating against lower/upper tangential
   // Q1 basis gives {31.5,40.5} and {-9,-16.5}, respectively. Cross-side test
   // traces distinguish coefficient-side selection from output-side selection.
   std::array<Real,8> expected{{0,31.5,0,40.5,-9,0,-16.5,0}};
   if constexpr (Active) { expected[4] += 22.5; expected[6] += 31.5; }
   return CheckAction<Active>(op, field, expected,
      Active ? "active interior sides" : "prescribed interior sides");
}
} // namespace

int main()
{
   bool ok = CheckVectorVolume<false>();
   ok = CheckVectorVolume<true>() && ok;
   ok = CheckScalarVolume<false>() && ok;
   ok = CheckScalarVolume<true>() && ok;
   ok = CheckInteriorSides<false>() && ok;
   ok = CheckInteriorSides<true>() && ok;
   if (!ok) { return 1; }
   std::cout << "Explicit scalar/vector coefficient field-input checks passed.\n";
   return 0;
}
