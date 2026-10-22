// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
using namespace gendil;
using Policy = SerialKernelConfiguration;
template<size_t Dim> using Coordinates = std::array<Real, Dim>;

template<bool Recursive, size_t Components>
GENDIL_HOST_DEVICE
auto Store(const std::array<Real, Components>& values)
{
   if constexpr (Recursive)
   {
      SerialRecursiveArray<Real, Components> result{};
      for (size_t c = 0; c < Components; ++c) result(c) = values[c];
      return result;
   }
   else return values;
}

template<size_t Components, size_t Dim>
GENDIL_HOST_DEVICE
auto RepresentedField(const Coordinates<Dim>& x, bool polynomial)
{
   std::array<Real, Components> value{};
   for (size_t c = 0; c < Components; ++c)
   {
      value[c] = .7-.4*c;
      for (size_t d = 0; d < Dim; ++d)
         value[c] += (.2*(c+1)*(d+2)+(c == d ? .5 : -.3))*x[d];
      if (polynomial)
      {
         value[c] += (.15+.1*c)*x[0]*x[0]-(.3+.2*c)*x[0]*x[1];
         if constexpr (Dim == 3) value[c] += (.25-.15*c)*x[0]*x[1]*x[2];
      }
   }
   return value;
}

template<size_t Dim>
GENDIL_HOST_DEVICE
Real NonconstantError(size_t component, const Coordinates<Dim>& x)
{
   Real result = (.3*component-.2)*(1+x[0]*x[0])+(.4+.2*component)*x[1]
      + .1*(component+1)*x[0]*x[1];
   if constexpr (Dim == 3) result -= (.2+.15*component)*x[2]*x[2];
   return result;
}

bool Close(Real actual, Real expected, const char* label)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) || actual < 0 ||
       std::abs(actual-expected) > 2e-12*(1+std::abs(expected)))
   {
      std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
      return false;
   }
   return true;
}

template<class Space, class Rule, class Exact>
Real Error(const Space& space, const Rule& rule, Exact& exact,
           const Vector& input, bool& ok)
{
   const auto* before_data = input.ReadHostData();
   const std::vector<Real> before(before_data, before_data+input.Size());
   // Named lvalue callbacks deliberately isolate this regression from issue 11.
   const Real result = L2Error<Policy>(space, rule, exact, input);
   if (!std::isfinite(result) || result < 0)
   {
      std::cerr << "L2Error returned an invalid norm: " << result << '\n';
      ok = false;
   }
   const auto* after = input.ReadHostData();
   for (size_t i = 0; i < before.size(); ++i)
   {
      if (after[i] != before[i])
      {
         std::cerr << "L2Error modified its input at index " << i << '\n';
         ok = false;
         break;
      }
   }
   return result;
}

template<size_t Dim, size_t Components, bool H1>
bool CheckSpace()
{
   const auto mesh = []
   {
      if constexpr (Dim == 2) return CartesianMesh<2>({2, 3}, {.4, .3}, {.15, -.2}, false);
      else return CartesianMesh<3>({2, 2, 2}, {.4, .3, .5}, {.15, -.2, .35}, false);
   }();
   const auto scalar_fe = []
   {
      if constexpr (Dim == 2) return MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
      else return MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   }();
   const auto make_space = [&](const auto& fe)
   {
      if constexpr (H1) return MakeFiniteElementSpace(mesh, fe, CartesianH1RestrictionSpecification{});
      else return MakeFiniteElementSpace(mesh, fe);
   };
   const auto vector_space = make_space(MakeHomogeneousVectorFiniteElement<Components>(scalar_fe));
   const auto scalar_space = make_space(scalar_fe);
   const auto rule = []
   {
      if constexpr (Dim == 2) return MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{});
      else return MakeIntegrationRule(IntegrationRuleNumPoints<4, 4, 4>{});
   }();
   constexpr Real volume = Dim == 2 ? .8*.9 : .8*.6*1.;
   std::array<Real, Components> offsets{};
   Real offset_squared = 0;
   for (size_t c = 0; c < Components; ++c)
   {
      offsets[c] = (c%2 ? -1 : 1)*(.5+.75*c);
      offset_squared += offsets[c]*offsets[c];
   }
   bool ok = true;
   const auto check_storage = [&]<bool Recursive>()
   {
      for (const bool polynomial : {false, true})
      {
         auto represented = [=] GENDIL_HOST_DEVICE (const Coordinates<Dim>& x)
         { return Store<Recursive>(RepresentedField<Components>(x, polynomial)); };
         const auto input = InterpolateNodalField<Policy>(vector_space, represented);
         ok = Close(Error(vector_space, rule, represented, input, ok), 0,
            "represented vector field") && ok;
         auto shifted = [=] GENDIL_HOST_DEVICE (const Coordinates<Dim>& x)
         {
            auto value = RepresentedField<Components>(x, polynomial);
            for (size_t c = 0; c < Components; ++c) value[c] += offsets[c];
            return Store<Recursive>(value);
         };
         ok = Close(Error(vector_space, rule, shifted, input, ok), std::sqrt(volume*offset_squared),
            "constant component offsets") && ok;
         if (!polynomial) continue;
         auto exact = [] GENDIL_HOST_DEVICE (const Coordinates<Dim>& x)
         {
            auto value = RepresentedField<Components>(x, true);
            for (size_t c = 0; c < Components; ++c) value[c] += NonconstantError(c, x);
            return Store<Recursive>(value);
         };
         const Real vector_error = Error(vector_space, rule, exact, input, ok);
         Real scalar_squared = 0;
         for (size_t c = 0; c < Components; ++c)
         {
            auto scalar_field = [=] GENDIL_HOST_DEVICE (const Coordinates<Dim>& x)
            { return RepresentedField<Components>(x, true)[c]; };
            auto scalar_exact = [=] GENDIL_HOST_DEVICE (const Coordinates<Dim>& x)
            { return RepresentedField<Components>(x, true)[c]+NonconstantError(c, x); };
            const auto component_input = InterpolateNodalField<Policy>(scalar_space, scalar_field);
            const Real component_error = Error(scalar_space, rule, scalar_exact, component_input, ok);
            if (!(std::isfinite(component_error) && component_error > 1e-6)) ok = false;
            scalar_squared += component_error*component_error;
         }
         ok = Close(vector_error*vector_error, scalar_squared,
            "vector error squared equals sum of scalar component errors squared") && ok;
      }
   };
   check_storage.template operator()<false>();
   check_storage.template operator()<true>();
   if (!ok) std::cerr << Dim << "D " << Components << "-component " << (H1 ? "H1" : "DG")
                      << " vector L2 error failed\n";
   return ok;
}

bool CheckOriginalCase()
{
   const CartesianMesh<2> mesh({2, 2}, {.5, .5}, {0., 0.}, false);
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto space = MakeFiniteElementSpace(mesh, MakeHomogeneousVectorFiniteElement<2>(fe));
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{});
   Vector input(GetAlgebraicDofExtent(space));
   input = 0;
   auto array_exact = [] GENDIL_HOST_DEVICE (const Coordinates<2>&) { return std::array<Real, 2>{1, 2}; };
   auto recursive_exact = [] GENDIL_HOST_DEVICE (const Coordinates<2>&) { return Store<true>(std::array<Real, 2>{1, 2}); };
   bool ok = true;
   ok = Close(Error(space, rule, array_exact, input, ok), std::sqrt(Real{5}), "zero versus (1,2), array") && ok;
   return Close(Error(space, rule, recursive_exact, input, ok), std::sqrt(Real{5}), "zero versus (1,2), recursive array") && ok;
}
} // namespace

int main()
{
   bool ok = CheckOriginalCase();
   ok = CheckSpace<2, 3, false>() && ok;
   ok = CheckSpace<2, 3, true>() && ok;
   ok = CheckSpace<3, 3, false>() && ok;
   return CheckSpace<3, 3, true>() && ok ? 0 : 1;
}
