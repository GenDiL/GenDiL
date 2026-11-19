// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace gendil;

// Keep fixture-dependent template instantiations externally linked:
// AppleClang 21's OpenMP liveness analysis scales poorly for internal definitions.
namespace facet_execution_test
{
constexpr Integer dofs = 8;
using Matrix = std::array<Real, dofs * dofs>;

// Two Q1 cells occupy [0,1]x[0,1] and [1,3]x[0,1]. Geometry is
// value-owned, including on the GPU. Runtime connectivity uses identity
// coordinate permutations; rotated geometry has a separate regression test.
struct Cell
{
   static constexpr Integer Dim = 2;
   using geometry = HyperCube<2>;
   using physical_coordinates = std::array<Real,2>;
   using jacobian = std::array<Real,2>;
   template<class Rule> using QuadData = typename HyperCubeCell<2>::template QuadData<Rule>;
   Real origin;
   Real width;
   template<class Data>
   GENDIL_HOST_DEVICE void GetValuesAndJacobian(const TensorIndex<2>& qi, const Data& data,
      physical_coordinates& x, jacobian& J) const
   { x = {origin + width*GetCoord<0>(data,qi[0]), GetCoord<1>(data,qi[1])}; J = {width,1.0}; }
};

template<class Orientation>
GENDIL_HOST_DEVICE void ApplyOrientationToCell(const Orientation&, Cell&) {}

struct StaticMesh : CartesianMesh<2>
{
   using cell_type = Cell;
   StaticMesh() : CartesianMesh<2>({2,1},{1.,1.},{0.,0.},false) {}
   GENDIL_HOST_DEVICE cell_type GetCell(GlobalIndex cell) const
   { return {Real(cell),cell == 0 ? 1. : 2.}; }
};

struct RuntimeMesh : StaticMesh
{
   UnstructuredConformingConnectivity<HyperCube<2>> connectivity{2};
   RuntimeMesh()
   {
      for (Integer cell = 0; cell < 2; ++cell)
         for (Integer f = 0; f < 4; ++f)
            connectivity[cell].faces[f] = {GlobalIndex{}, {}, MakeReferencePermutation<2>(), {}, {}, true};
      connectivity[0].faces[2] = {1, {}, MakeReferencePermutation<2>(), {}, {}, false};
      connectivity[1].faces[0] = {0, {}, MakeReferencePermutation<2>(), {}, {}, false};
      ToDevice(2,connectivity.element_connectivities);
   }
   template<Integer Face>
   GENDIL_HOST_DEVICE auto GetLocalFaceInfo(GlobalIndex cell, std::integral_constant<Integer,Face> f) const
   { return connectivity.GetLocalFaceInfo(cell,f); }
};

enum class Term
{
   JumpJump, JumpAverage, AverageJump, AverageAverage,
   MinusPenalty, PlusPenalty, MinusJacobian, PlusJacobian, MinusPlus, PlusMinus,
   Gradient, Sipg, BoundValue, BoundGradient, Upwind, UpwindGradient
};

// Independent Q1 basis/physical derivatives and three-point Gauss integration.
// No GenDiL trace, mapping, quadrature, interpolation, or pullback code is used.
Matrix Expected(Term term, bool runtime, Real speed = 1.)
{
   Matrix result{};
   const std::array<Real,3> points{0.5-std::sqrt(0.15),0.5,0.5+std::sqrt(0.15)};
   const std::array<Real,3> weights{5./18.,4./9.,5./18.};
   const Integer minus_cell = runtime ? 0 : 1;
   const Real normal = runtime ? 1. : -1.;
   const Real rho_minus = minus_cell == 0 ? 1. : 0.5;
   const Real rho_plus = minus_cell == 0 ? 0.5 : 1.;
   for (Integer q = 0; q < 3; ++q)
   {
      std::array<Real,dofs> phi{}, jump{}, average{}, minus{}, plus{};
      std::array<std::array<Real,2>,dofs> gradient{};
      for (Integer cell = 0; cell < 2; ++cell)
         for (Integer j = 0; j < 2; ++j)
            for (Integer i = 0; i < 2; ++i)
            {
               const Integer k = 4*cell+i+2*j;
               const Real xi = cell == 0 ? 1. : 0.;
               const Real lx = i == 0 ? 1.-xi : xi;
               const Real ly = j == 0 ? 1.-points[q] : points[q];
               phi[k] = lx*ly;
               gradient[k] = {(2.*i-1.)*ly/(cell == 0 ? 1. : 2.), lx*(2.*j-1.)};
               jump[k] = (cell == minus_cell ? 1. : -1.)*phi[k];
               average[k] = 0.5*phi[k];
               (cell == minus_cell ? minus[k] : plus[k]) = phi[k];
            }
      for (Integer row = 0; row < dofs; ++row)
         for (Integer col = 0; col < dofs; ++col)
         {
            Real entry = 0.;
            switch (term)
            {
               case Term::JumpJump: entry = jump[col]*jump[row]; break;
               case Term::JumpAverage: entry = jump[col]*average[row]; break;
               case Term::AverageJump: entry = average[col]*jump[row]; break;
               case Term::AverageAverage: entry = average[col]*average[row]; break;
               case Term::MinusPenalty: entry = rho_minus*jump[col]*jump[row]; break;
               case Term::PlusPenalty: entry = rho_plus*jump[col]*jump[row]; break;
               case Term::MinusJacobian: entry = (rho_minus+2.)*jump[col]*jump[row]; break;
               case Term::PlusJacobian: entry = (rho_plus+2.)*jump[col]*jump[row]; break;
               case Term::MinusPlus: entry = minus[col]*plus[row]; break;
               case Term::PlusMinus: entry = plus[col]*minus[row]; break;
               case Term::Gradient:
                  for (Integer d = 0; d < 2; ++d)
                     entry += rho_minus*0.5*gradient[col][d]*gradient[row][d]
                        *(row/4 == minus_cell ? 1. : -1.);
                  break;
               case Term::Sipg:
                  entry = -0.5*normal*gradient[col][0]*jump[row]
                     -jump[col]*0.5*normal*gradient[row][0]
                     +90.*0.75*jump[col]*jump[row];
                  break;
               case Term::BoundValue:
                  entry = (minus_cell == 0 ? 2.3-0.2*points[q] : 4.6+0.7*points[q])
                     *jump[col]*jump[row];
                  break;
               case Term::BoundGradient:
                  entry = (minus_cell == 0 ? 1. : -0.1)*jump[col]*jump[row];
                  break;
               case Term::Upwind:
               case Term::UpwindGradient:
               {
                  const Real bn = normal*speed;
                  const Integer selected = bn >= 0 ? minus_cell : 1-minus_cell;
                  if (col/4 == selected)
                     entry = bn*jump[row]*(term == Term::Upwind ? phi[col]
                        : 1.3*gradient[col][0]-0.7*gradient[col][1]);
                  break;
               }
            }
            result[row*dofs+col] += weights[q]*entry;
         }
   }
   return result;
}

bool Check(const char* label, const Matrix& actual, const Matrix& expected)
{
   Real error = 0., scale = 1.;
   for (Integer i = 0; i < dofs*dofs; ++i)
   {
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      { std::cerr << label << ": non-finite entry\n"; return false; }
      error = std::max(error,std::abs(actual[i]-expected[i]));
      scale = std::max(scale,std::abs(expected[i]));
   }
   if (error <= 2.e-11*scale) return true;
   std::cerr << label << ": scaled error " << error/scale << '\n';
   return false;
}

template<class Operator>
Matrix Dense(const Operator& op)
{
   Matrix matrix{};
   Vector x(dofs), y(dofs);
   for (Integer column = 0; column < dofs; ++column)
   {
      // WriteHostData discards prior device contents: fill the whole basis vector.
      Real* basis = x.WriteHostData();
      std::fill_n(basis, dofs, 0.0);
      basis[column] = 1.0;
      op(x,y);
      GENDIL_DEVICE_SYNC;
      const Real* output = y.ReadHostData();
      for (Integer row = 0; row < dofs; ++row) matrix[row*dofs+column] = output[row];
   }
   return matrix;
}

template<class Policy, class Form, class Context, class Rule>
Matrix Evaluate(const Form& form, const Context& context, const Rule& rule)
{ return Dense(MakeGenericOperator<Policy>(form,context,rule)); }

template<class Policy, class Local, class Global, class Rule>
bool CheckForms(const Local& local, const Global& global, const Rule& rule, bool runtime)
{
   TrialSpace<"u"> u; TestSpace<"v"> v; InteriorFacets<"m"> faces;
   const auto rho = MakeCoefficient<"rho",InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real r) { return r; });
   const auto inverse = MakeCoefficient<"inverse_entries",JacobianInverse>(
      [] GENDIL_HOST_DEVICE (const std::array<Real,2>& J) {
         return GetValue(J,0)+2.*GetValue(J,1);
      });
   const auto w = MakeCoefficient<"w_value",FieldValue<"w">>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   const auto dw = MakeCoefficient<"w_gradient",FieldGradient<"w">>(
      [] GENDIL_HOST_DEVICE (const std::array<Real,2>& value) { return value[0]+2.*value[1]; });
   bool success = true;
   auto check = [&](const char* label, const auto& expression, Term term, Real speed = 1.) {
      const auto form = integrate(faces,expression);
      const auto a = Evaluate<Policy>(form,local,rule);
      const auto b = Evaluate<Policy>(form,global,rule);
      success = Check(label,a,Expected(term,runtime,speed)) && success;
      success = Check(label,b,Expected(term,runtime,speed)) && success;
      success = Check("local/global",a,b) && success;
   };
   check("jump-jump",jump(u)*jump(v),Term::JumpJump);
   check("jump-average",jump(u)*average(v),Term::JumpAverage);
   check("average-jump",average(u)*jump(v),Term::AverageJump);
   check("average-average",average(u)*average(v),Term::AverageAverage);
   check("minus normal scale",minus(rho)*jump(u)*jump(v),Term::MinusPenalty);
   check("plus normal scale",plus(rho)*jump(u)*jump(v),Term::PlusPenalty);
   check("minus diagonal inverse Jacobian",minus(inverse)*jump(u)*jump(v),Term::MinusJacobian);
   check("plus diagonal inverse Jacobian",plus(inverse)*jump(u)*jump(v),Term::PlusJacobian);
   check("explicit minus-plus traces",minus(u)*plus(v),Term::MinusPlus);
   check("explicit plus-minus traces",plus(u)*minus(v),Term::PlusMinus);
   check("gradient channel",minus(rho)*dot(average(grad(u)),jump(grad(v))),Term::Gradient);
   check("minus bound values",minus(w)*jump(u)*jump(v),Term::BoundValue);
   check("plus bound gradients",plus(dw)*jump(u)*jump(v),Term::BoundGradient);
   const auto flux = -average(dot(grad(u),Normal{}))*jump(v)
      -jump(u)*average(dot(grad(v),Normal{}));
   const auto penalty = 90.*average(rho)*jump(u)*jump(v);
   check("SIPDG",flux+penalty,Term::Sipg);
   const auto first = minus(rho)*jump(u)*jump(v);
   const auto second = average(u)*jump(v);
   auto grouping = [&](const auto& context, const auto& first_term, const auto& second_term) {
      const auto combined = Evaluate<Policy>(integrate(faces,first_term+second_term+flux),context,rule);
      const auto split = Evaluate<Policy>(integrate(faces,first_term)+integrate(faces,second_term)
         +integrate(faces,flux),context,rule);
      auto separate = Evaluate<Policy>(integrate(faces,first_term),context,rule);
      const auto b = Evaluate<Policy>(integrate(faces,second_term),context,rule);
      const auto c = Evaluate<Policy>(integrate(faces,flux),context,rule);
      for (Integer i = 0; i < dofs*dofs; ++i) separate[i] += b[i]+c[i];
      success = Check("split/combined",combined,split) && success;
      success = Check("separate/combined",combined,separate) && success;
   };
   grouping(local,first,second); grouping(global,first,second);
   const auto minus_inverse = minus(inverse)*jump(u)*jump(v);
   const auto plus_inverse = plus(inverse)*jump(u)*jump(v);
   grouping(local,minus_inverse,plus_inverse);
   grouping(global,minus_inverse,plus_inverse);
   const auto direction = MakeVectorCoefficient<"direction">(
      [] GENDIL_HOST_DEVICE () { return std::array<Real,2>{1.3,-0.7}; });
   for (Real speed : {Real(-2.),Real(0.),Real(2.)})
   {
      const auto beta = MakeVectorCoefficient<"beta">([speed] GENDIL_HOST_DEVICE () {
         return std::array<Real,2>{speed,0.5*speed}; });
      const auto a = upwind(beta,u)*jump(v);
      const auto b = dot(upwind(beta,grad(u)),direction)*jump(v);
      check("upwind",a,Term::Upwind,speed);
      check("upwind gradient",b,Term::UpwindGradient,speed);
      const auto combined = Evaluate<Policy>(integrate(faces,a+b),local,rule);
      success = Check("upwind split/combined",combined,
         Evaluate<Policy>(integrate(faces,a)+integrate(faces,b),local,rule)) && success;
      success = Check("upwind combined local/global",combined,
         Evaluate<Policy>(integrate(faces,a+b),global,rule)) && success;
   }
   if constexpr (!is_device_configuration_v<Policy>)
   {
      auto assembly = [&](const auto& form, const Matrix& reference) {
         success = Check("global action / independent assembly reference",
            Evaluate<Policy>(form,global,rule),reference) && success;
         success = Check("local BSR rows and neighbor columns",
            Dense(GenericAssembly<MatrixAssemblyType::BSR,Policy>(form,local,rule)),reference) && success;
         const auto raw = GenericAssembly<MatrixAssemblyType::RawCOO,Policy>(form,local,rule);
         Matrix assembled{};
         const auto* rows = ReadHost(raw.rows);
         const auto* columns = ReadHost(raw.cols);
         const auto* values = ReadHost(raw.values);
         for (size_t i = 0; i < raw.nnz_raw; ++i)
         {
            if (rows[i] >= dofs || columns[i] >= dofs)
            { std::cerr << "local COO index outside output\n"; success = false; return; }
            assembled[rows[i]*dofs+columns[i]] += values[i];
         }
         success = Check("local COO rows and neighbor columns",assembled,reference) && success;
         // Keep this serial assembly check independent of optional device
         // sparse-finalization libraries, including in CUDA/HIP builds.
         const auto coo = FinalizeRawCOOToCOOHost(raw,HostCOOBackend<>{});
         const auto csr = FinalizeRawCOOToCSRHost(raw,HostCSRBackend<>{});
         success = Check("finalized local COO / independent reference",
            Dense(coo),reference) && success;
         success = Check("finalized local CSR / independent reference",
            Dense(csr),reference) && success;
      };
      auto assembly_reference = Expected(Term::MinusPenalty,runtime);
      const auto average_jump_reference = Expected(Term::AverageJump,runtime);
      for (Integer i = 0; i < dofs*dofs; ++i)
         assembly_reference[i] += average_jump_reference[i];
      assembly(integrate(faces,first+second),assembly_reference);
      // Each form requests only one trial side and only the opposite test row.
      assembly(integrate(faces,minus(u)*plus(v)),Expected(Term::MinusPlus,runtime));
      assembly(integrate(faces,plus(u)*minus(v)),Expected(Term::PlusMinus,runtime));
   }
   return success;
}

template<class Policy, class Mesh, class Faces>
bool RunPolicy(const Mesh& mesh, const Faces& faces, bool runtime)
{
   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<3,3>{});
   const auto local_space = MakeFiniteElementSpace(mesh,fe);
   const auto partition = MakePartition(MakeCellPart(mesh),MakeInteriorFacePart<0,0>(faces));
   const auto global_space = MakeMixedFiniteElementSpace(partition,std::tuple{fe},DGDirectSumNumbering{});
   Vector bound(dofs);
   Real* values = bound.WriteHostData();
   for (Integer cell = 0; cell < 2; ++cell)
      for (Integer j = 0; j < 2; ++j)
         for (Integer i = 0; i < 2; ++i)
         {
            const Real x = cell == 0 ? Real(i) : 1.+2.*i;
            values[4*cell+i+2*j] = cell == 0 ? 2.+0.3*x-0.2*j : 5.-0.4*x+0.7*j;
         }
   const auto local = MakeWeakFormContext(MakeTrialField<"u">(local_space),MakeTestField<"v">(local_space),
      MakeFiniteElementField<"w">(local_space,MakeReadOnlyElementTensorView<Policy>(local_space,bound)),
      MakeIntegrationDomain<"m">(mesh));
   const auto global = MakeWeakFormContext(MakeTrialField<"u">(global_space),MakeTestField<"v">(global_space),
      MakeFiniteElementField<"w">(global_space,MakeReadOnlyElementTensorView<Policy>(local_space,bound)),
      MakeIntegrationDomain<"m">(partition));
   return CheckForms<Policy>(local,global,rule,runtime);
}
} // namespace

int main()
{
   using namespace facet_execution_test;
   const StaticMesh mesh;
   const auto static_faces = MakeCartesianInteriorFaceConnectivity<2>({2,1});
   const RuntimeMesh runtime;
   const auto runtime_faces = MakeUnstructuredInteriorFaceConnectivity(runtime);
   bool success = RunPolicy<SerialKernelConfiguration>(mesh,static_faces,false);
   success = RunPolicy<SerialKernelConfiguration>(runtime,runtime_faces,true) && success;
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<4,4>,2,1>;
   success = RunPolicy<Policy>(mesh,static_faces,false) && success;
   success = RunPolicy<Policy>(runtime,runtime_faces,true) && success;
#endif
   if (!success) return 1;
   std::cout << "Facet execution patterns match independent canonical Q1 matrices.\n";
   return 0;
}
