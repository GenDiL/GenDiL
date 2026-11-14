// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>
#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace gendil;

// External linkage avoids excessive AppleClang OpenMP optimization time for
// fixture-dependent template instantiations.
namespace mfem_sipdg_facet_test
{
constexpr Integer order = 2;
constexpr Integer mesh_order = 2;
constexpr Integer num_quad_1d = 5;
constexpr Real kappa = 10 * (order + 1) * (order + 1);
constexpr Integer global_batch_size = 4;
using Matrix = std::vector<Real>;

template<int Dim>
using LocalPolicy =
#if defined(GENDIL_USE_DEVICE)
   DeviceKernelConfiguration<ThreadBlockLayout<num_quad_1d,num_quad_1d>,Dim,1>;
#else
   SerialKernelConfiguration;
#endif

bool Check(bool condition, const std::string& message)
{
   if (!condition) std::cerr << "FAILED: " << message << '\n';
   return condition;
}

template<int Dim>
mfem::Mesh MakeMesh(bool curved, bool rotated)
{
   // Widths 1 and 2 make the two inverse normal cell sizes unequal even on
   // the affine fixture. Rotate the right cell before constructing topology.
   mfem::Mesh mesh(Dim, Dim == 2 ? 6 : 12, 2);
   for (int z = 0; z < (Dim == 2 ? 1 : 2); ++z)
      for (int y = 0; y < 2; ++y)
         for (int x = 0; x < 3; ++x)
         {
            double vertex[3]{double(x == 2 ? 3 : x), double(y), double(z)};
            mesh.AddVertex(vertex);
         }
   if constexpr (Dim == 2)
   {
      int left[4]{0,1,4,3}, right[4]{1,2,5,4};
      if (rotated) std::rotate(right, right + 1, right + 4);
      mesh.AddQuad(left);
      mesh.AddQuad(right);
   }
   else
   {
      int left[8]{0,1,4,3,6,7,10,9}, right[8]{1,2,5,4,7,8,11,10};
      if (rotated)
      {
         std::rotate(right, right + 1, right + 4);
         std::rotate(right + 4, right + 5, right + 8);
      }
      mesh.AddHex(left);
      mesh.AddHex(right);
   }
   // Finalize topology and generated boundary attributes together; manually
   // adding boundaries after FinalizeQuadMesh leaves its attribute list stale.
   mesh.FinalizeMesh();
   mesh.SetCurvature(mesh_order);
   if (curved)
      mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
      {
         y = x;
         y[0] += 0.2*x[1]*(1.-x[1]);
         if constexpr (Dim == 3) y[0] += 0.15*x[2]*(1.-x[2]);
      });
   return mesh;
}

template<int Dim>
const mfem::IntegrationRule& FaceRule()
{
   return mfem::IntRules.Get(Dim == 2 ? mfem::Geometry::SEGMENT : mfem::Geometry::SQUARE,
                            2*(num_quad_1d - 1));
}

template<int Dim>
bool CheckGeometry(mfem::Mesh& mesh, bool curved)
{
   const auto faces = mesh.GetFaceIndices(mfem::FaceType::Interior);
   if (!Check(faces.Size() == 1, "fixture must have one shared facet")) return false;
   auto* face = mesh.GetInteriorFaceTransformations(faces[0]);
   mfem::Vector normal(Dim), x(Dim);
   Real min_measure = std::numeric_limits<Real>::max(), max_measure = 0.;
   bool ok = true;
   const auto& rule = FaceRule<Dim>();
   for (int q = 0; q < rule.GetNPoints(); ++q)
   {
      face->SetAllIntPoints(&rule.IntPoint(q));
      mfem::CalcOrtho(face->Jacobian(), normal);
      const Real measure = normal.Norml2();
      min_measure = std::min(min_measure, measure);
      max_measure = std::max(max_measure, measure);
      face->Transform(rule.IntPoint(q), x);
      const Real expected_measure = curved ?
         std::sqrt(1. + std::pow(0.2*(1.-2.*x[1]), 2)
            + (Dim == 3 ? std::pow(0.15*(1.-2.*x[2]), 2) : 0.)) : 1.;
      const Real expected_x = 1. + (curved ?
         0.2*x[1]*(1.-x[1]) + (Dim == 3 ? 0.15*x[2]*(1.-x[2]) : 0.) : 0.);
      ok = Check(std::abs(measure - expected_measure) < 1.e-12 &&
                 std::abs(x[0] - expected_x) < 1.e-12,
                 "shared facet does not match analytic geometry") && ok;
      const Real det1 = face->Elem1->Jacobian().Det();
      const Real det2 = face->Elem2->Jacobian().Det();
      ok = Check(det1 > 0. && det2 > 0., "fixture has an inverted element") && ok;
      const Real rho1 = measure/det1, rho2 = measure/det2;
      ok = Check(std::isfinite(rho1) && std::isfinite(rho2) &&
                 std::abs(std::max(rho1,rho2)/std::min(rho1,rho2) - 2.) < 1.e-12,
                 "fixture must have unequal normal scales with ratio two") && ok;
   }
   return Check(!curved || max_measure - min_measure > 1.e-3,
                "curved fixture must have a varying shared-facet metric") && ok;
}

template<int Dim>
Matrix MFEMReference(mfem::Mesh& mesh, bool full, Real penalty = kappa)
{
   mfem::L2_FECollection collection(order, Dim, mfem::BasisType::GaussLobatto);
   mfem::FiniteElementSpace space(&mesh, &collection);
   const int n = space.GetNDofs();
   // Map MFEM's element-local nodal ordering explicitly to GenDiL's
   // contiguous, x-fastest tensor ordering, rather than assuming equality.
   std::vector<int> from_mfem(n, -1);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mfem::Array<int> dofs;
      space.GetElementDofs(e, dofs);
      const auto& nodes = space.GetFE(e)->GetNodes();
      for (int j = 0; j < dofs.Size(); ++j)
      {
         const auto& ip = nodes.IntPoint(j);
         const double coords[3]{ip.x, ip.y, ip.z};
         int lex = 0, stride = 1;
         for (int d = 0; d < Dim; ++d)
         {
            const int index = static_cast<int>(std::lround(order*coords[d]));
            GENDIL_VERIFY(index >= 0 && index <= order &&
                          std::abs(coords[d] - double(index)/order) < 1.e-12,
                          "Expected Q2 Gauss-Lobatto nodes in the MFEM reference.");
            lex += stride*index;
            stride *= order + 1;
         }
         GENDIL_VERIFY(dofs[j] >= 0 && from_mfem[dofs[j]] == -1,
                       "Expected disjoint unsigned L2 degrees of freedom.");
         from_mfem[dofs[j]] = e*dofs.Size() + lex;
      }
   }
   mfem::FunctionCoefficient mu([](const mfem::Vector& x) -> mfem::real_t
   {
      return 1. + 0.1*x[0] + 0.05*x[1] + (Dim == 3 ? 0.03*x[2] : 0.);
   });
   mfem::BilinearForm form(&space);
   auto* interior = new mfem::DGDiffusionIntegrator(mu, -1., penalty);
   interior->SetIntegrationRule(FaceRule<Dim>());
   form.AddInteriorFaceIntegrator(interior);
   if (full)
   {
      auto* volume = new mfem::DiffusionIntegrator(mu);
      volume->SetIntegrationRule(mfem::IntRules.Get(
         Dim == 2 ? mfem::Geometry::SQUARE : mfem::Geometry::CUBE,
         2*(num_quad_1d - 1)));
      form.AddDomainIntegrator(volume);
      auto* boundary = new mfem::DGDiffusionIntegrator(mu, -1., penalty);
      boundary->SetIntegrationRule(FaceRule<Dim>());
      form.AddBdrFaceIntegrator(boundary);
   }
   form.Assemble();
   form.Finalize();
   const auto& sparse = form.SpMat();
   const auto* rows = sparse.HostReadI();
   const auto* cols = sparse.HostReadJ();
   const auto* values = sparse.HostReadData();
   Matrix result(n*n, 0.);
   for (int row = 0; row < n; ++row)
      for (int entry = rows[row]; entry < rows[row+1]; ++entry)
         result[from_mfem[row]*n + from_mfem[cols[entry]]] = values[entry];
   return result;
}

bool Near(const Matrix& actual, const Matrix& expected, const std::string& label)
{
   if (!Check(actual.size() == expected.size(), label + " matrix size")) return false;
   Real error = 0., scale = 1.;
   for (size_t i = 0; i < actual.size(); ++i)
   {
      if (!Check(std::isfinite(actual[i]) && std::isfinite(expected[i]),
                 label + " non-finite entry")) return false;
      error = std::max(error, std::abs(actual[i] - expected[i]));
      scale = std::max(scale, std::abs(expected[i]));
   }
   std::cout << "  " << label << ": " << error/scale << '\n';
   return Check(error <= 2.e-11*scale, label);
}

bool Symmetric(const Matrix& matrix, int n, const std::string& label)
{
   Matrix transpose(matrix.size());
   for (int row = 0; row < n; ++row)
      for (int col = 0; col < n; ++col)
         transpose[row*n + col] = matrix[col*n + row];
   return Near(matrix, transpose, label + " symmetry");
}

template<class Operator>
Matrix DenseOperator(const Operator& op, int n)
{
   Vector x(n), y(n);
   Matrix matrix(n*n);
   for (int col = 0; col < n; ++col)
   {
      auto* input = x.WriteHostData();
      std::fill(input, input + n, Real(0.));
      input[col] = 1.;
      op(x, y);
      GENDIL_DEVICE_SYNC;
      const auto* output = y.ReadHostData();
      for (int row = 0; row < n; ++row) matrix[row*n + col] = output[row];
   }
   return matrix;
}

template<class Policy, class Form, class Context, class Rule>
Matrix Dense(const Form& form, const Context& context, const Rule& rule, int n)
{
   const auto op = MakeGenericOperator<Policy>(form, context, rule);
   return DenseOperator(op, n);
}

Matrix InterelementCoupling(const Matrix& matrix, int n)
{
   // This two-element broken space has contiguous, equally sized cell blocks.
   // Off-diagonal blocks contain only the shared interior-facet contribution.
   Matrix result(matrix);
   for (int row = 0; row < n; ++row)
      for (int col = 0; col < n; ++col)
         if (row/(n/2) == col/(n/2)) result[row*n + col] = 0.;
   return result;
}

template<class Partition, size_t... I>
auto ConformingPartition(const Partition& partition, std::index_sequence<I...>)
{
   GENDIL_VERIFY(((std::get<I + sizeof...(I)>(partition.InteriorFaceParts())
                     .face_mesh.GetNumberOfFaces() == 0) && ...),
                 "This test requires conforming geometry.");
   const auto count = (std::get<I>(partition.InteriorFaceParts())
                         .face_mesh.GetNumberOfFaces() + ...);
   GENDIL_VERIFY(count == 1 && count < global_batch_size,
                 "Expected one interior face, exercising an incomplete device batch.");
   return MakePartition(std::get<0>(partition.CellParts()),
                        std::get<I>(partition.InteriorFaceParts())...,
                        std::get<I>(partition.BoundaryFaceParts())...);
}

template<int Dim>
auto LocalMesh(mfem::Mesh& mesh)
{
   if constexpr (Dim == 2) return MakeQuadMesh<mesh_order>(mesh);
   else return MakeHexMesh<mesh_order>(mesh);
}

template<size_t... I>
auto FiniteElement(std::index_sequence<I...>)
{ return MakeLobattoFiniteElement(FiniteElementOrders<(static_cast<void>(I),order)...>{}); }

template<size_t... I>
auto IntegrationRule(std::index_sequence<I...>)
{ return MakeIntegrationRule(IntegrationRuleNumPoints<(static_cast<void>(I),num_quad_1d)...>{}); }

template<int Dim, class Combined, class Split, class LocalContext, class GlobalContext, class Rule>
bool CheckForms(const Combined& combined, const Split& split,
                const LocalContext& local, const GlobalContext& global,
                const Rule& rule, int n, const Matrix& reference, const std::string& label)
{
#if defined(GENDIL_USE_DEVICE)
   using GlobalPolicy = std::conditional_t<Dim == 2,
      DeviceKernelConfiguration<ThreadBlockLayout<num_quad_1d>,1,global_batch_size>,
      DeviceKernelConfiguration<ThreadBlockLayout<num_quad_1d,num_quad_1d>,2,global_batch_size>>;
#else
   using GlobalPolicy = SerialKernelConfiguration;
#endif
   // Imported geometry points to the selected backend's memory. Run GenDiL
   // with that backend; the separately assembled MFEM matrix is the host oracle.
   const auto lc = Dense<LocalPolicy<Dim>>(combined, local, rule, n);
   const auto ls = Dense<LocalPolicy<Dim>>(split, local, rule, n);
   const auto gc = Dense<GlobalPolicy>(combined, global, rule, n);
   const auto gs = Dense<GlobalPolicy>(split, global, rule, n);
   bool ok = Near(lc, reference, label + " local combined/MFEM");
   ok = Near(ls, reference, label + " local split/MFEM") && ok;
   ok = Near(gc, reference, label + " global combined/MFEM") && ok;
   ok = Near(gs, reference, label + " global split/MFEM") && ok;
   ok = Near(lc, gc, label + " local/global combined") && ok;
   ok = Near(ls, gs, label + " local/global split") && ok;
   ok = Near(lc, ls, label + " local combined/split") && ok;
   ok = Near(gc, gs, label + " global combined/split") && ok;
   ok = Symmetric(lc, n, label + " local") && ok;
   ok = Symmetric(gc, n, label + " global") && ok;
   return ok;
}

template<int Dim>
bool RunCase(bool curved, bool rotated)
{
   std::cout << "Q2 SIPDG, " << Dim << "D, unequal " << (curved ? "curved" : "affine")
             << " cells, right reference rotation " << rotated << '\n';
   auto mfem_mesh = MakeMesh<Dim>(curved, rotated);
   if (!CheckGeometry<Dim>(mfem_mesh, curved)) return false;
   const auto interior_reference = MFEMReference<Dim>(mfem_mesh, false);
   const auto full_reference = MFEMReference<Dim>(mfem_mesh, true);
   const auto no_penalty = MFEMReference<Dim>(mfem_mesh, false, 0.);
   Real penalty_difference = 0.;
   for (size_t i = 0; i < no_penalty.size(); ++i)
      penalty_difference = std::max(penalty_difference,
         std::abs(interior_reference[i] - no_penalty[i]));
   if (!Check(penalty_difference > 1., "reference must exercise a nonzero penalty")) return false;

   auto mesh = LocalMesh<Dim>(mfem_mesh);
   auto imported = MakeGlobalPartition<HyperCube<Dim>,mesh_order>(mfem_mesh);
   const auto partition = ConformingPartition(imported, std::make_index_sequence<2*Dim>{});
   const auto fe = FiniteElement(std::make_index_sequence<Dim>{});
   const auto local_space = MakeFiniteElementSpace(mesh, fe);
   const auto global_space = MakeMixedFiniteElementSpace(partition, std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   const int n = local_space.GetNumberOfFiniteElementDofs();
   if (!Check(n == global_space.GetNumberOfFiniteElementDofs() &&
              size_t(n*n) == interior_reference.size(), "inconsistent degree counts")) return false;
   const auto local = MakeWeakFormContext(MakeTrialField<"u">(local_space),
      MakeIntegrationDomain<"mesh">(mesh));
   const auto global = MakeWeakFormContext(MakeTrialField<"u">(global_space),
      MakeIntegrationDomain<"mesh">(partition));
   const auto rule = IntegrationRule(std::make_index_sequence<Dim>{});
   auto diffusivity = [] GENDIL_HOST_DEVICE (const std::array<Real,Dim>& x) -> Real
   {
      Real value = 1. + 0.1*x[0] + 0.05*x[1];
      if constexpr (Dim == 3) value += 0.03*x[2];
      return value;
   };
   const auto mu = MakeCoefficient<"diffusivity",PhysicalCoordinates>(diffusivity);
   const auto tau = MakeCoefficient<"penalty",InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real rho) -> Real { return kappa*rho; });
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"mesh"> faces;
   const auto consistency = -average(mu*dot(grad(u),Normal{}))*jump(v);
   const auto adjoint = -jump(u)*average(mu*dot(grad(v),Normal{}));
   // MFEM averages mu*rho from the two elements; its sigma=-1 gives SIPDG.
   const auto penalty = average(tau*mu)*jump(u)*jump(v);
   const auto combined = integrate(faces, consistency + adjoint + penalty);
   const auto split = integrate(faces, consistency) + integrate(faces, adjoint)
                    + integrate(faces, penalty);
   const auto other = integrate(Cells<"mesh">{}, mu*dot(grad(u),grad(v)))
      + integrate(BoundaryFacets<"mesh">{},
         -mu*dot(grad(u),Normal{})*v - u*mu*dot(grad(v),Normal{}) + tau*mu*u*v);
   // Compare the isolated interior contribution as well as the full operator:
   // volume and boundary terms must not mask an interior-facet discrepancy.
   bool ok = CheckForms<Dim>(combined, split, local, global, rule, n,
                             interior_reference, "interior");
   ok = CheckForms<Dim>(other + combined, other + split, local, global, rule, n,
                        full_reference, "full") && ok;

   const auto legacy_operator = MakeDiffusionOperator<LocalPolicy<Dim>>(
      local_space, rule, diffusivity, -1., kappa);
   const auto legacy = DenseOperator(legacy_operator, n);
   ok = Near(legacy, full_reference, "legacy full/MFEM") && ok;
   ok = Symmetric(legacy, n, "legacy full") && ok;
   const auto coupling = InterelementCoupling(legacy, n);
   ok = Near(coupling, InterelementCoupling(interior_reference, n),
              "legacy interior coupling/MFEM") && ok;
   return Symmetric(coupling, n, "legacy interior coupling") && ok;
}
} // namespace mfem_sipdg_facet_test

int main()
{
#if defined(GENDIL_USE_CUDA)
   mfem::Device device("cuda");
#elif defined(GENDIL_USE_HIP)
   mfem::Device device("hip");
#else
   mfem::Device device("cpu");
#endif
   bool ok = true;
   for (bool curved : {false, true})
      for (bool rotated : {false, true})
         ok = mfem_sipdg_facet_test::RunCase<GENDIL_SIPDG_TEST_DIM>(curved, rotated) && ok;
   return ok ? 0 : 1;
}
