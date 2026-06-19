// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace gendil;

namespace
{

struct FullSharedThreadedFaceKernelPolicy :
   public DeviceKernelConfiguration<ThreadBlockLayout<3>, 1, 1>
{
   using face_read_dofs_policy = FullSharedFaceReadDofsPolicy;
   using face_write_dofs_policy = FullSharedFaceWriteDofsPolicy;
};

template <typename VectorType>
void FillPattern(VectorType& x)
{
   Real* data = x.WriteHostData();
   for (Integer i = 0; i < x.Size(); ++i)
   {
      data[i] = 0.37 + 0.19 * i + 0.03 * ((i % 5) + 1);
   }
}

template <typename OperatorType>
std::vector<Real> BuildDenseMatrix(
   const OperatorType& op,
   const Integer size)
{
   std::vector<Real> dense(static_cast<size_t>(size * size), 0.0);
   Vector x(size);
   Vector y(size);

   for (Integer col = 0; col < size; ++col)
   {
      x = 0.0;
      y = 0.0;
      x.WriteHostData()[col] = 1.0;
      op(x, y);

      const Real* y_data = y.ReadHostData();
      for (Integer row = 0; row < size; ++row)
      {
         dense[static_cast<size_t>(row * size + col)] = y_data[row];
      }
   }

   return dense;
}

std::vector<Real> MakeRankOneReference(const std::vector<Real>& trace)
{
   const size_t size = trace.size();
   std::vector<Real> ref(size * size, 0.0);
   for (size_t row = 0; row < size; ++row)
   {
      for (size_t col = 0; col < size; ++col)
      {
         ref[row * size + col] = trace[row] * trace[col];
      }
   }
   return ref;
}

void AddOuterProduct(
   std::vector<Real>& ref,
   const std::vector<Real>& row_trace,
   const std::vector<Real>& col_trace)
{
   GENDIL_VERIFY(
      row_trace.size() == col_trace.size(),
      "Trace vector sizes do not match.");
   const size_t size = row_trace.size();
   GENDIL_VERIFY(ref.size() == size * size, "Dense matrix size mismatch.");

   for (size_t row = 0; row < size; ++row)
   {
      for (size_t col = 0; col < size; ++col)
      {
         ref[row * size + col] += row_trace[row] * col_trace[col];
      }
   }
}

std::vector<Real> MakeOuterProductReference(
   const std::vector<Real>& row_trace,
   const std::vector<Real>& col_trace)
{
   const size_t size = row_trace.size();
   std::vector<Real> ref(size * size, 0.0);
   AddOuterProduct(ref, row_trace, col_trace);
   return ref;
}

std::vector<Real> ScaleDenseReference(
   std::vector<Real> ref,
   const Real scale)
{
   for (auto& value : ref)
   {
      value *= scale;
   }
   return ref;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceJumpTrace()
{
   if constexpr (P2Minus)
   {
      // p2 minus on the + face, p1 plus on the - face.
      return { 0.0, 0.0, 1.0, -1.0, 0.0 };
   }
   else
   {
      // p1 minus on the + face, p2 plus on the - face.
      return { 0.0, 1.0, -1.0, 0.0, 0.0 };
   }
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceMinusTrace()
{
   std::vector<Real> trace(5, 0.0);
   if constexpr (P2Minus)
   {
      trace[2] = 1.0;
   }
   else
   {
      trace[1] = 1.0;
   }
   return trace;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfacePlusTrace()
{
   std::vector<Real> trace(5, 0.0);
   if constexpr (P2Minus)
   {
      trace[3] = 1.0;
   }
   else
   {
      trace[2] = 1.0;
   }
   return trace;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceAverageTrace()
{
   auto trace = MakeScalarInterfaceMinusTrace<P2Minus>();
   auto plus_trace = MakeScalarInterfacePlusTrace<P2Minus>();
   for (size_t i = 0; i < trace.size(); ++i)
   {
      trace[i] = 0.5 * (trace[i] + plus_trace[i]);
   }
   return trace;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceUpwindTrialTrace(const Real speed)
{
   std::vector<Real> trace(5, 0.0);
   if constexpr (P2Minus)
   {
      trace[speed >= 0.0 ? 2 : 3] = speed;
   }
   else
   {
      trace[speed >= 0.0 ? 1 : 2] = speed;
   }
   return trace;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceUpwindReference(const Real speed)
{
   return MakeOuterProductReference(
      MakeScalarInterfaceJumpTrace<P2Minus>(),
      MakeScalarInterfaceUpwindTrialTrace<P2Minus>(speed));
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceMinusNormalGradientTrace()
{
   if constexpr (P2Minus)
   {
      // p2 minus on the + face, with canonical normal pointing minus-to-plus.
      return { 1.0, -4.0, 3.0, 0.0, 0.0 };
   }
   else
   {
      // p1 minus on the + face, with canonical normal pointing minus-to-plus.
      return { -1.0, 1.0, 0.0, 0.0, 0.0 };
   }
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfacePlusNormalGradientTrace()
{
   if constexpr (P2Minus)
   {
      // p1 plus on the - face. Normal{} remains canonical minus-to-plus.
      return { 0.0, 0.0, 0.0, -1.0, 1.0 };
   }
   else
   {
      // p2 plus on the - face. Normal{} remains canonical minus-to-plus.
      return { 0.0, 0.0, -3.0, 4.0, -1.0 };
   }
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceAverageMuNormalGradientTrace(
   const Real mu_minus,
   const Real mu_plus)
{
   auto trace = MakeScalarInterfaceMinusNormalGradientTrace<P2Minus>();
   auto plus_trace = MakeScalarInterfacePlusNormalGradientTrace<P2Minus>();
   for (size_t i = 0; i < trace.size(); ++i)
   {
      trace[i] = 0.5 * (mu_minus * trace[i] + mu_plus * plus_trace[i]);
   }
   return trace;
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceSipdgConsistencyReference(
   const Real mu_minus,
   const Real mu_plus)
{
   return ScaleDenseReference(
      MakeOuterProductReference(
         MakeScalarInterfaceJumpTrace<P2Minus>(),
         MakeScalarInterfaceAverageMuNormalGradientTrace<P2Minus>(
            mu_minus,
            mu_plus)),
      -1.0);
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceSipdgSymmetryReference(
   const Real mu_minus,
   const Real mu_plus)
{
   return ScaleDenseReference(
      MakeOuterProductReference(
         MakeScalarInterfaceAverageMuNormalGradientTrace<P2Minus>(
            mu_minus,
            mu_plus),
         MakeScalarInterfaceJumpTrace<P2Minus>()),
      -1.0);
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceSipdgPenaltyReference(const Real eta)
{
   return ScaleDenseReference(
      MakeRankOneReference(MakeScalarInterfaceJumpTrace<P2Minus>()),
      eta);
}

template <bool P2Minus>
std::vector<Real> MakeScalarInterfaceSipdgReference(
   const Real mu_minus,
   const Real mu_plus,
   const Real eta)
{
   auto ref =
      MakeScalarInterfaceSipdgConsistencyReference<P2Minus>(
         mu_minus,
         mu_plus);
   AddOuterProduct(
      ref,
      MakeScalarInterfaceAverageMuNormalGradientTrace<P2Minus>(
         -mu_minus,
         -mu_plus),
      MakeScalarInterfaceJumpTrace<P2Minus>());
   auto penalty = MakeScalarInterfaceSipdgPenaltyReference<P2Minus>(eta);
   for (size_t i = 0; i < ref.size(); ++i)
   {
      ref[i] += penalty[i];
   }
   return ref;
}

std::vector<Real> MakeSameSpaceP2InteriorSipdgConsistencyReference(
   const Real mu_minus,
   const Real mu_plus)
{
   const std::vector<Real> jump_trace{0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
   // Two p2 cells of size h=0.5. The canonical interior face in this fixture
   // uses the right cell's left face as minus and the left cell's right face as
   // plus; Normal{} points from that minus side to plus.
   const std::vector<Real> minus_normal_gradient{
      0.0, 0.0, 0.0, 6.0, -8.0, 2.0};
   const std::vector<Real> plus_normal_gradient{
      -2.0, 8.0, -6.0, 0.0, 0.0, 0.0};

   std::vector<Real> average_flux(jump_trace.size(), 0.0);
   for (size_t i = 0; i < average_flux.size(); ++i)
   {
      average_flux[i] =
         0.5 * (
            mu_minus * minus_normal_gradient[i] +
            mu_plus * plus_normal_gradient[i]);
   }
   return MakeOuterProductReference(jump_trace, average_flux);
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceJumpReference()
{
   constexpr size_t num_components = 2;
   constexpr size_t minus_order = P2Minus ? 2 : 1;
   constexpr size_t plus_order = P2Minus ? 1 : 2;
   constexpr size_t minus_scalar_dofs = minus_order + 1;
   constexpr size_t plus_scalar_dofs = plus_order + 1;
   constexpr size_t minus_dofs = num_components * minus_scalar_dofs;
   constexpr size_t total_dofs =
      minus_dofs + num_components * plus_scalar_dofs;

   std::vector<Real> ref(total_dofs * total_dofs, 0.0);
   for (size_t comp = 0; comp < num_components; ++comp)
   {
      std::vector<Real> trace(total_dofs, 0.0);
      trace[comp * minus_scalar_dofs + minus_order] = 1.0;
      trace[minus_dofs + comp * plus_scalar_dofs] = -1.0;
      AddOuterProduct(ref, trace, trace);
   }
   return ref;
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceExplicitTraceReference(
   const Real minus_weight,
   const Real plus_weight)
{
   constexpr size_t num_components = 2;
   constexpr size_t minus_order = P2Minus ? 2 : 1;
   constexpr size_t plus_order = P2Minus ? 1 : 2;
   constexpr size_t minus_scalar_dofs = minus_order + 1;
   constexpr size_t plus_scalar_dofs = plus_order + 1;
   constexpr size_t minus_dofs = num_components * minus_scalar_dofs;
   constexpr size_t total_dofs =
      minus_dofs + num_components * plus_scalar_dofs;

   std::vector<Real> ref(total_dofs * total_dofs, 0.0);
   for (size_t comp = 0; comp < num_components; ++comp)
   {
      std::vector<Real> row_trace(total_dofs, 0.0);
      std::vector<Real> col_trace(total_dofs, 0.0);
      const size_t minus_trace =
         comp * minus_scalar_dofs + minus_order;
      const size_t plus_trace =
         minus_dofs + comp * plus_scalar_dofs;
      row_trace[minus_trace] = 1.0;
      row_trace[plus_trace] = -1.0;
      col_trace[minus_trace] = minus_weight;
      col_trace[plus_trace] = plus_weight;
      AddOuterProduct(ref, row_trace, col_trace);
   }
   return ref;
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceUpwindReference(const Real speed)
{
   constexpr size_t num_components = 2;
   constexpr size_t minus_order = P2Minus ? 2 : 1;
   constexpr size_t plus_order = P2Minus ? 1 : 2;
   constexpr size_t minus_scalar_dofs = minus_order + 1;
   constexpr size_t plus_scalar_dofs = plus_order + 1;
   constexpr size_t minus_dofs = num_components * minus_scalar_dofs;
   constexpr size_t total_dofs =
      minus_dofs + num_components * plus_scalar_dofs;

   std::vector<Real> ref(total_dofs * total_dofs, 0.0);
   for (size_t comp = 0; comp < num_components; ++comp)
   {
      std::vector<Real> row_trace(total_dofs, 0.0);
      std::vector<Real> col_trace(total_dofs, 0.0);
      const size_t minus_trace =
         comp * minus_scalar_dofs + minus_order;
      const size_t plus_trace =
         minus_dofs + comp * plus_scalar_dofs;
      row_trace[minus_trace] = 1.0;
      row_trace[plus_trace] = -1.0;
      col_trace[speed >= 0.0 ? minus_trace : plus_trace] = speed;
      AddOuterProduct(ref, row_trace, col_trace);
   }
   return ref;
}

std::vector<Real> LiftScalarInterfaceReferenceToVector(
   const std::vector<Real>& scalar_ref,
   const size_t minus_scalar_dofs,
   const size_t plus_scalar_dofs)
{
   constexpr size_t num_components = 2;
   const size_t scalar_dofs = minus_scalar_dofs + plus_scalar_dofs;
   const size_t minus_dofs = num_components * minus_scalar_dofs;
   const size_t total_dofs = num_components * scalar_dofs;

   GENDIL_VERIFY(
      scalar_ref.size() == scalar_dofs * scalar_dofs,
      "Scalar reference size does not match scalar interface layout.");

   auto vector_index =
      [=] (const size_t component, const size_t scalar_index)
      {
         if (scalar_index < minus_scalar_dofs)
         {
            return component * minus_scalar_dofs + scalar_index;
         }
         return minus_dofs
              + component * plus_scalar_dofs
              + (scalar_index - minus_scalar_dofs);
      };

   std::vector<Real> ref(total_dofs * total_dofs, 0.0);
   for (size_t comp = 0; comp < num_components; ++comp)
   {
      for (size_t row = 0; row < scalar_dofs; ++row)
      {
         for (size_t col = 0; col < scalar_dofs; ++col)
         {
            ref[
               vector_index(comp, row) * total_dofs
               + vector_index(comp, col)] =
                  scalar_ref[row * scalar_dofs + col];
         }
      }
   }
   return ref;
}

std::vector<Real> LiftScalarReferenceToComponentBlocks(
   const std::vector<Real>& scalar_ref)
{
   constexpr size_t num_components = 2;
   const size_t scalar_dofs =
      static_cast<size_t>(std::sqrt(static_cast<Real>(scalar_ref.size())));
   GENDIL_VERIFY(
      scalar_ref.size() == scalar_dofs * scalar_dofs,
      "Scalar reference is not square.");

   const size_t total_dofs = num_components * scalar_dofs;
   std::vector<Real> ref(total_dofs * total_dofs, 0.0);
   for (size_t comp = 0; comp < num_components; ++comp)
   {
      for (size_t row = 0; row < scalar_dofs; ++row)
      {
         for (size_t col = 0; col < scalar_dofs; ++col)
         {
            ref[
               (comp * scalar_dofs + row) * total_dofs
               + (comp * scalar_dofs + col)] =
                  scalar_ref[row * scalar_dofs + col];
         }
      }
   }
   return ref;
}

template <bool P2Minus>
std::vector<Real> LiftScalarInterfaceReferenceToVector(
   const std::vector<Real>& scalar_ref)
{
   constexpr size_t minus_order = P2Minus ? 2 : 1;
   constexpr size_t plus_order = P2Minus ? 1 : 2;
   return LiftScalarInterfaceReferenceToVector(
      scalar_ref,
      minus_order + 1,
      plus_order + 1);
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceSipdgConsistencyReference(
   const Real mu_minus,
   const Real mu_plus)
{
   return LiftScalarInterfaceReferenceToVector<P2Minus>(
      MakeScalarInterfaceSipdgConsistencyReference<P2Minus>(
         mu_minus,
         mu_plus));
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceSipdgSymmetryReference(
   const Real mu_minus,
   const Real mu_plus)
{
   return LiftScalarInterfaceReferenceToVector<P2Minus>(
      MakeScalarInterfaceSipdgSymmetryReference<P2Minus>(
         mu_minus,
         mu_plus));
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceSipdgPenaltyReference(const Real eta)
{
   return LiftScalarInterfaceReferenceToVector<P2Minus>(
      MakeScalarInterfaceSipdgPenaltyReference<P2Minus>(eta));
}

template <bool P2Minus>
std::vector<Real> MakeVectorInterfaceSipdgReference(
   const Real mu_minus,
   const Real mu_plus,
   const Real eta)
{
   return LiftScalarInterfaceReferenceToVector<P2Minus>(
      MakeScalarInterfaceSipdgReference<P2Minus>(
         mu_minus,
         mu_plus,
         eta));
}

std::vector<Real> MakeSameSpaceP2InteriorSipdgSymmetryReference(
   const Real mu_minus,
   const Real mu_plus)
{
   const std::vector<Real> jump_trace{0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
   const std::vector<Real> minus_normal_gradient{
      0.0, 0.0, 0.0, 6.0, -8.0, 2.0};
   const std::vector<Real> plus_normal_gradient{
      -2.0, 8.0, -6.0, 0.0, 0.0, 0.0};

   std::vector<Real> average_flux(jump_trace.size(), 0.0);
   for (size_t i = 0; i < average_flux.size(); ++i)
   {
      average_flux[i] =
         0.5 * (
            mu_minus * minus_normal_gradient[i] +
            mu_plus * plus_normal_gradient[i]);
   }
   return MakeOuterProductReference(average_flux, jump_trace);
}

std::vector<Real> MakeSameSpaceP2InteriorSipdgPenaltyReference(const Real eta)
{
   const std::vector<Real> jump_trace{0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
   return ScaleDenseReference(MakeRankOneReference(jump_trace), eta);
}

std::vector<Real> MakeSameSpaceP2InteriorSipdgReference(
   const Real mu_minus,
   const Real mu_plus,
   const Real eta)
{
   auto ref =
      MakeSameSpaceP2InteriorSipdgConsistencyReference(mu_minus, mu_plus);
   const auto symmetry =
      MakeSameSpaceP2InteriorSipdgSymmetryReference(mu_minus, mu_plus);
   const auto penalty =
      MakeSameSpaceP2InteriorSipdgPenaltyReference(eta);
   for (size_t i = 0; i < ref.size(); ++i)
   {
      ref[i] += symmetry[i] + penalty[i];
   }
   return ref;
}

std::vector<Real> MakeSameSpaceP2VectorSipdgReference(
   const Real mu_minus,
   const Real mu_plus,
   const Real eta)
{
   return LiftScalarReferenceToComponentBlocks(
      MakeSameSpaceP2InteriorSipdgReference(mu_minus, mu_plus, eta));
}

bool CheckDenseClose(
   const char* label,
   const std::vector<Real>& dense,
   const std::vector<Real>& ref,
   const Real tol = 1.0e-12)
{
   GENDIL_VERIFY(dense.size() == ref.size(), "Dense matrix sizes do not match.");

   Real max_err = 0.0;
   for (size_t i = 0; i < dense.size(); ++i)
   {
      max_err = std::max(max_err, std::abs(dense[i] - ref[i]));
   }

   std::cout << label << " | dense max error = " << max_err << "\n";
   if (max_err > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      const size_t size =
         static_cast<size_t>(std::sqrt(static_cast<Real>(dense.size())));
      for (size_t row = 0; row < size; ++row)
      {
         for (size_t col = 0; col < size; ++col)
         {
            const size_t idx = row * size + col;
            std::cerr
               << "  (" << row << ", " << col << "): got "
               << dense[idx] << ", expected " << ref[idx] << "\n";
         }
      }
      return false;
   }
   return true;
}

template <typename VectorType>
void AddInto(VectorType& dst, const VectorType& src)
{
   GENDIL_VERIFY(dst.Size() == src.Size(), "Vector sizes do not match.");

   Real* dst_data = dst.ReadWriteHostData();
   const Real* src_data = src.ReadHostData();
   for (Integer i = 0; i < dst.Size(); ++i)
   {
      dst_data[i] += src_data[i];
   }
}

template <typename OperatorType, typename VectorType>
void ApplyAndAdd(const OperatorType& op, const VectorType& x, VectorType& dst)
{
   VectorType tmp(dst.Size());
   tmp = 0.0;
   op(x, tmp);
   AddInto(dst, tmp);
}

template <typename VectorType>
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");

   Real err_sq = 0.0;
   Real ref_sq = 0.0;
   const Real* a_data = a.ReadHostData();
   const Real* b_data = b.ReadHostData();
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real diff = a_data[i] - b_data[i];
      err_sq += diff * diff;
      ref_sq += b_data[i] * b_data[i];
   }

   const Real err = std::sqrt(err_sq);
   const Real ref = std::sqrt(ref_sq);
   return ref == 0.0 ? err : err / ref;
}

template <typename VectorType>
bool CheckClose(
   const char* label,
   const VectorType& a,
   const VectorType& b,
   Real tol = 1.0e-12)
{
   const Real rel = RelativeL2Error(a, b);
   std::cout << label << " | relative L2 error = " << rel << "\n";
   if (rel > tol)
   {
      std::cerr << "FAILED: " << label << "\n";
      return false;
   }
   return true;
}

template<class T, class = void>
struct has_minus_side : std::false_type {};

template<class T>
struct has_minus_side<
   T,
   std::void_t<decltype(std::declval<const T&>().MinusSide())>>
   : std::true_type {};

template<class T>
inline constexpr bool has_minus_side_v =
   has_minus_side<T>::value;

template<class T, class = void>
struct has_plus_side : std::false_type {};

template<class T>
struct has_plus_side<
   T,
   std::void_t<decltype(std::declval<const T&>().PlusSide())>>
   : std::true_type {};

template<class T>
inline constexpr bool has_plus_side_v =
   has_plus_side<T>::value;

template<class T, class = void>
struct has_get_plus_finite_element_space : std::false_type {};

template<class T>
struct has_get_plus_finite_element_space<
   T,
   std::void_t<decltype(std::declval<const T&>().GetPlusFiniteElementSpace())>>
   : std::true_type {};

template<class T>
inline constexpr bool has_get_plus_finite_element_space_v =
   has_get_plus_finite_element_space<T>::value;

auto MakeReferenceData()
{
   constexpr Integer order0 = 1;
   constexpr Integer order1 = 2;
   constexpr GlobalIndex num_cells = 3;
   const Real h = 1.0 / num_cells;

   Cartesian1DMesh mesh(h, num_cells);

   auto fe0 = MakeLobattoFiniteElement(FiniteElementOrders<order0>{});
   auto fes0_unshifted = MakeFiniteElementSpace(mesh, fe0);
   const Integer ndofs0 = fes0_unshifted.GetNumberOfFiniteElementDofs();
   auto fes0 = MakeFiniteElementSpace(mesh, fe0, L2Restriction{0});

   auto fe1 = MakeLobattoFiniteElement(FiniteElementOrders<order1>{});
   auto fes1 = MakeFiniteElementSpace(mesh, fe1, L2Restriction{ndofs0});

   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   auto boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});

   using InteriorFaceMesh0 =
      std::tuple_element_t<0, std::remove_cvref_t<decltype(interior_faces)>>;
   using BoundaryFaceMesh0 =
      std::tuple_element_t<0, std::remove_cvref_t<decltype(boundary_faces)>>;
   using IntermeshFaceMesh =
      CartesianIntermeshFaceConnectivity<1, 0>;
   using NonconformingIntermeshFaceMesh =
      NonconformingCartesianIntermeshFaceConnectivity<1, 0>;
   static_assert(global_face_mesh_has_static_face_family_v<InteriorFaceMesh0>);
   static_assert(global_face_mesh_has_static_face_family_v<BoundaryFaceMesh0>);
   static_assert(global_face_mesh_has_static_face_family_v<IntermeshFaceMesh>);
   static_assert(
      global_face_mesh_has_static_face_family_v<
         NonconformingIntermeshFaceMesh>);
   static_assert(global_face_mesh_minus_local_face_index_v<InteriorFaceMesh0> == 0);
   static_assert(global_face_mesh_plus_local_face_index_v<InteriorFaceMesh0> == 1);
   static_assert(global_face_mesh_minus_local_face_index_v<IntermeshFaceMesh> == 0);
   static_assert(global_face_mesh_plus_local_face_index_v<IntermeshFaceMesh> == 1);

   auto interior_fes0 =
      MakeGlobalInteriorFaceFiniteElementSpace(fes0, interior_faces);
   auto interior_fes1 =
      MakeGlobalInteriorFaceFiniteElementSpace(fes1, interior_faces);
   auto cross_interior_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(fes0, fes1, interior_faces);
   auto boundary_fes0 =
      MakeGlobalBoundaryFaceFiniteElementSpace(fes0, boundary_faces);
   auto boundary_fes1 =
      MakeGlobalBoundaryFaceFiniteElementSpace(fes1, boundary_faces);

   return std::tuple{
      mesh,
      fes0,
      fes1,
      interior_fes0,
      interior_fes1,
      cross_interior_fes,
      boundary_fes0,
      boundary_fes1};
}

bool TestAggregation()
{
   auto data = MakeReferenceData();
   const auto& fes0 = std::get<1>(data);
   const auto& fes1 = std::get<2>(data);
   const auto& interior_fes0 = std::get<3>(data);
   const auto& cross_interior_fes = std::get<5>(data);
   const auto& boundary_fes0 = std::get<6>(data);

   auto singleton = MakeMixedFiniteElementSpace(fes0);
   static_assert(decltype(singleton)::num_cell_spaces == 1);
   static_assert(decltype(singleton)::num_interior_face_spaces == 0);
   static_assert(decltype(singleton)::num_boundary_face_spaces == 0);

   auto homogeneous_domain = MakeIntegrationDomain<"solid">(fes0);
   using HomogeneousDomainEntry = decltype(homogeneous_domain);
   static_assert(
      is_integration_domain_key_v<typename HomogeneousDomainEntry::key_type>);
   static_assert(
      is_integration_domain_v<typename HomogeneousDomainEntry::value_type>);
   static_assert(
      std::is_same_v<
         typename HomogeneousDomainEntry::value_type,
         IntegrationDomain<std::remove_cvref_t<decltype(fes0)>>>);

   auto mixed = MakeMixedFiniteElementSpace(
      fes0,
      fes1,
      interior_fes0,
      cross_interior_fes,
      boundary_fes0);

   static_assert(decltype(mixed)::num_cell_spaces == 2);
   static_assert(decltype(mixed)::num_interior_face_spaces == 2);
   static_assert(decltype(mixed)::num_boundary_face_spaces == 2);
   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(mixed.GetCellFiniteElementSpace<0>())>,
         std::remove_cvref_t<decltype(fes0)>>);
   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(mixed.GetCellFiniteElementSpace<1>())>,
         std::remove_cvref_t<decltype(fes1)>>);
   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(mixed.GetInteriorFaceFiniteElementSpace<1>())>,
         std::remove_cvref_t<decltype(std::get<0>(cross_interior_fes))>>);
   static_assert(
      std::is_same_v<
         std::remove_cvref_t<decltype(mixed.GetBoundaryFaceFiniteElementSpace<0>())>,
         std::remove_cvref_t<decltype(std::get<0>(boundary_fes0))>>);
   static_assert(
      !is_tuple_v<
         decltype(mixed.GetInteriorFaceFiniteElementSpace<0>().GetFaceMesh())>);
   static_assert(
      !is_tuple_v<
         decltype(mixed.GetBoundaryFaceFiniteElementSpace<0>().GetFaceMesh())>);
   static_assert(
      std::remove_cvref_t<
         decltype(std::get<0>(interior_fes0))>::is_same_space_batch);
   static_assert(
      !std::remove_cvref_t<
         decltype(std::get<0>(cross_interior_fes))>::is_same_space_batch);

   auto mixed_domain = MakeIntegrationDomain<"solid">(mixed);
   using MixedDomainEntry = decltype(mixed_domain);
   static_assert(is_integration_domain_key_v<typename MixedDomainEntry::key_type>);
   static_assert(is_integration_domain_v<typename MixedDomainEntry::value_type>);
   static_assert(
      std::is_same_v<
         typename MixedDomainEntry::value_type,
         IntegrationDomain<std::remove_cvref_t<decltype(mixed)>>>);

   auto homogeneous_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fes0),
      MakeIntegrationDomain<"solid">(fes0));
   using HomogeneousContext = decltype(homogeneous_ctx);
   static_assert(HomogeneousContext::template has_domain<"solid">());
   static_assert(!HomogeneousContext::template has_interior_face_domain<"solid">());
   static_assert(!HomogeneousContext::template has_boundary_face_domain<"solid">());
   static_assert(
      is_cell_integration_domain_v<
         std::remove_cvref_t<decltype(homogeneous_ctx.template domain<"solid">())>>);

   auto mixed_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   using MixedContext = decltype(mixed_ctx);
   static_assert(MixedContext::template has_domain<"solid">());
   static_assert(MixedContext::template has_interior_face_domain<"solid">());
   static_assert(MixedContext::template has_boundary_face_domain<"solid">());
   static_assert(
      is_cell_integration_domain_v<
         std::remove_cvref_t<decltype(mixed_ctx.template domain<"solid">())>>);
   static_assert(
      is_interior_face_integration_domain_v<
         std::remove_cvref_t<
            decltype(mixed_ctx.template interior_face_domain<"solid">())>>);
   static_assert(
      is_boundary_face_integration_domain_v<
         std::remove_cvref_t<
            decltype(mixed_ctx.template boundary_face_domain<"solid">())>>);

   bool success = true;
   success = (singleton.GetNumberOfCellFiniteElementSpaces() == 1) && success;
   success = (singleton.GetNumberOfFiniteElementDofs() ==
      fes0.GetNumberOfFiniteElementDofs()) && success;
   success = (mixed.GetNumberOfCellFiniteElementSpaces() == 2) && success;
   success = (mixed.GetNumberOfInteriorFaceFiniteElementSpaces() == 2) && success;
   success = (mixed.GetNumberOfBoundaryFaceFiniteElementSpaces() == 2) && success;
   success =
      (mixed.GetNumberOfFiniteElementDofs() ==
       fes0.GetNumberOfFiniteElementDofs() +
          fes1.GetNumberOfFiniteElementDofs()) && success;
   success =
      (mixed.GetNumberOfFiniteElements() ==
       fes0.GetNumberOfFiniteElements() +
          fes1.GetNumberOfFiniteElements()) && success;
   success = (mixed.GetNumberOfInteriorFaces() == 4) && success;
   success = (mixed.GetNumberOfBoundaryFaces() == 2) && success;

   if (!success)
   {
      std::cerr << "FAILED: mixed finite element space aggregation\n";
   }
   else
   {
      std::cout << "PASS: mixed finite element space aggregation\n";
   }
   return success;
}

template <bool ExplicitTestField>
bool TestCellLayout(const char* label)
{
   auto data = MakeReferenceData();
   const auto& fes0 = std::get<1>(data);
   const auto& fes1 = std::get<2>(data);

   auto mixed = MakeMixedFiniteElementSpace(fes0, fes1);
   Vector x(mixed.GetNumberOfFiniteElementDofs());
   Vector y_mixed(mixed.GetNumberOfFiniteElementDofs());
   Vector y_ref(mixed.GetNumberOfFiniteElementDofs());
   FillPattern(x);
   y_mixed = 0.0;
   y_ref = 0.0;

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"solid"> cells;
   auto form = integrate(cells, u * v);
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

   if constexpr (ExplicitTestField)
   {
      auto mixed_ctx = MakeWeakFormContext(
         MakeTrialField<"u">(mixed),
         MakeTestField<"u">(mixed),
         MakeIntegrationDomain<"solid">(mixed));
      auto mixed_op =
         MakeGenericOperator<SerialKernelConfiguration>(
            form,
            mixed_ctx,
            integration_rule);
      mixed_op(x, y_mixed);
   }
   else
   {
      auto mixed_ctx = MakeWeakFormContext(
         MakeTrialField<"u">(mixed),
         MakeIntegrationDomain<"solid">(mixed));
      auto mixed_op =
         MakeGenericOperator<SerialKernelConfiguration>(
            form,
            mixed_ctx,
            integration_rule);
      mixed_op(x, y_mixed);
   }

   auto ctx0 = MakeWeakFormContext(
      MakeTrialField<"u">(fes0),
      MakeIntegrationDomain<"solid">(fes0));
   auto ctx1 = MakeWeakFormContext(
      MakeTrialField<"u">(fes1),
      MakeIntegrationDomain<"solid">(fes1));

   auto op0 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx0, integration_rule);
   auto op1 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx1, integration_rule);

   ApplyAndAdd(op0, x, y_ref);
   ApplyAndAdd(op1, x, y_ref);

   return CheckClose(label, y_mixed, y_ref);
}

bool TestInteriorAndBoundaryFaces()
{
   auto data = MakeReferenceData();
   const auto& fes0 = std::get<1>(data);
   const auto& fes1 = std::get<2>(data);
   const auto& interior_fes0 = std::get<3>(data);
   const auto& interior_fes1 = std::get<4>(data);
   const auto& boundary_fes0 = std::get<6>(data);
   const auto& boundary_fes1 = std::get<7>(data);

   auto mixed = MakeMixedFiniteElementSpace(
      fes0,
      fes1,
      interior_fes0,
      interior_fes1,
      boundary_fes0,
      boundary_fes1);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   Vector y_mixed(mixed.GetNumberOfFiniteElementDofs());
   Vector y_ref(mixed.GetNumberOfFiniteElementDofs());
   FillPattern(x);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"solid"> interior_facets;
   BoundaryFacets<"solid"> boundary_facets;

   auto form =
      integrate(interior_facets, jump(u) * jump(v))
      + integrate(boundary_facets, u * v);
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

   auto mixed_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   auto mixed_op =
      MakeGenericOperator<SerialKernelConfiguration>(
         form,
         mixed_ctx,
         integration_rule);

   y_mixed = 0.0;
   mixed_op(x, y_mixed);

   y_ref = 0.0;
   auto singleton0 =
      MakeMixedFiniteElementSpace(fes0, interior_fes0, boundary_fes0);
   auto singleton1 =
      MakeMixedFiniteElementSpace(fes1, interior_fes1, boundary_fes1);
   auto ctx0 = MakeWeakFormContext(
      MakeTrialField<"u">(singleton0),
      MakeIntegrationDomain<"solid">(singleton0));
   auto ctx1 = MakeWeakFormContext(
      MakeTrialField<"u">(singleton1),
      MakeIntegrationDomain<"solid">(singleton1));

   auto op0 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx0, integration_rule);
   auto op1 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx1, integration_rule);
   ApplyAndAdd(op0, x, y_ref);
   ApplyAndAdd(op1, x, y_ref);

   return CheckClose("mixed same-space face dispatch", y_mixed, y_ref);
}

// TODO: restore named boundary-subdomain dispatch coverage when a parent-domain
// restricted boundary API such as MakeBoundaryDomain<"solid", "outer_wall">(...)
// exists. This cleanup keeps BoundaryFacets<"solid"> full-domain coverage only.

bool TestUnusedCrossSpaceDomainIsNotSelected()
{
   auto data = MakeReferenceData();
   const auto& fes0 = std::get<1>(data);
   const auto& fes1 = std::get<2>(data);
   const auto& cross_interior_fes = std::get<5>(data);

   auto active_mixed = MakeMixedFiniteElementSpace(fes0, fes1);
   auto unused_mixed = MakeMixedFiniteElementSpace(
      fes0,
      fes1,
      cross_interior_fes);

   Vector x(active_mixed.GetNumberOfFiniteElementDofs());
   Vector y_mixed(active_mixed.GetNumberOfFiniteElementDofs());
   Vector y_ref(active_mixed.GetNumberOfFiniteElementDofs());
   FillPattern(x);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"solid"> cells;
   auto form = integrate(cells, u * v);
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

   auto mixed_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(active_mixed),
      MakeIntegrationDomain<"solid">(active_mixed),
      MakeIntegrationDomain<"unused">(unused_mixed));
   auto mixed_op =
      MakeGenericOperator<SerialKernelConfiguration>(
         form,
         mixed_ctx,
         integration_rule);

   y_mixed = 0.0;
   mixed_op(x, y_mixed);

   y_ref = 0.0;
   auto ctx0 = MakeWeakFormContext(
      MakeTrialField<"u">(fes0),
      MakeIntegrationDomain<"solid">(fes0));
   auto ctx1 = MakeWeakFormContext(
      MakeTrialField<"u">(fes1),
      MakeIntegrationDomain<"solid">(fes1));

   auto op0 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx0, integration_rule);
   auto op1 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx1, integration_rule);
   ApplyAndAdd(op0, x, y_ref);
   ApplyAndAdd(op1, x, y_ref);

   return CheckClose(
      "mixed unused cross-space domain is not selected",
      y_mixed,
      y_ref);
}

bool TestRestrictedFaceContextsKeepFaceFieldBindings()
{
   auto data = MakeReferenceData();
   const auto& fes0 = std::get<1>(data);
   const auto& fes1 = std::get<2>(data);
   const auto& interior_fes0 = std::get<3>(data);
   const auto& interior_fes1 = std::get<4>(data);
   const auto& cross_interior_fes = std::get<5>(data);
   const auto& boundary_fes0 = std::get<6>(data);
   const auto& boundary_fes1 = std::get<7>(data);

   auto mixed = MakeMixedFiniteElementSpace(
      fes0,
      fes1,
      interior_fes0,
      interior_fes1,
      boundary_fes0,
      boundary_fes1);

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));

   const auto& interior_fes1_0 = std::get<0>(interior_fes1);
   const auto& cross_interior_fes_0 = std::get<0>(cross_interior_fes);
   const auto& boundary_fes1_0 = std::get<0>(boundary_fes1);

   using InteriorFaceSpace1 = std::remove_cvref_t<decltype(interior_fes1_0)>;
   using CrossInteriorFaceSpace =
      std::remove_cvref_t<decltype(cross_interior_fes_0)>;
   using BoundaryFaceSpace1 = std::remove_cvref_t<decltype(boundary_fes1_0)>;
   using CellSpace0 = std::remove_cvref_t<decltype(fes0)>;

   static_assert(!is_face_finite_element_space_v<CellSpace0>);
   static_assert(is_tuple_v<decltype(interior_fes1)>);
   static_assert(is_tuple_v<decltype(boundary_fes1)>);
   static_assert(!is_tuple_v<decltype(interior_fes1_0.GetFaceMesh())>);
   static_assert(!is_tuple_v<decltype(boundary_fes1_0.GetFaceMesh())>);
   static_assert(is_interior_face_finite_element_space_v<InteriorFaceSpace1>);
   static_assert(is_face_finite_element_space_v<InteriorFaceSpace1>);
   static_assert(is_same_space_interior_face_finite_element_space_v<InteriorFaceSpace1>);
   static_assert(!is_two_space_interior_face_finite_element_space_v<InteriorFaceSpace1>);
   static_assert(!requires_two_sided_face_qdata_v<InteriorFaceSpace1>);
   static_assert(supports_one_sided_face_qdata_v<InteriorFaceSpace1>);

   static_assert(is_interior_face_finite_element_space_v<CrossInteriorFaceSpace>);
   static_assert(is_face_finite_element_space_v<CrossInteriorFaceSpace>);
   static_assert(!is_same_space_interior_face_finite_element_space_v<CrossInteriorFaceSpace>);
   static_assert(is_two_space_interior_face_finite_element_space_v<CrossInteriorFaceSpace>);
   static_assert(requires_two_sided_face_qdata_v<CrossInteriorFaceSpace>);
   static_assert(!supports_one_sided_face_qdata_v<CrossInteriorFaceSpace>);

   static_assert(is_boundary_face_finite_element_space_v<BoundaryFaceSpace1>);
   static_assert(is_face_finite_element_space_v<BoundaryFaceSpace1>);
   static_assert(!requires_two_sided_face_qdata_v<BoundaryFaceSpace1>);
   static_assert(supports_one_sided_face_qdata_v<BoundaryFaceSpace1>);

   auto volume_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fes0),
      MakeIntegrationDomain<"solid">(fes0));
   CellExecutionBatch<"solid", 0, CellSpace0> cell_batch{ fes0 };
   auto local_facet_ctx =
      MakeRestrictedWeakFormContext<"u", "u">(
         volume_ctx,
         InteriorFacets<"solid">{},
         cell_batch);
   static_assert(std::is_same_v<
      std::remove_cvref_t<
         decltype(local_facet_ctx.template fe_field<"u">().space)>,
      CellSpace0>);

   static_assert(requires(const InteriorFaceSpace1& space) {
      space.GetMinusFiniteElementSpace();
      space.GetPlusFiniteElementSpace();
   });

   InteriorFaceExecutionBatch<"solid", 1, InteriorFaceSpace1>
      interior_batch{ interior_fes1_0 };
   auto interior_ctx =
      MakeRestrictedWeakFormContext<"u", "u">(
         ctx,
         InteriorFacets<"solid">{},
         interior_batch);
   static_assert(std::is_same_v<
      std::remove_cvref_t<
         decltype(interior_ctx.template fe_field<"u">().space)>,
      InteriorFaceSpace1>);

   static_assert(requires(const BoundaryFaceSpace1& space) {
      space.GetMinusFiniteElementSpace();
   });
   static_assert(!has_get_plus_finite_element_space_v<BoundaryFaceSpace1>);

   BoundaryFaceExecutionBatch<"solid", 2, BoundaryFaceSpace1>
      boundary_batch{ boundary_fes1_0 };
   auto boundary_ctx =
      MakeRestrictedWeakFormContext<"u", "u">(
         ctx,
         BoundaryFacets<"solid">{},
         boundary_batch);
   static_assert(std::is_same_v<
      std::remove_cvref_t<
         decltype(boundary_ctx.template fe_field<"u">().space)>,
      BoundaryFaceSpace1>);

   using IntegrationRule =
      std::remove_cvref_t<decltype(MakeIntegrationRule(IntegrationRuleNumPoints<5>{}))>;

   auto local_volume_qd =
      MakeFiniteElementFacetQuadData<IntegrationRule>(fes0);
   using LocalVolumeQD = std::remove_cvref_t<decltype(local_volume_qd)>;
   static_assert(requires(const LocalVolumeQD& qd) {
      qd.MinusSide();
      qd.PlusSide();
   });
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(local_volume_qd.MinusSide())>,
      std::remove_cvref_t<decltype(local_volume_qd.PlusSide())>>);

   auto same_space_qd =
      MakeGlobalFacetFiniteElementQuadData<IntegrationRule>(interior_fes1_0);
   using SameSpaceQD = std::remove_cvref_t<decltype(same_space_qd)>;
   static_assert(requires(const SameSpaceQD& qd) {
      qd.MinusSide();
      qd.PlusSide();
   });

   auto boundary_qd =
      MakeGlobalFacetFiniteElementQuadData<IntegrationRule>(boundary_fes1_0);
   using BoundaryQD = std::remove_cvref_t<decltype(boundary_qd)>;
   static_assert(requires(const BoundaryQD& qd) {
      qd.MinusSide();
   });
   static_assert(!has_plus_side_v<BoundaryQD>);

   auto two_space_qd =
      MakeGlobalFacetFiniteElementQuadData<IntegrationRule>(cross_interior_fes_0);
   using TwoSpaceQD = std::remove_cvref_t<decltype(two_space_qd)>;
   static_assert(requires(const TwoSpaceQD& qd) {
      qd.MinusSide();
      qd.PlusSide();
   });
   static_assert(!std::is_same_v<
      std::remove_cvref_t<decltype(two_space_qd.MinusSide())>,
      std::remove_cvref_t<decltype(two_space_qd.PlusSide())>>);

   struct TestFaceSide
   {
      GlobalIndex cell_index;
      constexpr GlobalIndex GetCellIndex() const { return cell_index; }
   };

   struct TestFaceInfo
   {
      TestFaceSide minus_side;
      TestFaceSide plus_side;
      constexpr TestFaceSide MinusSide() const { return minus_side; }
      constexpr TestFaceSide PlusSide() const { return plus_side; }
   };

   TestFaceInfo face_info{ TestFaceSide{0}, TestFaceSide{1} };
   auto interior_minus_binding =
      MakeMinusFacetFieldBinding(face_info, interior_fes1_0, same_space_qd);
   auto interior_plus_binding =
      MakePlusFacetFieldBinding(face_info, interior_fes1_0, same_space_qd);
   auto boundary_minus_binding =
      MakeMinusFacetFieldBinding(face_info, boundary_fes1_0, boundary_qd);

   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(interior_minus_binding.volume_space)>,
      std::remove_cvref_t<decltype(interior_fes1_0.GetMinusFiniteElementSpace())>>);
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(interior_plus_binding.volume_space)>,
      std::remove_cvref_t<decltype(interior_fes1_0.GetPlusFiniteElementSpace())>>);
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(boundary_minus_binding.volume_space)>,
      std::remove_cvref_t<decltype(boundary_fes1_0.GetMinusFiniteElementSpace())>>);
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(interior_minus_binding.qdata)>,
      std::remove_cvref_t<decltype(same_space_qd.MinusSide())>>);
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(interior_plus_binding.qdata)>,
      std::remove_cvref_t<decltype(same_space_qd.PlusSide())>>);
   static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(boundary_minus_binding.qdata)>,
      std::remove_cvref_t<decltype(boundary_qd.MinusSide())>>);

   return true;
}

bool TestDuplicateSameTypeBoundaryUsesDescriptorCellIndex()
{
   constexpr Integer order = 1;
   constexpr GlobalIndex num_cells = 3;
   const Real h = 1.0 / num_cells;

   Cartesian1DMesh mesh(h, num_cells);
   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<order>{});
   auto fes_unshifted = MakeFiniteElementSpace(mesh, fe);
   const Integer ndofs = fes_unshifted.GetNumberOfFiniteElementDofs();
   auto fes0 = MakeFiniteElementSpace(mesh, fe, L2Restriction{0});
   auto fes1 = MakeFiniteElementSpace(mesh, fe, L2Restriction{ndofs});

   static_assert(std::is_same_v<decltype(fes0), decltype(fes1)>);

   auto boundary_faces =
      MakeCartesianBoundaryFaceConnectivity<1>({num_cells});
   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   auto boundary_fes0 =
      MakeGlobalBoundaryFaceFiniteElementSpace(fes0, boundary_faces);
   auto boundary_fes1 =
      MakeGlobalBoundaryFaceFiniteElementSpace(fes1, boundary_faces);
   auto two_space_same_type_interior_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(fes0, fes1, interior_faces);
   const auto& two_space_same_type_interior_fes_0 =
      std::get<0>(two_space_same_type_interior_fes);
   using TwoSpaceSameTypeInteriorFaceSpace =
      std::remove_cvref_t<decltype(two_space_same_type_interior_fes_0)>;
   static_assert(is_tuple_v<decltype(two_space_same_type_interior_fes)>);
   static_assert(
      is_two_space_interior_face_finite_element_space_v<
         TwoSpaceSameTypeInteriorFaceSpace>);
   static_assert(
      !is_same_space_interior_face_finite_element_space_v<
         TwoSpaceSameTypeInteriorFaceSpace>);
   static_assert(
      requires_two_sided_face_qdata_v<TwoSpaceSameTypeInteriorFaceSpace>);

   using IntegrationRule =
      std::remove_cvref_t<decltype(MakeIntegrationRule(IntegrationRuleNumPoints<5>{}))>;
   auto same_type_two_space_qd =
      MakeGlobalFacetFiniteElementQuadData<IntegrationRule>(
         two_space_same_type_interior_fes_0);
   static_assert(!std::is_same_v<
      std::remove_cvref_t<decltype(same_type_two_space_qd.MinusSide())>,
      std::remove_cvref_t<decltype(same_type_two_space_qd.PlusSide())>>);

   auto mixed = MakeMixedFiniteElementSpace(
      fes0,
      fes1,
      boundary_fes0,
      boundary_fes1);

   Vector x(mixed.GetNumberOfFiniteElementDofs());
   Vector y_mixed(mixed.GetNumberOfFiniteElementDofs());
   Vector y_ref(mixed.GetNumberOfFiniteElementDofs());
   FillPattern(x);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   BoundaryFacets<"solid"> boundary_facets;
   auto form = integrate(boundary_facets, u * v);
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

   auto mixed_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   auto mixed_op =
      MakeGenericOperator<SerialKernelConfiguration>(
         form,
         mixed_ctx,
         integration_rule);

   y_mixed = 0.0;
   mixed_op(x, y_mixed);

   auto singleton0 =
      MakeMixedFiniteElementSpace(fes0, boundary_fes0);
   auto singleton1 =
      MakeMixedFiniteElementSpace(fes1, boundary_fes1);
   auto ctx0 = MakeWeakFormContext(
      MakeTrialField<"u">(singleton0),
      MakeIntegrationDomain<"solid">(singleton0));
   auto ctx1 = MakeWeakFormContext(
      MakeTrialField<"u">(singleton1),
      MakeIntegrationDomain<"solid">(singleton1));

   auto op0 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx0, integration_rule);
   auto op1 =
      MakeGenericOperator<SerialKernelConfiguration>(form, ctx1, integration_rule);

   y_ref = 0.0;
   ApplyAndAdd(op0, x, y_ref);
   ApplyAndAdd(op1, x, y_ref);

   return CheckClose(
      "mixed duplicate same-type boundary dispatch uses descriptor FaceI",
      y_mixed,
      y_ref);
}

template <class KernelPolicy, class FiniteElementSpace, class FaceSpace, class Form>
bool CheckSameSpaceCanonicalGlobalInteriorDispatch(
   const char* label,
   const FiniteElementSpace& fe_space,
   const FaceSpace& interior_face_fes,
   const Form& form)
{
   auto singleton_global_fes =
      MakeMixedFiniteElementSpace(fe_space, interior_face_fes);

   constexpr auto TrialName = requirements<Form>::trial_name;
   constexpr auto TestName = requirements<Form>::test_name;
   InteriorFacets<"solid"> interior_facets;
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});

	   auto ctx = MakeWeakFormContext(
	      MakeTrialField<TrialName>(singleton_global_fes),
	      MakeIntegrationDomain<"solid">(singleton_global_fes));
	   auto public_op =
	      MakeGenericOperator<KernelPolicy>(
	         form,
	         ctx,
	         integration_rule);

   Vector x(fe_space.GetNumberOfFiniteElementDofs());
   FillPattern(x);
	   Vector y_public(x.Size());
	   Vector y_canonical(x.Size());
	   y_public = 0.0;
	   y_canonical = 0.0;

	   public_op(x, y_public);

   using SelectedFaceSpace = std::remove_cvref_t<decltype(interior_face_fes)>;
   InteriorFaceExecutionBatch<"solid", 0, SelectedFaceSpace>
      batch{ interior_face_fes };
   auto restricted_ctx =
      MakeRestrictedWeakFormContext<TrialName, TestName>(
         ctx,
         interior_facets,
         batch);

   auto dofs_in = MakeReadOnlyElementTensorView<KernelPolicy>(
      fe_space,
      x);
   auto dofs_out = MakeReadWriteElementTensorView<KernelPolicy>(
      fe_space,
      y_canonical);

   GenericCanonicalGlobalInteriorFacetDomainOperator<KernelPolicy>(
      restricted_ctx,
      interior_face_fes,
      form,
      integration_rule,
      dofs_in,
      dofs_in,
      dofs_out,
      dofs_out);

	   return CheckClose(label, y_canonical, y_public);
}

template <class KernelPolicy>
bool TestSameSpaceCanonicalGlobalInteriorParity()
{
   bool success = true;
   constexpr GlobalIndex num_cells = 2;
   Cartesian1DMesh mesh(0.5, num_cells);

   auto scalar_fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   auto scalar_fes = MakeFiniteElementSpace(mesh, scalar_fe);
   auto interior_faces =
      MakeCartesianInteriorFaceConnectivity<1>({num_cells});
   auto scalar_face_fes_tuple =
      MakeGlobalInteriorFaceFiniteElementSpace(scalar_fes, interior_faces);
   const auto& scalar_face_fes = std::get<0>(scalar_face_fes_tuple);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"solid"> interior_facets;
	   auto scalar_jump_form = integrate(interior_facets, jump(u) * jump(v));
	   success =
	      CheckSameSpaceCanonicalGlobalInteriorDispatch<KernelPolicy>(
	         "same-space scalar jump public dispatch uses canonical path",
	         scalar_fes,
	         scalar_face_fes,
	         scalar_jump_form) &&
      success;

   auto beta_plus =
      MakeVectorCoefficient<"beta", PhysicalCoordinate>(
         [] (const auto&) -> std::array<Real, 1>
         {
            return { 1.0 };
         });
	   auto scalar_upwind_form =
	      integrate(interior_facets, upwind(average(beta_plus), u) * jump(v));
	   success =
	      CheckSameSpaceCanonicalGlobalInteriorDispatch<KernelPolicy>(
	         "same-space scalar upwind public dispatch uses canonical path",
	         scalar_fes,
	         scalar_face_fes,
	         scalar_upwind_form) &&
      success;

   auto sipdg_mu_const =
      MakeCoefficient<"sipdg_mu_const">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return 1.75;
         });
   auto sipdg_eta_const =
      MakeCoefficient<"sipdg_eta_const">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return 4.25;
         });
   auto scalar_sipdg_form =
      integrate(
         interior_facets,
         - average(sipdg_mu_const * dot(grad(u), Normal{})) * jump(v)
	         - average(sipdg_mu_const * dot(grad(v), Normal{})) * jump(u)
	         + sipdg_eta_const * jump(u) * jump(v));
	   success =
	      CheckSameSpaceCanonicalGlobalInteriorDispatch<KernelPolicy>(
	         "same-space scalar SIPDG public dispatch uses canonical path",
	         scalar_fes,
	         scalar_face_fes,
	         scalar_sipdg_form) &&
      success;

   Vector mu_h(scalar_fes.GetNumberOfFiniteElementDofs());
   mu_h = 0.0;
   mu_h.WriteHostData()[2] = 2.0;
   mu_h.WriteHostData()[3] = 5.0;
   auto mu_view =
      MakeReadOnlyElementTensorView<KernelPolicy>(
         scalar_fes,
         mu_h);
   auto mu =
      MakeCoefficient<"mu", FieldValue<"mu">>(
         [] GENDIL_HOST_DEVICE (const Real mu_q) -> Real
         {
            return mu_q;
         });
   auto singleton_global_fes =
      MakeMixedFiniteElementSpace(scalar_fes, scalar_face_fes);
   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});
   auto ctx_mu =
      MakeWeakFormContext(
         MakeTrialField<"u">(singleton_global_fes),
         MakeFiniteElementField<"mu">(singleton_global_fes, mu_view),
         MakeIntegrationDomain<"solid">(singleton_global_fes));
   const Integer scalar_size = scalar_fes.GetNumberOfFiniteElementDofs();
   const std::vector<Real> scalar_jump_trace{0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
   const auto scalar_jump_ref = MakeRankOneReference(scalar_jump_trace);
   // For this Cartesian same-space face batch, the canonical minus trace is the
   // right cell's left trace and the canonical plus trace is the left cell's
   // right trace.
   constexpr Real mu_minus_trace = 5.0;
   constexpr Real mu_plus_trace = 2.0;

   auto minus_mu_form =
      integrate(interior_facets, minus(mu) * jump(u) * jump(v));
   auto plus_mu_form =
      integrate(interior_facets, plus(mu) * jump(u) * jump(v));
   auto jump_mu_form =
      integrate(interior_facets, jump(mu) * jump(u) * jump(v));
   auto average_mu_form =
      integrate(interior_facets, average(mu) * jump(u) * jump(v));
   success =
      CheckDenseClose(
         "same-space minus(mu) discontinuous trace dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               minus_mu_form,
               ctx_mu,
               integration_rule),
            scalar_size),
         ScaleDenseReference(scalar_jump_ref, mu_minus_trace)) &&
      success;
   success =
      CheckDenseClose(
         "same-space plus(mu) discontinuous trace dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               plus_mu_form,
               ctx_mu,
               integration_rule),
            scalar_size),
         ScaleDenseReference(scalar_jump_ref, mu_plus_trace)) &&
      success;
   success =
      CheckDenseClose(
         "same-space jump(mu) equals minus(mu)-plus(mu) dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               jump_mu_form,
               ctx_mu,
               integration_rule),
            scalar_size),
         ScaleDenseReference(scalar_jump_ref, mu_minus_trace - mu_plus_trace)) &&
      success;
   success =
      CheckDenseClose(
         "same-space average(mu) equals half minus-plus dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               average_mu_form,
               ctx_mu,
               integration_rule),
            scalar_size),
         ScaleDenseReference(
            scalar_jump_ref,
            Real(0.5) * (mu_minus_trace + mu_plus_trace))) &&
      success;

   auto sipdg_mu_consistency_form =
      integrate(
         interior_facets,
         - average(mu * dot(grad(u), Normal{})) * jump(v));
   success =
      CheckDenseClose(
         "same-space discontinuous mu SIPDG consistency dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               sipdg_mu_consistency_form,
               ctx_mu,
               integration_rule),
            scalar_size),
         MakeSameSpaceP2InteriorSipdgConsistencyReference(
            mu_minus_trace,
            mu_plus_trace)) &&
      success;

   auto vector_fe =
      MakeVectorFiniteElement(scalar_fe, scalar_fe);
   auto vector_fes = MakeFiniteElementSpace(mesh, vector_fe);
   auto vector_face_fes_tuple =
      MakeGlobalInteriorFaceFiniteElementSpace(vector_fes, interior_faces);
   const auto& vector_face_fes = std::get<0>(vector_face_fes_tuple);

   VectorTrialSpace<"U"> U;
   VectorTestSpace<"U"> V;
	   auto vector_jump_form =
	      integrate(interior_facets, dot(jump(U), jump(V)));
	   success =
	      CheckSameSpaceCanonicalGlobalInteriorDispatch<KernelPolicy>(
	         "same-space vector jump public dispatch uses canonical path",
	         vector_fes,
	         vector_face_fes,
	         vector_jump_form) &&
      success;

	   auto vector_upwind_form =
	      integrate(interior_facets, dot(upwind(average(beta_plus), U), jump(V)));
	   success =
	      CheckSameSpaceCanonicalGlobalInteriorDispatch<KernelPolicy>(
	         "same-space vector upwind public dispatch uses canonical path",
	         vector_fes,
	         vector_face_fes,
	         vector_upwind_form) &&
      success;

   auto vector_singleton_global_fes =
      MakeMixedFiniteElementSpace(vector_fes, vector_face_fes);
   auto vector_ctx = MakeWeakFormContext(
      MakeTrialField<"U">(vector_singleton_global_fes),
      MakeIntegrationDomain<"solid">(vector_singleton_global_fes));
   auto local_vector_ctx = MakeWeakFormContext(
      MakeTrialField<"U">(vector_fes),
      MakeIntegrationDomain<"solid">(vector_fes));
   constexpr Real vector_sipdg_mu = 1.75;
   constexpr Real vector_sipdg_eta = 4.25;
   auto vector_sipdg_mu_const =
      MakeCoefficient<"vector_sipdg_mu_const">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return vector_sipdg_mu;
         });
   auto vector_sipdg_eta_const =
      MakeCoefficient<"vector_sipdg_eta_const">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return vector_sipdg_eta;
         });
   const Integer vector_size = vector_fes.GetNumberOfFiniteElementDofs();
   auto check_local_and_global_vector_sipdg =
      [&] (
         const char* local_label,
         const char* global_label,
         const char* parity_label,
         const auto& form,
         const std::vector<Real>& reference)
      {
         auto local_dense =
            BuildDenseMatrix(
               MakeGenericOperator<KernelPolicy>(
                  form,
                  local_vector_ctx,
                  integration_rule),
               vector_size);
         auto global_dense =
            BuildDenseMatrix(
               MakeGenericOperator<KernelPolicy>(
                  form,
                  vector_ctx,
                  integration_rule),
               vector_size);

         bool piece_success =
            CheckDenseClose(
               local_label,
               local_dense,
               reference) &&
            CheckDenseClose(
               global_label,
               global_dense,
               reference) &&
            CheckDenseClose(
               parity_label,
               local_dense,
               global_dense);
         success = piece_success && success;
      };

   auto vector_sipdg_consistency_form =
      integrate(
         interior_facets,
         - dot(
            average(vector_sipdg_mu_const * (grad(U) * Normal{})),
            jump(V)));
   check_local_and_global_vector_sipdg(
      "local same-space componentwise vector SIPDG consistency dense reference",
      "canonical global same-space componentwise vector SIPDG consistency dense reference",
      "local/global same-space componentwise vector SIPDG consistency parity",
      vector_sipdg_consistency_form,
      LiftScalarReferenceToComponentBlocks(
         MakeSameSpaceP2InteriorSipdgConsistencyReference(
            vector_sipdg_mu,
            vector_sipdg_mu)));

   auto vector_sipdg_symmetry_form =
      integrate(
         interior_facets,
         - dot(
            average(vector_sipdg_mu_const * (grad(V) * Normal{})),
            jump(U)));
   check_local_and_global_vector_sipdg(
      "local same-space componentwise vector SIPDG symmetry dense reference",
      "canonical global same-space componentwise vector SIPDG symmetry dense reference",
      "local/global same-space componentwise vector SIPDG symmetry parity",
      vector_sipdg_symmetry_form,
      LiftScalarReferenceToComponentBlocks(
         MakeSameSpaceP2InteriorSipdgSymmetryReference(
            vector_sipdg_mu,
            vector_sipdg_mu)));

   auto vector_sipdg_penalty_form =
      integrate(
         interior_facets,
         vector_sipdg_eta_const * dot(jump(U), jump(V)));
   check_local_and_global_vector_sipdg(
      "local same-space componentwise vector SIPDG penalty dense reference",
      "canonical global same-space componentwise vector SIPDG penalty dense reference",
      "local/global same-space componentwise vector SIPDG penalty parity",
      vector_sipdg_penalty_form,
      LiftScalarReferenceToComponentBlocks(
         MakeSameSpaceP2InteriorSipdgPenaltyReference(vector_sipdg_eta)));

   auto vector_sipdg_form =
      integrate(
         interior_facets,
         - dot(
            average(vector_sipdg_mu_const * (grad(U) * Normal{})),
            jump(V))
         - dot(
            average(vector_sipdg_mu_const * (grad(V) * Normal{})),
            jump(U))
         + vector_sipdg_eta_const * dot(jump(U), jump(V)));
   check_local_and_global_vector_sipdg(
      "local same-space componentwise vector SIPDG complete dense reference",
      "canonical global same-space componentwise vector SIPDG complete dense reference",
      "local/global same-space componentwise vector SIPDG complete parity",
      vector_sipdg_form,
      MakeSameSpaceP2VectorSipdgReference(
         vector_sipdg_mu,
         vector_sipdg_mu,
         vector_sipdg_eta));

   Vector vector_mu_h(scalar_fes.GetNumberOfFiniteElementDofs());
   vector_mu_h = 0.0;
   vector_mu_h.WriteHostData()[2] = 2.0;
   vector_mu_h.WriteHostData()[3] = 5.0;
   auto vector_mu_view =
      MakeReadOnlyElementTensorView<KernelPolicy>(
         scalar_fes,
         vector_mu_h);
   auto vector_mu =
      MakeCoefficient<"vector_mu", FieldValue<"vector_mu">>(
         [] GENDIL_HOST_DEVICE (const Real mu_q) -> Real
         {
            return mu_q;
         });
   auto scalar_singleton_global_fes =
      MakeMixedFiniteElementSpace(scalar_fes, scalar_face_fes);
   auto vector_mu_ctx = MakeWeakFormContext(
      MakeTrialField<"U">(vector_singleton_global_fes),
      MakeFiniteElementField<"vector_mu">(
         scalar_singleton_global_fes,
         vector_mu_view),
      MakeIntegrationDomain<"solid">(vector_singleton_global_fes));
   auto vector_mu_consistency_form =
      integrate(
         interior_facets,
         - dot(average(vector_mu * (grad(U) * Normal{})), jump(V)));
   success =
      CheckDenseClose(
         "same-space discontinuous mu componentwise vector SIPDG consistency dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_mu_consistency_form,
               vector_mu_ctx,
               integration_rule),
            vector_fes.GetNumberOfFiniteElementDofs()),
         LiftScalarReferenceToComponentBlocks(
            MakeSameSpaceP2InteriorSipdgConsistencyReference(
               mu_minus_trace,
               mu_plus_trace))) &&
      success;

   return success;
}

template <class KernelPolicy, bool P2Minus>
bool TestConformingPAdaptiveGlobalInteriorDenseReference(const char* label)
{
   constexpr GlobalIndex num_cells = 1;
   Cartesian1DMesh mesh_left(1.0, num_cells);
   Cartesian1DMesh mesh_right(1.0, num_cells);

   auto fe_p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe_p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});

   auto minus_fes_unshifted =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(mesh_left, fe_p2);
         }
         else
         {
            return MakeFiniteElementSpace(mesh_left, fe_p1);
         }
      }();
   const Integer ndofs_minus =
      minus_fes_unshifted.GetNumberOfFiniteElementDofs();

   auto minus_fes =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(
               mesh_left,
               fe_p2,
               L2Restriction{0});
         }
         else
         {
            return MakeFiniteElementSpace(
               mesh_left,
               fe_p1,
               L2Restriction{0});
         }
      }();
   auto plus_fes =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(
               mesh_right,
               fe_p1,
               L2Restriction{ndofs_minus});
         }
         else
         {
            return MakeFiniteElementSpace(
               mesh_right,
               fe_p2,
               L2Restriction{ndofs_minus});
         }
      }();

   using InterfaceConnectivity = CartesianIntermeshFaceConnectivity<1, 1>;
   static_assert(InterfaceConnectivity::minus_local_face_index == 1);
   static_assert(InterfaceConnectivity::plus_local_face_index == 0);
   static_assert(InterfaceConnectivity::axis == 0);
   static_assert(InterfaceConnectivity::sign == +1);

   InterfaceConnectivity interface_faces({num_cells}, {num_cells});
   auto face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(
         minus_fes,
         plus_fes,
         interface_faces);
   using FaceSpace =
      std::remove_cvref_t<decltype(face_fes)>;
   static_assert(is_interior_face_finite_element_space_v<FaceSpace>);
   static_assert(is_two_space_interior_face_finite_element_space_v<FaceSpace>);
   static_assert(!is_same_space_interior_face_finite_element_space_v<FaceSpace>);
   static_assert(requires_two_sided_face_qdata_v<FaceSpace>);

   auto mixed = MakeMixedFiniteElementSpace(
      minus_fes,
      plus_fes,
      face_fes);
   static_assert(decltype(mixed)::num_cell_spaces == 2);
   static_assert(decltype(mixed)::num_interior_face_spaces == 1);

   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});
   using IntegrationRule = std::remove_cvref_t<decltype(integration_rule)>;
   auto face_qd =
      MakeGlobalFacetFiniteElementQuadData<IntegrationRule>(face_fes);
   static_assert(requires(const decltype(face_qd)& qd) {
      qd.MinusSide();
      qd.PlusSide();
   });
   static_assert(!std::is_same_v<
      std::remove_cvref_t<decltype(face_qd.MinusSide())>,
      std::remove_cvref_t<decltype(face_qd.PlusSide())>>);

   auto ctx = MakeWeakFormContext(
      MakeTrialField<"u">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   InteriorFaceExecutionBatch<"solid", 0, FaceSpace>
      batch{ face_fes };
   auto restricted_ctx =
      MakeRestrictedWeakFormContext<"u", "u">(
         ctx,
         InteriorFacets<"solid">{},
         batch);
   auto facet_ctx =
      MakeFacetOperatorContext(
         restricted_ctx,
         integration_rule,
         face_fes);
   const auto& restricted_qd =
      facet_ctx.template finite_element_facet_quad_data<"u">();
   static_assert(!std::is_same_v<
      std::remove_cvref_t<decltype(restricted_qd.MinusSide())>,
      std::remove_cvref_t<decltype(restricted_qd.PlusSide())>>);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   InteriorFacets<"solid"> interior_facets;
   auto form = integrate(interior_facets, jump(u) * jump(v));
   using GlobalInteriorChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(form));
   static_assert(GlobalInteriorChannels::template contains<ValueMinusChannel>());
   static_assert(GlobalInteriorChannels::template contains<ValuePlusChannel>());
   static_assert(!GlobalInteriorChannels::template contains<ValueChannel>());
   static_assert(
      !global_interior_channels_require_plus_side_jacobian_v<
         GlobalInteriorChannels>);
   static_assert(!local_interior_context_requires_plus_side_jacobian_v<
      decltype(form)>);
   static_assert(!global_interior_context_requires_plus_side_jacobian_v<
      decltype(form),
      GlobalInteriorChannels>);

   auto minus_test_form = integrate(interior_facets, minus(v));
   using MinusTestChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         minus_test_form));
   static_assert(MinusTestChannels::template contains<ValueMinusChannel>());
   static_assert(!MinusTestChannels::template contains<ValuePlusChannel>());
   static_assert(!has_unqualified_interior_test_trace_v<
      decltype(minus(v))>);

   auto plus_test_form = integrate(interior_facets, plus(v));
   using PlusTestChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         plus_test_form));
   static_assert(PlusTestChannels::template contains<ValuePlusChannel>());
   static_assert(!PlusTestChannels::template contains<ValueMinusChannel>());
   static_assert(!has_unqualified_interior_test_trace_v<
      decltype(plus(v))>);

   auto minus_trial_form = integrate(interior_facets, minus(u) * jump(v));
   using MinusTrialChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         minus_trial_form));
   static_assert(MinusTrialChannels::template contains<ValueMinusChannel>());
   static_assert(MinusTrialChannels::template contains<ValuePlusChannel>());

   auto plus_trial_form = integrate(interior_facets, plus(u) * jump(v));
   using PlusTrialChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         plus_trial_form));
   static_assert(PlusTrialChannels::template contains<ValueMinusChannel>());
   static_assert(PlusTrialChannels::template contains<ValuePlusChannel>());

   auto unqualified_test_gradient = dot(grad(v), Normal{});
   static_assert(
      has_unqualified_interior_test_trace_v<
         decltype(unqualified_test_gradient)>);
   auto traced_test_gradient = dot(average(grad(v)), Normal{});
   static_assert(
      !has_unqualified_interior_test_trace_v<
         decltype(traced_test_gradient)>);
   auto traced_test_gradient_form =
      integrate(interior_facets, traced_test_gradient);
   using TracedTestGradientChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         traced_test_gradient_form));
   static_assert(
      TracedTestGradientChannels::template contains<GradientPlusChannel>());
   static_assert(
      global_interior_channels_require_plus_side_jacobian_v<
         TracedTestGradientChannels>);
   static_assert(!local_interior_context_requires_plus_side_jacobian_v<
      decltype(traced_test_gradient_form)>);
   static_assert(
      global_interior_context_requires_plus_side_jacobian_v<
         decltype(traced_test_gradient_form),
         TracedTestGradientChannels>);

   auto minus_test_gradient_form =
      integrate(interior_facets, dot(minus(grad(v)), Normal{}));
   using MinusTestGradientChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         minus_test_gradient_form));
   static_assert(
      MinusTestGradientChannels::template contains<GradientMinusChannel>());
   static_assert(
      !MinusTestGradientChannels::template contains<GradientPlusChannel>());
   static_assert(
      !global_interior_channels_require_plus_side_jacobian_v<
         MinusTestGradientChannels>);

   auto plus_test_gradient_form =
      integrate(interior_facets, dot(plus(grad(v)), Normal{}));
   using PlusTestGradientChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         plus_test_gradient_form));
   static_assert(
      PlusTestGradientChannels::template contains<GradientPlusChannel>());
   static_assert(
      !PlusTestGradientChannels::template contains<GradientMinusChannel>());
   static_assert(
      global_interior_channels_require_plus_side_jacobian_v<
         PlusTestGradientChannels>);

   auto mu =
      MakeCoefficient<"mu", FieldValue<"mu">>(
         [] GENDIL_HOST_DEVICE (const Real mu_q) -> Real
         {
            return mu_q;
         });
   auto traced_coefficient_flux_form =
      integrate(
         interior_facets,
         average(mu * dot(grad(v), Normal{})));
   using TracedCoefficientFluxChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         traced_coefficient_flux_form));
   static_assert(
      TracedCoefficientFluxChannels::template contains<GradientMinusChannel>());
   static_assert(
      TracedCoefficientFluxChannels::template contains<GradientPlusChannel>());
   static_assert(
      global_interior_channels_require_plus_side_jacobian_v<
         TracedCoefficientFluxChannels>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(traced_coefficient_flux_form)>);

   auto eta_F =
      MakeCoefficient<"eta_F">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return 7.0;
         });
   auto sipdg_consistency_form =
      integrate(
         interior_facets,
         - average(mu * dot(grad(u), Normal{})) * jump(v));
   using SipdgConsistencyChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         sipdg_consistency_form));
   static_assert(
      SipdgConsistencyChannels::template contains<ValueMinusChannel>());
   static_assert(
      SipdgConsistencyChannels::template contains<ValuePlusChannel>());
   static_assert(
      !SipdgConsistencyChannels::template contains<GradientMinusChannel>());
   static_assert(
      !SipdgConsistencyChannels::template contains<GradientPlusChannel>());
   static_assert(requires_plus_side_jacobian_v<
      decltype(sipdg_consistency_form)>);

   auto sipdg_symmetry_form =
      integrate(
         interior_facets,
         - average(mu * dot(grad(v), Normal{})) * jump(u));
   using SipdgSymmetryChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         sipdg_symmetry_form));
   static_assert(
      SipdgSymmetryChannels::template contains<GradientMinusChannel>());
   static_assert(
      SipdgSymmetryChannels::template contains<GradientPlusChannel>());
   static_assert(
      !SipdgSymmetryChannels::template contains<ValueMinusChannel>());
   static_assert(
      !SipdgSymmetryChannels::template contains<ValuePlusChannel>());
   static_assert(
      global_interior_channels_require_plus_side_jacobian_v<
         SipdgSymmetryChannels>);
   static_assert(
      global_interior_context_requires_plus_side_jacobian_v<
         decltype(sipdg_symmetry_form),
         SipdgSymmetryChannels>);

   auto sipdg_penalty_form =
      integrate(interior_facets, eta_F * jump(u) * jump(v));
   using SipdgPenaltyChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         sipdg_penalty_form));
   static_assert(
      SipdgPenaltyChannels::template contains<ValueMinusChannel>());
   static_assert(
      SipdgPenaltyChannels::template contains<ValuePlusChannel>());
   static_assert(
      !SipdgPenaltyChannels::template contains<GradientMinusChannel>());
   static_assert(
      !SipdgPenaltyChannels::template contains<GradientPlusChannel>());
   static_assert(!requires_plus_side_jacobian_v<
      decltype(sipdg_penalty_form)>);
   static_assert(
      !global_interior_channels_require_plus_side_jacobian_v<
         SipdgPenaltyChannels>);

   auto scalar_sipdg_form =
      integrate(
         interior_facets,
         - average(mu * dot(grad(u), Normal{})) * jump(v)
         - average(mu * dot(grad(v), Normal{})) * jump(u)
         + eta_F * jump(u) * jump(v));
   using ScalarSipdgChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         scalar_sipdg_form));
   static_assert(ScalarSipdgChannels::template contains<ValueMinusChannel>());
   static_assert(ScalarSipdgChannels::template contains<ValuePlusChannel>());
   static_assert(ScalarSipdgChannels::template contains<GradientMinusChannel>());
   static_assert(ScalarSipdgChannels::template contains<GradientPlusChannel>());
   static_assert(
      global_interior_channels_require_plus_side_jacobian_v<
         ScalarSipdgChannels>);
   static_assert(!has_unqualified_interior_test_trace_v<
      decltype(scalar_sipdg_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(scalar_sipdg_form)>);

   auto untraced_coefficient_form =
      integrate(interior_facets, mu * jump(v));
   static_assert(has_unqualified_side_dependent_inputs_v<
      decltype(untraced_coefficient_form)>);

   auto minus_mu_form =
      integrate(interior_facets, minus(mu) * jump(u) * jump(v));
   auto plus_mu_form =
      integrate(interior_facets, plus(mu) * jump(u) * jump(v));
   auto jump_mu_form =
      integrate(interior_facets, jump(mu) * jump(u) * jump(v));
   auto average_mu_form =
      integrate(interior_facets, average(mu) * jump(u) * jump(v));
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(minus_mu_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(plus_mu_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(jump_mu_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(average_mu_form)>);

   auto side_independent_c =
      MakeCoefficient<"c">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return 2.0;
         });
   auto minus_c_form =
      integrate(interior_facets, minus(side_independent_c) * jump(u) * jump(v));
   auto plus_c_form =
      integrate(interior_facets, plus(side_independent_c) * jump(u) * jump(v));
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(minus_c_form)>);
   static_assert(!has_unqualified_side_dependent_inputs_v<
      decltype(plus_c_form)>);

   auto beta_unit =
      MakeVectorCoefficient<"beta_unit">(
         [] GENDIL_HOST_DEVICE () -> std::array<Real, 1>
         {
            return { 1.0 };
         });
   auto plus_normal_form =
      integrate(
         interior_facets,
         dot(plus(Normal{}), beta_unit) * jump(u) * jump(v));

   auto scalar_normal_flux_form =
      integrate(
         interior_facets,
         average(dot(grad(u), Normal{})) * jump(v));
   using ScalarNormalFluxChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         scalar_normal_flux_form));
   static_assert(local_interior_context_requires_plus_side_jacobian_v<
      decltype(scalar_normal_flux_form)>);
   static_assert(global_interior_context_requires_plus_side_jacobian_v<
      decltype(scalar_normal_flux_form),
      ScalarNormalFluxChannels>);

   VectorTrialSpace<"U"> U;
   VectorTestSpace<"U"> V;
   auto vector_jump_form =
      integrate(interior_facets, dot(jump(U), jump(V)));
   using VectorJumpChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         vector_jump_form));
   static_assert(VectorJumpChannels::template contains<ValueMinusChannel>());
   static_assert(VectorJumpChannels::template contains<ValuePlusChannel>());
   static_assert(!VectorJumpChannels::template contains<ValueChannel>());
   static_assert(!global_interior_channels_require_plus_side_jacobian_v<
      VectorJumpChannels>);

   auto vector_minus_form =
      integrate(interior_facets, dot(minus(U), jump(V)));
   auto vector_plus_form =
      integrate(interior_facets, dot(plus(U), jump(V)));
   using VectorMinusChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         vector_minus_form));
   using VectorPlusChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         vector_plus_form));
   static_assert(VectorMinusChannels::template contains<ValueMinusChannel>());
   static_assert(VectorMinusChannels::template contains<ValuePlusChannel>());
   static_assert(VectorPlusChannels::template contains<ValueMinusChannel>());
   static_assert(VectorPlusChannels::template contains<ValuePlusChannel>());

   auto beta =
      MakeVectorCoefficient<"beta", PhysicalCoordinate>(
         [] (const auto&) -> std::array<Real, 1>
         {
            return { 1.0 };
         });
   auto scalar_upwind_form =
      integrate(interior_facets, upwind(average(beta), u) * jump(v));
   using ScalarUpwindChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         scalar_upwind_form));
   static_assert(ScalarUpwindChannels::template contains<ValueMinusChannel>());
   static_assert(ScalarUpwindChannels::template contains<ValuePlusChannel>());
   static_assert(!ScalarUpwindChannels::template contains<ValueChannel>());
   static_assert(!global_interior_channels_require_plus_side_jacobian_v<
      ScalarUpwindChannels>);

   auto vector_upwind_form =
      integrate(interior_facets, dot(upwind(average(beta), U), jump(V)));
   using VectorUpwindChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         vector_upwind_form));
   static_assert(VectorUpwindChannels::template contains<ValueMinusChannel>());
   static_assert(VectorUpwindChannels::template contains<ValuePlusChannel>());
   static_assert(!VectorUpwindChannels::template contains<ValueChannel>());
   static_assert(!global_interior_channels_require_plus_side_jacobian_v<
      VectorUpwindChannels>);

   auto vector_sipdg_form =
      integrate(
         interior_facets,
         - dot(average(mu * (grad(U) * Normal{})), jump(V))
         - dot(average(mu * (grad(V) * Normal{})), jump(U))
         + eta_F * dot(jump(U), jump(V)));
   using VectorSipdgChannels =
      decltype(LowerGlobalInteriorFacetIntegrandToPullbackChannels(
         vector_sipdg_form));
   static_assert(VectorSipdgChannels::template contains<ValueMinusChannel>());
   static_assert(VectorSipdgChannels::template contains<ValuePlusChannel>());
   static_assert(VectorSipdgChannels::template contains<GradientMinusChannel>());
   static_assert(VectorSipdgChannels::template contains<GradientPlusChannel>());

   using SmallIntegrationRule =
      decltype(MakeIntegrationRule(IntegrationRuleNumPoints<1>{}));
   using MinusSpace = std::remove_cvref_t<decltype(minus_fes)>;
   using PlusSpace = std::remove_cvref_t<decltype(plus_fes)>;
   constexpr size_t small_integrand_scratch =
      generic_operator_integrand_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule>;
   constexpr size_t read_minus_scratch =
      generic_operator_face_read_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         MinusSpace,
         Vector>;
   constexpr size_t read_plus_scratch =
      generic_operator_face_read_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         PlusSpace,
         Vector>;
   constexpr size_t write_minus_scratch =
      generic_operator_face_write_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         MinusSpace,
         Vector>;
   constexpr size_t write_plus_scratch =
      generic_operator_face_write_scratch_requirement_v<
         FullSharedThreadedFaceKernelPolicy,
         PlusSpace,
         Vector>;
   constexpr size_t accurate_small_rule_scratch =
      Max(
         small_integrand_scratch,
         read_minus_scratch,
         read_plus_scratch,
         write_minus_scratch,
         write_plus_scratch);
   constexpr size_t conservative_small_rule_scratch =
      Max(
         small_integrand_scratch,
         read_minus_scratch + read_plus_scratch,
         write_minus_scratch + write_plus_scratch);
   static_assert(read_minus_scratch != read_plus_scratch);
   static_assert(write_minus_scratch != write_plus_scratch);
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         MinusSpace,
         PlusSpace,
         MinusSpace,
         PlusSpace,
         decltype(form),
         Vector,
         Vector,
         Vector,
         Vector> == accurate_small_rule_scratch);
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         MinusSpace,
         PlusSpace,
         MinusSpace,
         PlusSpace,
         decltype(form),
         Vector,
         Vector,
         Vector,
         Vector> != conservative_small_rule_scratch);
   // Scalar SIPDG is now admitted, and the named two-space scratch trait must
   // account for the value and gradient channel/read/write lifetime model.
   static_assert(
      two_space_global_interior_required_shared_memory_v<
         FullSharedThreadedFaceKernelPolicy,
         SmallIntegrationRule,
         MinusSpace,
         PlusSpace,
         MinusSpace,
         PlusSpace,
         decltype(scalar_sipdg_form),
         Vector,
         Vector,
         Vector,
         Vector> == accurate_small_rule_scratch);

   auto op =
      MakeGenericOperator<KernelPolicy>(
         form,
         ctx,
         integration_rule);

   const Integer size = mixed.GetNumberOfFiniteElementDofs();
   const auto dense = BuildDenseMatrix(op, size);

   std::vector<Real> trace(static_cast<size_t>(size), 0.0);
   if constexpr (P2Minus)
   {
      // p2 minus on the + face, p1 plus on the - face.
      trace = { 0.0, 0.0, 1.0, -1.0, 0.0 };
   }
   else
   {
      // p1 minus on the + face, p2 plus on the - face.
      trace = { 0.0, 1.0, -1.0, 0.0, 0.0 };
   }

   const auto ref = MakeRankOneReference(trace);
   bool success = CheckDenseClose(label, dense, ref);

   const auto jump_trace = MakeScalarInterfaceJumpTrace<P2Minus>();
   const auto minus_trace = MakeScalarInterfaceMinusTrace<P2Minus>();
   const auto plus_trace = MakeScalarInterfacePlusTrace<P2Minus>();
   const auto average_trace = MakeScalarInterfaceAverageTrace<P2Minus>();

   auto minus_trial_op =
      MakeGenericOperator<KernelPolicy>(
         minus_trial_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus minus(u) trace dense reference"
            : "p1-minus/p2-plus minus(u) trace dense reference",
         BuildDenseMatrix(minus_trial_op, size),
         MakeOuterProductReference(jump_trace, minus_trace)) &&
      success;

   auto plus_trial_op =
      MakeGenericOperator<KernelPolicy>(
         plus_trial_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus plus(u) trace dense reference"
            : "p1-minus/p2-plus plus(u) trace dense reference",
         BuildDenseMatrix(plus_trial_op, size),
         MakeOuterProductReference(jump_trace, plus_trace)) &&
      success;

   auto explicit_jump_identity_form =
      integrate(interior_facets, (minus(u) - plus(u)) * jump(v));
   auto explicit_jump_identity_op =
      MakeGenericOperator<KernelPolicy>(
         explicit_jump_identity_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus jump identity via minus-plus dense reference"
            : "p1-minus/p2-plus jump identity via minus-plus dense reference",
         BuildDenseMatrix(explicit_jump_identity_op, size),
         ref) &&
      success;

   auto explicit_average_identity_form =
      integrate(interior_facets, (0.5 * (minus(u) + plus(u))) * jump(v));
   auto explicit_average_identity_op =
      MakeGenericOperator<KernelPolicy>(
         explicit_average_identity_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus average identity via half minus-plus dense reference"
            : "p1-minus/p2-plus average identity via half minus-plus dense reference",
         BuildDenseMatrix(explicit_average_identity_op, size),
         MakeOuterProductReference(jump_trace, average_trace)) &&
      success;

   auto average_trial_form =
      integrate(interior_facets, average(u) * jump(v));
   auto average_trial_op =
      MakeGenericOperator<KernelPolicy>(
         average_trial_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus average(u) dense reference"
            : "p1-minus/p2-plus average(u) dense reference",
         BuildDenseMatrix(average_trial_op, size),
         MakeOuterProductReference(jump_trace, average_trace)) &&
      success;

   auto minus_scale_form =
      integrate(interior_facets, minus(ScaleExpr{2.0}) * jump(u) * jump(v));
   auto plus_scale_form =
      integrate(interior_facets, plus(ScaleExpr{2.0}) * jump(u) * jump(v));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus minus(ScaleExpr) identity dense reference"
            : "p1-minus/p2-plus minus(ScaleExpr) identity dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               minus_scale_form,
               ctx,
               integration_rule),
            size),
         ScaleDenseReference(ref, 2.0)) &&
      success;
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus plus(ScaleExpr) identity dense reference"
            : "p1-minus/p2-plus plus(ScaleExpr) identity dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               plus_scale_form,
               ctx,
               integration_rule),
            size),
         ScaleDenseReference(ref, 2.0)) &&
      success;

   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus minus(single-valued coefficient) identity dense reference"
            : "p1-minus/p2-plus minus(single-valued coefficient) identity dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               minus_c_form,
               ctx,
               integration_rule),
            size),
         ScaleDenseReference(ref, 2.0)) &&
      success;
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus plus(single-valued coefficient) identity dense reference"
            : "p1-minus/p2-plus plus(single-valued coefficient) identity dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               plus_c_form,
               ctx,
               integration_rule),
            size),
         ScaleDenseReference(ref, 2.0)) &&
      success;

   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus plus(Normal) canonical direction dense reference"
            : "p1-minus/p2-plus plus(Normal) canonical direction dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               plus_normal_form,
               ctx,
               integration_rule),
            size),
         ref) &&
      success;

   auto scalar_upwind_positive =
      MakeGenericOperator<KernelPolicy>(
         scalar_upwind_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar upwind positive dense reference"
            : "p1-minus/p2-plus scalar upwind positive dense reference",
         BuildDenseMatrix(scalar_upwind_positive, size),
         MakeScalarInterfaceUpwindReference<P2Minus>(1.0)) &&
      success;

   auto beta_negative =
      MakeVectorCoefficient<"beta", PhysicalCoordinate>(
         [] (const auto&) -> std::array<Real, 1>
         {
            return { -2.0 };
         });
   auto scalar_upwind_negative_form =
      integrate(
         interior_facets,
         upwind(average(beta_negative), u) * jump(v));
   auto scalar_upwind_negative =
      MakeGenericOperator<KernelPolicy>(
         scalar_upwind_negative_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar upwind negative dense reference"
            : "p1-minus/p2-plus scalar upwind negative dense reference",
         BuildDenseMatrix(scalar_upwind_negative, size),
         MakeScalarInterfaceUpwindReference<P2Minus>(-2.0)) &&
      success;

   constexpr Real sipdg_mu = 1.75;
   constexpr Real sipdg_eta = 4.25;
   auto sipdg_mu_coeff =
      MakeCoefficient<"sipdg_mu">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return sipdg_mu;
         });
   auto sipdg_eta_coeff =
      MakeCoefficient<"sipdg_eta">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return sipdg_eta;
         });

   auto sipdg_consistency_runtime_form =
      integrate(
         interior_facets,
         - average(sipdg_mu_coeff * dot(grad(u), Normal{})) * jump(v));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar SIPDG consistency dense reference"
            : "p1-minus/p2-plus scalar SIPDG consistency dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               sipdg_consistency_runtime_form,
               ctx,
               integration_rule),
            size),
         MakeScalarInterfaceSipdgConsistencyReference<P2Minus>(
            sipdg_mu,
            sipdg_mu)) &&
      success;

   auto sipdg_symmetry_runtime_form =
      integrate(
         interior_facets,
         - average(sipdg_mu_coeff * dot(grad(v), Normal{})) * jump(u));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar SIPDG symmetry dense reference"
            : "p1-minus/p2-plus scalar SIPDG symmetry dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               sipdg_symmetry_runtime_form,
               ctx,
               integration_rule),
            size),
         MakeScalarInterfaceSipdgSymmetryReference<P2Minus>(
            sipdg_mu,
            sipdg_mu)) &&
      success;

   auto sipdg_penalty_runtime_form =
      integrate(interior_facets, sipdg_eta_coeff * jump(u) * jump(v));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar SIPDG penalty dense reference"
            : "p1-minus/p2-plus scalar SIPDG penalty dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               sipdg_penalty_runtime_form,
               ctx,
               integration_rule),
            size),
         MakeScalarInterfaceSipdgPenaltyReference<P2Minus>(sipdg_eta)) &&
      success;

   auto scalar_sipdg_runtime_form =
      integrate(
         interior_facets,
         - average(sipdg_mu_coeff * dot(grad(u), Normal{})) * jump(v)
         - average(sipdg_mu_coeff * dot(grad(v), Normal{})) * jump(u)
         + sipdg_eta_coeff * jump(u) * jump(v));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus scalar SIPDG complete dense reference"
            : "p1-minus/p2-plus scalar SIPDG complete dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               scalar_sipdg_runtime_form,
               ctx,
               integration_rule),
            size),
         MakeScalarInterfaceSipdgReference<P2Minus>(
            sipdg_mu,
            sipdg_mu,
            sipdg_eta)) &&
      success;

   return success;
}

template <class KernelPolicy, bool P2Minus>
bool TestConformingPAdaptiveGlobalInteriorVectorDenseReference()
{
   constexpr GlobalIndex num_cells = 1;
   Cartesian1DMesh mesh_left(1.0, num_cells);
   Cartesian1DMesh mesh_right(1.0, num_cells);

   auto fe_p1 = MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   auto fe_p2 = MakeLobattoFiniteElement(FiniteElementOrders<2>{});
   auto vector_fe_p1 = MakeVectorFiniteElement(fe_p1, fe_p1);
   auto vector_fe_p2 = MakeVectorFiniteElement(fe_p2, fe_p2);

   auto minus_fes_unshifted =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(mesh_left, vector_fe_p2);
         }
         else
         {
            return MakeFiniteElementSpace(mesh_left, vector_fe_p1);
         }
      }();
   const Integer ndofs_minus =
      minus_fes_unshifted.GetNumberOfFiniteElementDofs();

   auto minus_fes =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(
               mesh_left,
               vector_fe_p2,
               L2Restriction{0});
         }
         else
         {
            return MakeFiniteElementSpace(
               mesh_left,
               vector_fe_p1,
               L2Restriction{0});
         }
      }();
   auto plus_fes =
      [&] ()
      {
         if constexpr (P2Minus)
         {
            return MakeFiniteElementSpace(
               mesh_right,
               vector_fe_p1,
               L2Restriction{ndofs_minus});
         }
         else
         {
            return MakeFiniteElementSpace(
               mesh_right,
               vector_fe_p2,
               L2Restriction{ndofs_minus});
         }
      }();

   using InterfaceConnectivity = CartesianIntermeshFaceConnectivity<1, 1>;
   InterfaceConnectivity interface_faces({num_cells}, {num_cells});
   auto face_fes =
      MakeGlobalInteriorFaceFiniteElementSpace(
         minus_fes,
         plus_fes,
         interface_faces);
   auto mixed = MakeMixedFiniteElementSpace(
      minus_fes,
      plus_fes,
      face_fes);

   auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5>{});
   VectorTrialSpace<"U"> U;
   VectorTestSpace<"U"> V;
   InteriorFacets<"solid"> interior_facets;
   auto ctx = MakeWeakFormContext(
      MakeTrialField<"U">(mixed),
      MakeIntegrationDomain<"solid">(mixed));
   const Integer size = mixed.GetNumberOfFiniteElementDofs();

   auto vector_jump_form =
      integrate(interior_facets, dot(jump(U), jump(V)));
   auto vector_jump_op =
      MakeGenericOperator<KernelPolicy>(
         vector_jump_form,
         ctx,
         integration_rule);
   bool success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector jump dense reference"
            : "p1-minus/p2-plus vector jump dense reference",
         BuildDenseMatrix(vector_jump_op, size),
         MakeVectorInterfaceJumpReference<P2Minus>());

   auto vector_minus_form =
      integrate(interior_facets, dot(minus(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector minus(U) dense reference"
            : "p1-minus/p2-plus vector minus(U) dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_minus_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceExplicitTraceReference<P2Minus>(1.0, 0.0)) &&
      success;

   auto vector_plus_form =
      integrate(interior_facets, dot(plus(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector plus(U) dense reference"
            : "p1-minus/p2-plus vector plus(U) dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_plus_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceExplicitTraceReference<P2Minus>(0.0, 1.0)) &&
      success;

   auto vector_explicit_jump_form =
      integrate(interior_facets, dot(minus(U) - plus(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector jump identity via minus-plus dense reference"
            : "p1-minus/p2-plus vector jump identity via minus-plus dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_explicit_jump_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceJumpReference<P2Minus>()) &&
      success;

   auto vector_average_form =
      integrate(interior_facets, dot(average(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector average(U) dense reference"
            : "p1-minus/p2-plus vector average(U) dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_average_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceExplicitTraceReference<P2Minus>(0.5, 0.5)) &&
      success;

   auto beta_positive =
      MakeVectorCoefficient<"beta", PhysicalCoordinate>(
         [] (const auto&) -> std::array<Real, 1>
         {
            return { 1.0 };
         });
   auto vector_upwind_positive_form =
      integrate(
         interior_facets,
         dot(upwind(average(beta_positive), U), jump(V)));
   auto vector_upwind_positive =
      MakeGenericOperator<KernelPolicy>(
         vector_upwind_positive_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector upwind positive dense reference"
            : "p1-minus/p2-plus vector upwind positive dense reference",
         BuildDenseMatrix(vector_upwind_positive, size),
         MakeVectorInterfaceUpwindReference<P2Minus>(1.0)) &&
      success;

   auto beta_negative =
      MakeVectorCoefficient<"beta", PhysicalCoordinate>(
         [] (const auto&) -> std::array<Real, 1>
         {
            return { -2.0 };
         });
   auto vector_upwind_negative_form =
      integrate(
         interior_facets,
         dot(upwind(average(beta_negative), U), jump(V)));
   auto vector_upwind_negative =
      MakeGenericOperator<KernelPolicy>(
         vector_upwind_negative_form,
         ctx,
         integration_rule);
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector upwind negative dense reference"
            : "p1-minus/p2-plus vector upwind negative dense reference",
         BuildDenseMatrix(vector_upwind_negative, size),
         MakeVectorInterfaceUpwindReference<P2Minus>(-2.0)) &&
      success;

   constexpr Real sipdg_mu = 1.75;
   constexpr Real sipdg_eta = 4.25;
   auto sipdg_mu_coeff =
      MakeCoefficient<"vector_sipdg_mu">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return sipdg_mu;
         });
   auto sipdg_eta_coeff =
      MakeCoefficient<"vector_sipdg_eta">(
         [] GENDIL_HOST_DEVICE () -> Real
         {
            return sipdg_eta;
         });

   auto vector_sipdg_consistency_form =
      integrate(
         interior_facets,
         - dot(average(sipdg_mu_coeff * (grad(U) * Normal{})), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector SIPDG consistency dense reference"
            : "p1-minus/p2-plus vector SIPDG consistency dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_sipdg_consistency_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceSipdgConsistencyReference<P2Minus>(
            sipdg_mu,
            sipdg_mu)) &&
      success;

   auto vector_sipdg_symmetry_form =
      integrate(
         interior_facets,
         - dot(average(sipdg_mu_coeff * (grad(V) * Normal{})), jump(U)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector SIPDG symmetry dense reference"
            : "p1-minus/p2-plus vector SIPDG symmetry dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_sipdg_symmetry_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceSipdgSymmetryReference<P2Minus>(
            sipdg_mu,
            sipdg_mu)) &&
      success;

   auto vector_sipdg_penalty_form =
      integrate(interior_facets, sipdg_eta_coeff * dot(jump(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector SIPDG penalty dense reference"
            : "p1-minus/p2-plus vector SIPDG penalty dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_sipdg_penalty_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceSipdgPenaltyReference<P2Minus>(sipdg_eta)) &&
      success;

   auto vector_sipdg_form =
      integrate(
         interior_facets,
         - dot(average(sipdg_mu_coeff * (grad(U) * Normal{})), jump(V))
         - dot(average(sipdg_mu_coeff * (grad(V) * Normal{})), jump(U))
         + sipdg_eta_coeff * dot(jump(U), jump(V)));
   success =
      CheckDenseClose(
         P2Minus
            ? "p2-minus/p1-plus vector SIPDG complete dense reference"
            : "p1-minus/p2-plus vector SIPDG complete dense reference",
         BuildDenseMatrix(
            MakeGenericOperator<KernelPolicy>(
               vector_sipdg_form,
               ctx,
               integration_rule),
            size),
         MakeVectorInterfaceSipdgReference<P2Minus>(
            sipdg_mu,
            sipdg_mu,
            sipdg_eta)) &&
      success;

   return success;
}

bool TestPAdaptiveConformingGlobalInteriorCanonical()
{
   bool success = true;
   success =
      TestSameSpaceCanonicalGlobalInteriorParity<SerialKernelConfiguration>() &&
      success;
   success =
      TestConformingPAdaptiveGlobalInteriorDenseReference<
         SerialKernelConfiguration,
         false>("p1-minus/p2-plus dense reference") &&
      success;
   success =
      TestConformingPAdaptiveGlobalInteriorDenseReference<
         SerialKernelConfiguration,
         true>("p2-minus/p1-plus dense reference") &&
      success;
   success =
      TestConformingPAdaptiveGlobalInteriorVectorDenseReference<
         SerialKernelConfiguration,
         false>() &&
      success;
   success =
      TestConformingPAdaptiveGlobalInteriorVectorDenseReference<
         SerialKernelConfiguration,
         true>() &&
      success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestAggregation() && success;
   success = TestCellLayout<false>("mixed cell layout with trial-only field") && success;
   success = TestCellLayout<true>("mixed cell layout with explicit trial/test fields") && success;
   success = TestInteriorAndBoundaryFaces() && success;
   success = TestUnusedCrossSpaceDomainIsNotSelected() && success;
   success = TestRestrictedFaceContextsKeepFaceFieldBindings() && success;
   success = TestDuplicateSameTypeBoundaryUsesDescriptorCellIndex() && success;
   success = TestPAdaptiveConformingGlobalInteriorCanonical() && success;

   if (!success)
   {
      return 1;
   }

   std::cout << "\nAll mixed integration domain tests passed.\n";
   return 0;
}
