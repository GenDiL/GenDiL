// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

using namespace gendil;

namespace
{

constexpr Real device_oracle_tolerance = Real{1.0e-10};
constexpr Real device_sensitivity_tolerance = Real{1.0e-8};
constexpr Real device_nonzero_tolerance = Real{1.0e-12};

#if defined(GENDIL_USE_DEVICE)
template<Integer Q, Integer BatchSize>
using Face2DBatchKernelPolicy = DeviceKernelConfiguration<
   ThreadBlockLayout<Q>,
   1,
   BatchSize>;

template<Integer Q>
using Face2DKernelPolicy = Face2DBatchKernelPolicy<Q, 2>;

template<Integer BatchSize>
using RegisterOnlyBatchKernelPolicy = DeviceKernelConfiguration<
   ThreadBlockLayout<>,
   0,
   BatchSize>;

template<Integer Q>
using Face3DKernelPolicy = DeviceKernelConfiguration<
   ThreadBlockLayout<Q, Q>,
   2,
   1>;

static_assert(Face2DKernelPolicy<4>::thread_block_dim == 1);
static_assert(Face2DKernelPolicy<4>::shared_block_max_dim >= 1);
static_assert(Face3DKernelPolicy<4>::thread_block_dim == 2);
static_assert(Face3DKernelPolicy<4>::shared_block_max_dim >= 2);
static_assert(details::SparseAssemblyKernelContract<
   MatrixAssemblyType::RawCOO,
   Face2DBatchKernelPolicy<4, 4>>::batching_supported);
static_assert(!details::SparseAssemblyKernelContract<
   MatrixAssemblyType::BSR,
   Face2DBatchKernelPolicy<4, 4>>::batching_supported);
#else
template<Integer, Integer>
using Face2DBatchKernelPolicy = SerialKernelConfiguration;

template<Integer>
using Face2DKernelPolicy = SerialKernelConfiguration;

template<Integer>
using Face3DKernelPolicy = SerialKernelConfiguration;

template<Integer>
using RegisterOnlyBatchKernelPolicy = SerialKernelConfiguration;
#endif

struct RuntimeOrientedSegmentCell
{
   static constexpr Integer Dim = 1;
   using geometry = HyperCube<1>;
   using physical_coordinates = std::array<Real, 1>;
   using jacobian = std::array<Real, 1>;

   Point<1> origin{};
   Real extent = 0.0;

   template<class IntegrationRule>
   using QuadData = TensorProductData<std::tuple_element_t<
      0,
      typename IntegrationRule::points::points_1d_tuple>>;

   template<class QData>
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian(
      const TensorIndex<1>& q,
      const QData& qdata,
      physical_coordinates& x,
      jacobian& jacobian_) const
   {
      x[0] = origin[0] + extent * GetCoord<0>(qdata, q[0]);
      jacobian_[0] = extent;
   }

   GENDIL_HOST_DEVICE
   jacobian ComputeJacobian(const Point<1>&) const
   {
      return {extent};
   }
};

GENDIL_HOST_DEVICE GENDIL_INLINE
void ApplyOrientationToCell(
   const Permutation<1>& orientation,
   RuntimeOrientedSegmentCell& cell)
{
   GENDIL_ASSERT(
      IsValidSignedPermutation(orientation),
      "Runtime segment orientation must be a signed permutation.");
   if (orientation(0) < 0)
   {
      cell.origin[0] += cell.extent;
      cell.extent = -cell.extent;
   }
}

struct RuntimeOrientedLineMesh
{
   static constexpr Integer Dim = 1;
   using cell_type = RuntimeOrientedSegmentCell;

   struct ConnectivityIdentity
   {
      GlobalIndex size = 2;
   };

   Real h = 1.0;
   Point<1> mesh_origin{};
   ConnectivityIdentity connectivity{};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfCells() const
   {
      return connectivity.size;
   }

   GENDIL_HOST_DEVICE
   cell_type GetCell(const GlobalIndex element) const
   {
      return {
         Point<1>{mesh_origin[0] + h * static_cast<Real>(element)},
         h};
   }
};

struct StaticIdentityLineInteriorConnectivity
{
   using geometry = HyperCube<1>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      IdentityOrientation<1>,
      CanonicalVector<1, 0, 1>,
      ConformingFaceMap<1>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 0>,
      geometry,
      IdentityOrientation<1>,
      CanonicalVector<1, 0, -1>,
      ConformingFaceMap<1>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(GlobalIndex) const
   {
      return face_info_type{
         {0, {}, {}, {}, {}, {}},
         {1, {}, {}, {}, {}, {}}};
   }
};

struct RuntimeSignedLineInteriorConnectivity
{
   using geometry = HyperCube<1>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      IdentityOrientation<1>,
      CanonicalVector<1, 0, 1>,
      ConformingFaceMap<1>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 0>,
      geometry,
      Permutation<1>,
      CanonicalVector<1, 0, -1>,
      ConformingFaceMap<1>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(GlobalIndex) const
   {
      return face_info_type{
         {0, {}, {}, {}, {}, {}},
         {1, {}, Permutation<1>{{-1}}, {}, {}, {}}};
   }
};

using RuntimeSecondFactorOrientation = TensorProductOrientation<
   IdentityOrientation<1>,
   Permutation<1>>;
using RuntimeLeftNestedMiddleFactorOrientation = TensorProductOrientation<
   TensorProductOrientation<IdentityOrientation<1>, Permutation<1>>,
   IdentityOrientation<1>>;
using RuntimeRightNestedMiddleFactorOrientation = TensorProductOrientation<
   IdentityOrientation<1>,
   TensorProductOrientation<Permutation<1>, IdentityOrientation<1>>>;

struct ManualStaticIdentitySecondFactorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 3>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 1, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 1, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 2> records{Record{0, 2}, Record{1, 3}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const Record record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          {},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          {},
          {},
          {},
          {}}};
   }
};

struct RuntimeSignedSecondFactorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 3>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 1, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      RuntimeSecondFactorOrientation,
      CanonicalVector<2, 1, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 2> records{Record{0, 2}, Record{1, 3}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const Record record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          {},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          MakeTensorProductOrientation(
             IdentityOrientation<1>{},
             Permutation<1>{{-1}}),
          {},
          {},
          {}}};
   }
};

struct RuntimeUnsignedSecondFactorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 3>,
      geometry,
      IdentityOrientation<2>,
      CanonicalVector<2, 1, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      RuntimeSecondFactorOrientation,
      CanonicalVector<2, 1, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 2> records{Record{0, 2}, Record{1, 3}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const Record record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          {},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          MakeTensorProductOrientation(
             IdentityOrientation<1>{},
             Permutation<1>{{1}}),
          {},
          {},
          {}}};
   }
};

struct RuntimeSignedMiddleFactorConnectivity
{
   using geometry = HyperCube<3>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 4>,
      geometry,
      IdentityOrientation<3>,
      CanonicalVector<3, 1, 1>,
      ConformingFaceMap<3>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      RuntimeLeftNestedMiddleFactorOrientation,
      CanonicalVector<3, 1, -1>,
      ConformingFaceMap<3>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 4> records{
      Record{0, 2}, Record{1, 3}, Record{4, 6}, Record{5, 7}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const Record record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          {},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          MakeTensorProductOrientation(
             MakeTensorProductOrientation(
                IdentityOrientation<1>{},
                Permutation<1>{{-1}}),
             IdentityOrientation<1>{}),
          {},
          {},
          {}}};
   }
};

struct VectorMetrics
{
   Real absolute_error;
   Real relative_error;
   Real reference_norm;
};

VectorMetrics CompareVectors(const Vector& actual, const Vector& reference)
{
   if (actual.Size() != reference.Size())
   {
      return {Real{1}, Real{1}, Real{0}};
   }
   const Real* actual_data = actual.ReadHostData();
   const Real* reference_data = reference.ReadHostData();
   Real error2 = 0;
   Real norm2 = 0;
   for (Integer i = 0; i < actual.Size(); ++i)
   {
      const Real difference = actual_data[i] - reference_data[i];
      error2 += difference * difference;
      norm2 += reference_data[i] * reference_data[i];
   }
   const Real absolute_error = std::sqrt(error2);
   const Real reference_norm = std::sqrt(norm2);
   return {
      absolute_error,
      absolute_error / std::max(reference_norm, Real{1.0e-30}),
      reference_norm};
}

bool CheckClose(
   const char* label,
   const Vector& actual,
   const Vector& reference)
{
   const VectorMetrics metrics = CompareVectors(actual, reference);
   std::cout << label
             << ": absolute=" << metrics.absolute_error
             << " relative=" << metrics.relative_error
             << " reference=" << metrics.reference_norm << '\n';
   return actual.Size() == reference.Size() &&
      metrics.reference_norm > device_nonzero_tolerance &&
      metrics.absolute_error <= device_oracle_tolerance *
         std::max(Real{1}, metrics.reference_norm);
}

Real DifferenceNorm(const Vector& left, const Vector& right)
{
   return CompareVectors(left, right).absolute_error;
}

template<class Operator>
Vector ApplySynchronized(const Operator& op, const Vector& input)
{
   Vector output(input.Size());
   output = 0.0;
   op(input, output);
   GENDIL_DEVICE_SYNC;
   output.ReadHostData();
   return output;
}

void Fill2DInput(Vector& input)
{
   Real* data = input.WriteHostData();
   constexpr Integer dofs_1d = 3;
   constexpr Integer dofs_per_element = dofs_1d * dofs_1d;
   for (Integer i = 0; i < input.Size(); ++i)
   {
      const Integer element = i / dofs_per_element;
      const Integer local = i - element * dofs_per_element;
      const Integer x = local % dofs_1d;
      const Integer y = local / dofs_1d;
      data[i] = Real{0.17} + Real{0.11} * element +
                Real{0.03} * x + Real{0.07} * y +
                Real{0.013} * x * y;
   }
}

void Fill3DInput(Vector& input)
{
   Real* data = input.WriteHostData();
   constexpr Integer dofs_1d = 3;
   constexpr Integer dofs_per_element =
      dofs_1d * dofs_1d * dofs_1d;
   for (Integer i = 0; i < input.Size(); ++i)
   {
      const Integer element = i / dofs_per_element;
      const Integer local = i - element * dofs_per_element;
      const Integer x = local % dofs_1d;
      const Integer y = (local / dofs_1d) % dofs_1d;
      const Integer z = local / (dofs_1d * dofs_1d);
      data[i] = Real{0.19} + Real{0.09} * element +
                Real{0.021} * x + Real{0.057} * y +
                Real{0.113} * z + Real{0.007} * x * z;
   }
}

bool TestOriginalInteriorBoundarySmoke()
{
   constexpr GlobalIndex nx = 2;
   constexpr GlobalIndex ny = 3;
   const Cartesian1DMesh x_mesh(0.5, nx, -0.25);
   const Cartesian1DMesh y_mesh(0.25, ny, 0.5);
   const auto x_partition = MakePartition(
      MakeCellPart(x_mesh),
      MakeInteriorFacePart<0, 0>(
         MakeCartesianInteriorFaceConnectivity<1>({nx})),
      MakeBoundaryFacePart<0>(
         MakeCartesianBoundaryFaceConnectivity<1>({nx})));
   const auto y_partition = MakePartition(MakeCellPart(y_mesh));
   const auto product =
      MakeCartesianProductPartition(x_partition, y_partition);

   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto space = MakeMixedFiniteElementSpace(
      product,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto form =
      integrate(Cells<"mesh">{}, u * v) +
      integrate(InteriorFacets<"mesh">{}, jump(u) * jump(v)) +
      integrate(BoundaryFacets<"mesh">{}, u * v);
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(product));
   constexpr Integer q = 4;
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<q, q>{});
   const auto op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, context, rule);
   Vector input(space.GetNumberOfFiniteElementDofs());
   Fill2DInput(input);
   const Vector output = ApplySynchronized(op, input);
   const Real norm = CompareVectors(output, output).reference_norm;

   const auto raw_batch1 = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      Face2DBatchKernelPolicy<q, 1>>(
         form,
         context,
         rule);
   const auto coo_batch1 = FinalizeRawCOOToCOOHost(raw_batch1);
   const auto coo_batch1_output = ApplySynchronized(coo_batch1, input);

   const auto check_sparse_batch = [&]<class KernelPolicy>(
      const char* label)
   {
      const auto raw = GenericAssembly<
         MatrixAssemblyType::RawCOO,
         KernelPolicy>(
            form,
            context,
            rule);
      const auto coo = FinalizeRawCOOToCOOHost(raw);
      bool batch_success =
         raw.num_rows == raw_batch1.num_rows &&
         raw.num_cols == raw_batch1.num_cols &&
         raw.nnz_raw == raw_batch1.nnz_raw &&
         coo.nnz == coo_batch1.nnz;
      const auto got = GetHostReadView(coo);
      const auto reference = GetHostReadView(coo_batch1);
      for (GlobalIndex i = 0; i < coo.nnz && batch_success; ++i)
      {
         batch_success =
            got.rows[i] == reference.rows[i] &&
            got.cols[i] == reference.cols[i] &&
            std::abs(got.values[i] - reference.values[i]) <=
               device_oracle_tolerance *
                  std::max(Real{1}, std::abs(reference.values[i]));
      }
      const auto batch_output = ApplySynchronized(coo, input);
      if (!batch_success)
      {
         std::cerr << label
                   << ": canonical RawCOO triplets differ from BatchSize=1\n";
      }
      return batch_success &&
         CheckClose(label, batch_output, coo_batch1_output);
   };

   bool success =
      std::isfinite(norm) && norm > device_nonzero_tolerance;
   success = CheckClose(
      "partition sparse BatchSize=1 versus matrix-free",
      coo_batch1_output,
      output) && success;
   success = check_sparse_batch.template operator()<
      Face2DBatchKernelPolicy<q, 2>>(
         "partition sparse threaded BatchSize=2") && success;
   success = check_sparse_batch.template operator()<
      Face2DBatchKernelPolicy<q, 4>>(
         "partition sparse threaded BatchSize=4") && success;
   success = check_sparse_batch.template operator()<
      RegisterOnlyBatchKernelPolicy<2>>(
         "partition sparse register-only BatchSize=2") && success;
#if defined(GENDIL_USE_DEVICE)
   success = check_sparse_batch.template operator()<
      Face2DBatchKernelPolicy<q, device_warp_size>>(
         "partition sparse threaded warp BatchSize") && success;
#endif
   return success;
}

bool TestStaticIdentitySecondFactorOracle()
{
   const Cartesian1DMesh first_mesh(0.2, 2, -0.45);
   const Cartesian1DMesh second_mesh(0.7, 2, 0.35);
   const auto first = MakePartition(MakeCellPart(first_mesh));
   const auto second = MakePartition(
      MakeCellPart(second_mesh),
      MakeInteriorFacePart<0, 0>(
         StaticIdentityLineInteriorConnectivity{}));
   const auto generated = MakeCartesianProductPartition(first, second);
   const auto manual = MakePartition(
      MakeCellPart(MakeCartesianProductMesh(first_mesh, second_mesh)),
      MakeInteriorFacePart<0, 0>(
         ManualStaticIdentitySecondFactorConnectivity{}));

   const auto generated_face =
      std::get<0>(generated.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(1);
   const auto manual_face =
      std::get<0>(manual.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(1);
   using GeneratedFace = std::remove_cvref_t<decltype(generated_face)>;
   using ManualFace = std::remove_cvref_t<decltype(manual_face)>;
   static_assert(
      GeneratedFace::minus_side_type::local_face_index_type::value == 3);
   static_assert(
      GeneratedFace::plus_side_type::local_face_index_type::value == 1);
   static_assert(GeneratedFace::minus_side_type::normal_type::index == 1);
   static_assert(GeneratedFace::plus_side_type::normal_type::index == 1);
   static_assert(std::same_as<
      typename GeneratedFace::minus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename GeneratedFace::plus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename ManualFace::minus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename ManualFace::plus_side_type::orientation_type,
      IdentityOrientation<2>>);

   if (generated_face.MinusSide().GetCellIndex() != 1 ||
       generated_face.PlusSide().GetCellIndex() != 3 ||
       manual_face.MinusSide().GetCellIndex() != 1 ||
       manual_face.PlusSide().GetCellIndex() != 3)
   {
      return false;
   }

   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   const auto manual_space = MakeMixedFiniteElementSpace(
      manual,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto coefficient = MakeCoefficient<"a", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                Real{5} * X[0] * X[1];
      });
   const auto expression =
      coefficient * dot(plus(grad(u)), Normal{}) * jump(v);
   const auto form = integrate(InteriorFacets<"mesh">{}, expression);
   static_assert(requires_plus_side_jacobian_v<decltype(expression)>);
   static_assert(requires_plus_side_jacobian_v<decltype(form)>);
   constexpr Integer q = 4;
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<q, q>{});
   const auto generated_context = MakeWeakFormContext(
      MakeTrialField<"u">(generated_space),
      MakeIntegrationDomain<"mesh">(generated));
   const auto manual_context = MakeWeakFormContext(
      MakeTrialField<"u">(manual_space),
      MakeIntegrationDomain<"mesh">(manual));
   const auto generated_op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, generated_context, rule);
   const auto manual_op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, manual_context, rule);
   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   Fill2DInput(input);
   const Vector actual = ApplySynchronized(generated_op, input);
   const Vector reference = ApplySynchronized(manual_op, input);
   return CheckClose(
      "device generated versus manual static identity oracle",
      actual,
      reference);
}

bool TestRuntimeSignedSecondFactorOracleAndSensitivity()
{
   const Cartesian1DMesh first_mesh(0.2, 2, -0.45);
   const RuntimeOrientedLineMesh second_mesh{
      0.7, Point<1>{0.35}, {2}};
   const auto first = MakePartition(MakeCellPart(first_mesh));
   const auto second = MakePartition(
      MakeCellPart(second_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeSignedLineInteriorConnectivity{}));
   const auto generated = MakeCartesianProductPartition(first, second);
   const auto product_mesh =
      MakeCartesianProductMesh(first_mesh, second_mesh);
   const auto signed_manual = MakePartition(
      MakeCellPart(product_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeSignedSecondFactorConnectivity{}));
   const auto unsigned_manual = MakePartition(
      MakeCellPart(product_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeUnsignedSecondFactorConnectivity{}));

   const auto generated_face =
      std::get<0>(generated.InteriorFaceParts()).face_mesh.
         GetGlobalFaceInfo(0);
   using GeneratedFace = std::remove_cvref_t<decltype(generated_face)>;
   static_assert(std::same_as<
      typename GeneratedFace::minus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename GeneratedFace::plus_side_type::orientation_type,
      RuntimeSecondFactorOrientation>);
   if (FlattenOrientation(generated_face.PlusSide().GetOrientation()) !=
       Permutation<2>{{1, -2}})
   {
      return false;
   }

   const auto fe = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   const auto signed_space = MakeMixedFiniteElementSpace(
      signed_manual, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   const auto unsigned_space = MakeMixedFiniteElementSpace(
      unsigned_manual, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto coefficient = MakeCoefficient<"a", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                Real{5} * X[0] * X[1];
      });
   const auto expression =
      coefficient * dot(plus(grad(u)), Normal{}) * jump(v);
   static_assert(requires_plus_side_jacobian_v<decltype(expression)>);
   const auto form = integrate(
      InteriorFacets<"mesh">{},
      expression);
   constexpr Integer q = 4;
   const auto rule = MakeIntegrationRule(IntegrationRuleNumPoints<q, q>{});
   const auto generated_context = MakeWeakFormContext(
      MakeTrialField<"u">(generated_space),
      MakeIntegrationDomain<"mesh">(generated));
   const auto signed_context = MakeWeakFormContext(
      MakeTrialField<"u">(signed_space),
      MakeIntegrationDomain<"mesh">(signed_manual));
   const auto unsigned_context = MakeWeakFormContext(
      MakeTrialField<"u">(unsigned_space),
      MakeIntegrationDomain<"mesh">(unsigned_manual));
   const auto generated_op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, generated_context, rule);
   const auto signed_op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, signed_context, rule);
   const auto unsigned_op = MakeGenericOperator<Face2DKernelPolicy<q>>(
      form, unsigned_context, rule);
   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   Fill2DInput(input);
   const Vector actual = ApplySynchronized(generated_op, input);
   const Vector signed_reference = ApplySynchronized(signed_op, input);
   const Vector unsigned_control = ApplySynchronized(unsigned_op, input);
   const bool oracle_matches = CheckClose(
      "device signed orientation versus literal signed oracle",
      actual,
      signed_reference);
   const VectorMetrics signed_metrics =
      CompareVectors(signed_reference, signed_reference);
   const Real sensitivity =
      DifferenceNorm(signed_reference, unsigned_control);
   std::cout << "device signed/unsigned sensitivity: difference="
             << sensitivity << " reference="
             << signed_metrics.reference_norm << '\n';
   return oracle_matches &&
      sensitivity > device_sensitivity_tolerance *
         std::max(Real{1}, signed_metrics.reference_norm);
}

template<class FirstSpace, class SecondSpace, class ThirdSpace>
bool CheckL2Ordering(
   const FirstSpace& first,
   const SecondSpace& second,
   const ThirdSpace& third)
{
   const auto& first_cell = first.template GetCellFiniteElementSpace<0>();
   const auto& second_cell = second.template GetCellFiniteElementSpace<0>();
   const auto& third_cell = third.template GetCellFiniteElementSpace<0>();
   const auto& first_restriction = GetRestriction(first_cell);
   const auto& second_restriction = GetRestriction(second_cell);
   const auto& third_restriction = GetRestriction(third_cell);
   static_assert(ElementwiseIndependentRestriction<
      std::remove_cvref_t<decltype(first_restriction)>>);
   static_assert(ElementwiseIndependentRestriction<
      std::remove_cvref_t<decltype(second_restriction)>>);
   static_assert(ElementwiseIndependentRestriction<
      std::remove_cvref_t<decltype(third_restriction)>>);
   constexpr GlobalIndex local_dofs = 27;
   if (first_cell.GetNumberOfFiniteElements() !=
          second_cell.GetNumberOfFiniteElements() ||
       first_cell.GetNumberOfFiniteElements() !=
          third_cell.GetNumberOfFiniteElements())
   {
      return false;
   }
   for (GlobalIndex element = 0;
        element < first_cell.GetNumberOfFiniteElements();
        ++element)
   {
      for (GlobalIndex local = 0; local < local_dofs; ++local)
      {
         const GlobalIndex occurrence = element * local_dofs + local;
         if (GetGlobalDofIndex(first_cell, element, local) !=
                first_restriction.shift + occurrence ||
             GetGlobalDofIndex(second_cell, element, local) !=
                second_restriction.shift + occurrence ||
             GetGlobalDofIndex(third_cell, element, local) !=
                third_restriction.shift + occurrence)
         {
            return false;
         }
      }
   }
   return true;
}

bool TestNestedMiddleFactorOracle()
{
   const Cartesian1DMesh a_mesh(0.3, 2, -0.4);
   const RuntimeOrientedLineMesh b_mesh{
      0.7, Point<1>{0.2}, {2}};
   const Cartesian1DMesh c_mesh(0.15, 2, 1.1);
   const auto a = MakePartition(MakeCellPart(a_mesh));
   const auto b = MakePartition(
      MakeCellPart(b_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeSignedLineInteriorConnectivity{}));
   const auto c = MakePartition(MakeCellPart(c_mesh));
   const auto left = MakeCartesianProductPartition(
      MakeCartesianProductPartition(a, b),
      c);
   const auto right = MakeCartesianProductPartition(
      a,
      MakeCartesianProductPartition(b, c));
   const auto manual_mesh = MakeCartesianProductMesh(
      MakeCartesianProductMesh(a_mesh, b_mesh),
      c_mesh);
   const auto manual = MakePartition(
      MakeCellPart(manual_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeSignedMiddleFactorConnectivity{}));

   const auto left_face =
      std::get<0>(left.InteriorFaceParts()).face_mesh.GetGlobalFaceInfo(3);
   const auto right_face =
      std::get<0>(right.InteriorFaceParts()).face_mesh.GetGlobalFaceInfo(3);
   using LeftFace = std::remove_cvref_t<decltype(left_face)>;
   using RightFace = std::remove_cvref_t<decltype(right_face)>;
   static_assert(
      LeftFace::minus_side_type::local_face_index_type::value == 4);
   static_assert(
      LeftFace::plus_side_type::local_face_index_type::value == 1);
   static_assert(
      RightFace::minus_side_type::local_face_index_type::value == 4);
   static_assert(
      RightFace::plus_side_type::local_face_index_type::value == 1);
   static_assert(LeftFace::minus_side_type::normal_type::index == 1);
   static_assert(RightFace::minus_side_type::normal_type::index == 1);
   static_assert(std::same_as<
      typename LeftFace::minus_side_type::orientation_type,
      IdentityOrientation<3>>);
   static_assert(std::same_as<
      typename LeftFace::plus_side_type::orientation_type,
      RuntimeLeftNestedMiddleFactorOrientation>);
   static_assert(std::same_as<
      typename RightFace::minus_side_type::orientation_type,
      IdentityOrientation<3>>);
   static_assert(std::same_as<
      typename RightFace::plus_side_type::orientation_type,
      RuntimeRightNestedMiddleFactorOrientation>);
   if (FlattenOrientation(left_face.PlusSide().GetOrientation()) !=
          Permutation<3>{{1, -2, 3}} ||
       FlattenOrientation(right_face.PlusSide().GetOrientation()) !=
          Permutation<3>{{1, -2, 3}} ||
       left_face.MinusSide().GetCellIndex() != 5 ||
       left_face.PlusSide().GetCellIndex() != 7 ||
       right_face.MinusSide().GetCellIndex() != 5 ||
       right_face.PlusSide().GetCellIndex() != 7)
   {
      return false;
   }

   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   const auto left_space = MakeMixedFiniteElementSpace(
      left, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   const auto right_space = MakeMixedFiniteElementSpace(
      right, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   const auto manual_space = MakeMixedFiniteElementSpace(
      manual, std::tuple{fe}, std::tuple{ContiguousL2RestrictionSpecification{0}});
   if (!CheckL2Ordering(left_space, right_space, manual_space))
   {
      return false;
   }

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto coefficient = MakeCoefficient<"a", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                Real{5} * X[2] + Real{7} * X[0] * X[2];
      });
   const auto form = integrate(
      InteriorFacets<"mesh">{},
      coefficient * jump(u) * jump(v));
   constexpr Integer q = 4;
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<q, q, q>{});
   const auto left_context = MakeWeakFormContext(
      MakeTrialField<"u">(left_space),
      MakeIntegrationDomain<"mesh">(left));
   const auto right_context = MakeWeakFormContext(
      MakeTrialField<"u">(right_space),
      MakeIntegrationDomain<"mesh">(right));
   const auto manual_context = MakeWeakFormContext(
      MakeTrialField<"u">(manual_space),
      MakeIntegrationDomain<"mesh">(manual));
   const auto left_op = MakeGenericOperator<Face3DKernelPolicy<q>>(
      form, left_context, rule);
   const auto right_op = MakeGenericOperator<Face3DKernelPolicy<q>>(
      form, right_context, rule);
   const auto manual_op = MakeGenericOperator<Face3DKernelPolicy<q>>(
      form, manual_context, rule);
   Vector input(left_space.GetNumberOfFiniteElementDofs());
   Fill3DInput(input);
   const Vector left_output = ApplySynchronized(left_op, input);
   const Vector right_output = ApplySynchronized(right_op, input);
   const Vector manual_output = ApplySynchronized(manual_op, input);
   const bool left_matches = CheckClose(
      "device left-associated middle-factor oracle",
      left_output,
      manual_output);
   const bool right_matches = CheckClose(
      "device right-associated middle-factor oracle",
      right_output,
      manual_output);
   const bool associations_match = CheckClose(
      "device nested association equivalence",
      left_output,
      right_output);
   return left_matches && right_matches && associations_match;
}

} // namespace

int main()
{
   bool success = true;
   success = TestOriginalInteriorBoundarySmoke() && success;
   success = TestStaticIdentitySecondFactorOracle() && success;
   success = TestRuntimeSignedSecondFactorOracleAndSensitivity() && success;
   success = TestNestedMiddleFactorOracle() && success;
   return success ? 0 : 1;
}
