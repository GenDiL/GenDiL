// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

template <typename VectorType>
void FillPattern(VectorType& x)
{
   Real* data = x.WriteHostData();
   for (Integer i = 0; i < x.Size(); ++i)
   {
      data[i] = 0.37 + 0.19 * i + 0.03 * ((i % 5) + 1);
   }
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
      make_cartesian_interior_face_connectivity<1>({num_cells});
   auto boundary_faces =
      make_cartesian_boundary_face_connectivity<1>({num_cells});

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
      !is_std_tuple_v<
         decltype(mixed.GetInteriorFaceFiniteElementSpace<0>().GetFaceMesh())>);
   static_assert(
      !is_std_tuple_v<
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
   static_assert(is_std_tuple_v<decltype(interior_fes1)>);
   static_assert(is_std_tuple_v<decltype(boundary_fes1)>);
   static_assert(!is_std_tuple_v<decltype(interior_fes1_0.GetFaceMesh())>);
   static_assert(!is_std_tuple_v<decltype(boundary_fes1_0.GetFaceMesh())>);
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
      make_cartesian_boundary_face_connectivity<1>({num_cells});
   auto interior_faces =
      make_cartesian_interior_face_connectivity<1>({num_cells});
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
   static_assert(is_std_tuple_v<decltype(two_space_same_type_interior_fes)>);
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

   if (!success)
   {
      return 1;
   }

   std::cout << "\nAll mixed integration domain tests passed.\n";
   return 0;
}
