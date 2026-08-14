// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "cartesian-product-partition-test-helpers.hpp"

#include <array>
#include <concepts>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

using namespace gendil;
using namespace gendil::test;

namespace
{

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

struct RuntimeIdentityProductInteriorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 2>,
      geometry,
      Permutation<2>,
      CanonicalVector<2, 0, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 0>,
      geometry,
      Permutation<2>,
      CanonicalVector<2, 0, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 3> records{
      Record{0, 1}, Record{2, 3}, Record{4, 5}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const auto record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          Permutation<2>{{1, 2}},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          Permutation<2>{{1, 2}},
          {},
          {},
          {}}};
   }
};

struct RuntimeIdentitySecondFactorProductInteriorConnectivity
{
   using geometry = HyperCube<2>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, 3>,
      geometry,
      Permutation<2>,
      CanonicalVector<2, 1, 1>,
      ConformingFaceMap<2>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, 1>,
      geometry,
      Permutation<2>,
      CanonicalVector<2, 1, -1>,
      ConformingFaceMap<2>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
   };

   std::array<Record, 2> records{
      Record{0, 2}, Record{1, 3}};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return records.size();
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const auto record = records[face];
      return face_info_type{
         {record.minus_cell,
          {},
          Permutation<2>{{1, 2}},
          {},
          {},
          {}},
         {record.plus_cell,
          {},
          Permutation<2>{{1, 2}},
          {},
          {},
          {}}};
   }
};

struct ProductGeometryProbePoints
{
   GENDIL_HOST_DEVICE
   Real coord(const Integer q) const
   {
      return q == 0 ? Real{0.21} : Real{0.73};
   }
};

template<class FirstCell, class SecondCell>
bool CheckProductCellGeometry(
   const FirstCell& first,
   const SecondCell& second,
   const std::string& label)
{
   const auto factor_qdata =
      MakeTensorProductData(ProductGeometryProbePoints{});
   const auto product_qdata =
      MakeTensorProductData(factor_qdata, factor_qdata);
   const std::array points{
      TensorIndex<2>{GlobalIndex{0}, GlobalIndex{0}},
      TensorIndex<2>{GlobalIndex{1}, GlobalIndex{0}},
      TensorIndex<2>{GlobalIndex{0}, GlobalIndex{1}},
      TensorIndex<2>{GlobalIndex{1}, GlobalIndex{1}}};

   bool success = true;
   for (const auto& point : points)
   {
      typename FirstCell::physical_coordinates first_x{};
      typename SecondCell::physical_coordinates second_x{};
      typename FirstCell::jacobian first_j{};
      typename SecondCell::jacobian second_j{};
      first.GetValuesAndJacobian(
         point, product_qdata, first_x, first_j);
      second.GetValuesAndJacobian(
         point, product_qdata, second_x, second_j);
      for (Integer d = 0; d < 2; ++d)
      {
         success = Check(
            std::abs(first_x[d] - second_x[d]) <=
               product_partition_tolerance,
            label + " physical coordinate") && success;
      }
      success = Check(
         std::abs(std::get<0>(first_j)[0] -
                  std::get<0>(second_j)[0]) <=
            product_partition_tolerance &&
         std::abs(std::get<1>(first_j)[0] -
                  std::get<1>(second_j)[0]) <=
            product_partition_tolerance,
         label + " Jacobian") && success;
   }
   return success;
}

bool TestCellDomainWeightedMass()
{
   const Cartesian1DMesh first_mesh(0.5, 2, -0.25);
   const Cartesian1DMesh second_mesh(0.25, 3, 0.5);
   const auto first = MakePartition(MakeCellPart(first_mesh));
   const auto second = MakePartition(MakeCellPart(second_mesh));
   const auto product_partition =
      MakeCartesianProductPartition(first, second);
   const auto direct_mesh =
      MakeCartesianProductMesh(first_mesh, second_mesh);
   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto direct_space =
      MakeFiniteElementSpace(direct_mesh, fe, L2Restriction{0});
   const auto partition_space = MakeMixedFiniteElementSpace(
      product_partition,
      std::tuple{fe},
      std::tuple{L2Restriction{0}});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto rho = MakeCoefficient<"rho", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                X[0] * X[1];
      });
   const auto form = integrate(Cells<"mesh">{}, rho * u * v);
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5>{});
   const auto direct_context = MakeWeakFormContext(
      MakeTrialField<"u">(direct_space),
      MakeIntegrationDomain<"mesh">(direct_mesh));
   const auto partition_context = MakeWeakFormContext(
      MakeTrialField<"u">(partition_space),
      MakeIntegrationDomain<"mesh">(product_partition));
   const auto direct_op = MakeGenericOperator<SerialKernelConfiguration>(
      form, direct_context, rule);
   const auto partition_op = MakeGenericOperator<SerialKernelConfiguration>(
      form, partition_context, rule);

   Vector input(direct_space.GetNumberOfFiniteElementDofs());
   input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
   {
      input[i] = Real{0.5} + Real{0.017} * i + Real{0.0003} * i * i;
   }
   const Vector expected = Apply(direct_op, input);
   const Vector got = Apply(partition_op, input);
   return Check(
             VectorNorm(expected) > Real{1.0e-8},
             "weighted mass reference is nonzero") &&
          CheckVectorClose(
             "partition cell-domain weighted mass matches direct product mesh",
             got,
             expected);
}

bool TestBoundaryDomainWeightedMass()
{
   const Cartesian1DMesh first_mesh(0.5, 2, -0.25);
   const Cartesian1DMesh second_mesh(0.25, 3, 0.5);
   using SourceFaces = MaterializedBoundaryConnectivity<1, 0, 1>;
   const SourceFaces source_faces{{
      typename SourceFaces::Record{0, 71}}};
   const auto first = MakePartition(
      MakeCellPart(first_mesh),
      MakeBoundaryFacePart<0>(source_faces));
   const auto second = MakePartition(MakeCellPart(second_mesh));
   const auto generated = MakeCartesianProductPartition(first, second);

   using ProductFaces = MaterializedBoundaryConnectivity<2, 0, 3>;
   const ProductFaces product_faces{{
      typename ProductFaces::Record{0, 71},
      typename ProductFaces::Record{2, 71},
      typename ProductFaces::Record{4, 71}}};
   const auto manual = MakePartition(
      MakeCellPart(MakeCartesianProductMesh(first_mesh, second_mesh)),
      MakeBoundaryFacePart<0>(product_faces));

   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated,
      std::tuple{fe},
      std::tuple{L2Restriction{0}});
   const auto manual_space = MakeMixedFiniteElementSpace(
      manual,
      std::tuple{fe},
      std::tuple{L2Restriction{0}});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto coefficient = MakeCoefficient<"a", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                Real{5} * X[0] * X[1];
      });
   const auto form = integrate(
      BoundaryFacets<"mesh">{},
      coefficient * u * v);
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5>{});
   const auto generated_context = MakeWeakFormContext(
      MakeTrialField<"u">(generated_space),
      MakeIntegrationDomain<"mesh">(generated));
   const auto manual_context = MakeWeakFormContext(
      MakeTrialField<"u">(manual_space),
      MakeIntegrationDomain<"mesh">(manual));
   const auto generated_op = MakeGenericOperator<SerialKernelConfiguration>(
      form, generated_context, rule);
   const auto manual_op = MakeGenericOperator<SerialKernelConfiguration>(
      form, manual_context, rule);

   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
   {
      input[i] = Real{0.25} + Real{0.013} * i +
                 Real{0.0007} * i * i;
   }
   const Vector got = Apply(generated_op, input);
   const Vector expected = Apply(manual_op, input);
   return Check(
             VectorNorm(expected) > Real{1.0e-8},
             "boundary weighted-mass oracle is nonzero") &&
          CheckVectorClose(
             "product boundary adapter matches materialized connectivity",
             got,
             expected);
}

bool TestStaticIdentityPlusGeometryFirstFactor()
{
   const Cartesian1DMesh first_mesh(0.7, 2, 0.35);
   const Cartesian1DMesh second_mesh(0.2, 3, -0.45);
   const auto first = MakePartition(
      MakeCellPart(first_mesh),
      MakeInteriorFacePart<0, 0>(
         StaticIdentityLineInteriorConnectivity{}));
   const auto second = MakePartition(MakeCellPart(second_mesh));
   const auto generated = MakeCartesianProductPartition(first, second);

   const auto direct_product_mesh =
      MakeCartesianProductMesh(first_mesh, second_mesh);
   const auto manual = MakePartition(
      MakeCellPart(direct_product_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeIdentityProductInteriorConnectivity{}));

   const auto& generated_faces =
      std::get<0>(generated.InteriorFaceParts()).face_mesh;
   const auto& manual_faces =
      std::get<0>(manual.InteriorFaceParts()).face_mesh;
   const auto generated_face = generated_faces.GetGlobalFaceInfo(1);
   const auto manual_face = manual_faces.GetGlobalFaceInfo(1);
   using GeneratedFace =
      std::remove_cvref_t<decltype(generated_face)>;
   using ManualFace = std::remove_cvref_t<decltype(manual_face)>;
   static_assert(std::same_as<
      typename GeneratedFace::minus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename GeneratedFace::plus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename ManualFace::minus_side_type::orientation_type,
      Permutation<2>>);
   static_assert(std::same_as<
      typename ManualFace::plus_side_type::orientation_type,
      Permutation<2>>);

   auto generated_plus_cell =
      std::get<0>(generated.CellParts()).mesh.GetCell(
         generated_face.PlusSide().GetCellIndex());
   auto manual_plus_cell =
      std::get<0>(manual.CellParts()).mesh.GetCell(
         manual_face.PlusSide().GetCellIndex());
   ApplyOrientationToCell(
      generated_face.PlusSide().GetOrientation(),
      generated_plus_cell);
   ApplyOrientationToCell(
      manual_face.PlusSide().GetOrientation(),
      manual_plus_cell);
   bool success = CheckProductCellGeometry(
      generated_plus_cell,
      manual_plus_cell,
      "static/runtime plus-cell identity equivalence");

   const auto finite_element =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated,
      std::tuple{finite_element},
      std::tuple{L2Restriction{0}});
   const auto manual_space = MakeMixedFiniteElementSpace(
      manual,
      std::tuple{finite_element},
      std::tuple{L2Restriction{0}});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto coefficient =
      MakeCoefficient<"a", PhysicalCoordinates>(
         [] GENDIL_HOST_DEVICE (const auto& X)
         {
            return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                   Real{5} * X[0] * X[1];
         });
   const auto expression =
      coefficient * dot(plus(grad(u)), Normal{}) * jump(v);
   const auto form = integrate(
      InteriorFacets<"mesh">{},
      expression);
   static_assert(requires_plus_side_jacobian_v<decltype(expression)>);
   static_assert(requires_plus_side_jacobian_v<decltype(form)>);

   const auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5>{});
   const auto generated_context = MakeWeakFormContext(
      MakeTrialField<"u">(generated_space),
      MakeIntegrationDomain<"mesh">(generated));
   const auto manual_context = MakeWeakFormContext(
      MakeTrialField<"u">(manual_space),
      MakeIntegrationDomain<"mesh">(manual));
   const auto generated_operator =
      MakeGenericOperator<SerialKernelConfiguration>(
         form, generated_context, integration_rule);
   const auto manual_operator =
      MakeGenericOperator<SerialKernelConfiguration>(
         form, manual_context, integration_rule);

   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
   {
      input[i] = Real{0.4} + Real{0.019} * i +
                 Real{0.0008} * i * i;
   }
   const Vector got = Apply(generated_operator, input);
   const Vector expected = Apply(manual_operator, input);
   const Real absolute_error = DifferenceNorm(got, expected);
   const Real reference_norm = VectorNorm(expected);
   const Real relative_error = absolute_error /
      std::max(reference_norm, Real{1.0e-30});
   std::cout
      << "first-factor static/runtime plus-geometry operator: absolute="
      << absolute_error
      << " relative=" << relative_error
      << " reference=" << reference_norm << '\n';
   if (absolute_error >
       product_partition_tolerance * std::max(Real{1}, reference_norm))
   {
      std::cerr
         << "static/runtime plus-geometry operator error: absolute="
         << absolute_error
         << " relative=" << relative_error << '\n';
   }
   success = Check(
      reference_norm > Real{1.0e-8},
      "forced plus-geometry runtime oracle is nonzero") && success;
   success = CheckVectorClose(
      "static identity plus geometry matches runtime identity oracle",
      got,
      expected) && success;
   return success;
}

bool TestStaticIdentityPlusGeometrySecondFactor()
{
   const Cartesian1DMesh first_mesh(0.2, 2, -0.45);
   const Cartesian1DMesh second_mesh(0.7, 2, 0.35);
   const auto first = MakePartition(MakeCellPart(first_mesh));
   const auto second = MakePartition(
      MakeCellPart(second_mesh),
      MakeInteriorFacePart<0, 0>(
         StaticIdentityLineInteriorConnectivity{}));
   const auto generated = MakeCartesianProductPartition(first, second);

   const auto direct_product_mesh =
      MakeCartesianProductMesh(first_mesh, second_mesh);
   const auto manual = MakePartition(
      MakeCellPart(direct_product_mesh),
      MakeInteriorFacePart<0, 0>(
         RuntimeIdentitySecondFactorProductInteriorConnectivity{}));

   const auto& generated_faces =
      std::get<0>(generated.InteriorFaceParts()).face_mesh;
   const auto& manual_faces =
      std::get<0>(manual.InteriorFaceParts()).face_mesh;
   const auto generated_face = generated_faces.GetGlobalFaceInfo(1);
   const auto manual_face = manual_faces.GetGlobalFaceInfo(1);
   using GeneratedFace = std::remove_cvref_t<decltype(generated_face)>;
   using ManualFace = std::remove_cvref_t<decltype(manual_face)>;
   static_assert(
      GeneratedFace::minus_side_type::local_face_index_type::value == 3);
   static_assert(
      GeneratedFace::plus_side_type::local_face_index_type::value == 1);
   static_assert(GeneratedFace::minus_side_type::normal_type::index == 1);
   static_assert(GeneratedFace::minus_side_type::normal_type::sign == 1);
   static_assert(GeneratedFace::plus_side_type::normal_type::index == 1);
   static_assert(GeneratedFace::plus_side_type::normal_type::sign == -1);
   static_assert(std::same_as<
      typename GeneratedFace::minus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename GeneratedFace::plus_side_type::orientation_type,
      IdentityOrientation<2>>);
   static_assert(std::same_as<
      typename ManualFace::minus_side_type::orientation_type,
      Permutation<2>>);
   static_assert(std::same_as<
      typename ManualFace::plus_side_type::orientation_type,
      Permutation<2>>);

   bool success = Check(
      generated_face.MinusSide().GetCellIndex() == 1 &&
         generated_face.PlusSide().GetCellIndex() == 3 &&
         manual_face.MinusSide().GetCellIndex() == 1 &&
         manual_face.PlusSide().GetCellIndex() == 3,
      "second-factor identity oracle uses literal product cell indices");

   auto generated_plus_cell =
      std::get<0>(generated.CellParts()).mesh.GetCell(
         generated_face.PlusSide().GetCellIndex());
   auto manual_plus_cell =
      std::get<0>(manual.CellParts()).mesh.GetCell(
         manual_face.PlusSide().GetCellIndex());
   ApplyOrientationToCell(
      generated_face.PlusSide().GetOrientation(),
      generated_plus_cell);
   ApplyOrientationToCell(
      manual_face.PlusSide().GetOrientation(),
      manual_plus_cell);
   success = CheckProductCellGeometry(
      generated_plus_cell,
      manual_plus_cell,
      "second-factor static/runtime plus-cell identity equivalence") &&
      success;

   const auto finite_element =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated,
      std::tuple{finite_element},
      std::tuple{L2Restriction{0}});
   const auto manual_space = MakeMixedFiniteElementSpace(
      manual,
      std::tuple{finite_element},
      std::tuple{L2Restriction{0}});

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto coefficient =
      MakeCoefficient<"a", PhysicalCoordinates>(
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

   const auto integration_rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5>{});
   const auto generated_context = MakeWeakFormContext(
      MakeTrialField<"u">(generated_space),
      MakeIntegrationDomain<"mesh">(generated));
   const auto manual_context = MakeWeakFormContext(
      MakeTrialField<"u">(manual_space),
      MakeIntegrationDomain<"mesh">(manual));
   const auto generated_operator =
      MakeGenericOperator<SerialKernelConfiguration>(
         form, generated_context, integration_rule);
   const auto manual_operator =
      MakeGenericOperator<SerialKernelConfiguration>(
         form, manual_context, integration_rule);

   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
   {
      input[i] = Real{0.31} + Real{0.023} * i +
                 Real{0.0011} * i * i;
   }
   const Vector got = Apply(generated_operator, input);
   const Vector expected = Apply(manual_operator, input);
   const Real absolute_error = DifferenceNorm(got, expected);
   const Real reference_norm = VectorNorm(expected);
   const Real relative_error = absolute_error /
      std::max(reference_norm, Real{1.0e-30});
   std::cout
      << "second-factor static/runtime plus-geometry operator: absolute="
      << absolute_error
      << " relative=" << relative_error
      << " reference=" << reference_norm << '\n';
   if (absolute_error >
       product_partition_tolerance * std::max(Real{1}, reference_norm))
   {
      std::cerr
         << "second-factor static/runtime plus-geometry operator error: "
         << "absolute=" << absolute_error
         << " relative=" << relative_error << '\n';
   }
   success = Check(
      reference_norm > Real{1.0e-8},
      "second-factor forced plus-geometry runtime oracle is nonzero") &&
      success;
   success = CheckVectorClose(
      "second-factor static identity plus geometry matches runtime oracle",
      got,
      expected) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestCellDomainWeightedMass() && success;
   success = TestBoundaryDomainWeightedMass() && success;
   success = TestStaticIdentityPlusGeometryFirstFactor() && success;
   success = TestStaticIdentityPlusGeometrySecondFactor() && success;
   return success ? 0 : 1;
}
