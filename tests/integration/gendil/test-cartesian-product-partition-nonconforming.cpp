// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "cartesian-product-partition-test-helpers.hpp"

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

using namespace gendil;
using namespace gendil::test;

namespace
{

template<
   Integer Dim,
   Integer LocalFaceIndex,
   size_t NumRecords,
   Integer PlusFaceIndex =
      HyperCube<Dim>::GetOppositeFaceIndex(LocalFaceIndex),
   Integer NormalAxis =
      HyperCube<Dim>::GetNormalDimensionIndex(LocalFaceIndex),
   int NormalSign = HyperCube<Dim>::GetNormalSign(LocalFaceIndex),
   class PlusOrientation = Permutation<Dim>>
struct MaterializedNonconformingInteriorConnectivity
{
   using geometry = HyperCube<Dim>;
   using minus_side_type = FaceView<
      std::integral_constant<Integer, LocalFaceIndex>,
      geometry,
      IdentityOrientation<Dim>,
      CanonicalVector<Dim, NormalAxis, NormalSign>,
      NonconformingHyperCubeFaceMap<Dim>>;
   using plus_side_type = FaceView<
      std::integral_constant<Integer, PlusFaceIndex>,
      geometry,
      PlusOrientation,
      CanonicalVector<Dim, NormalAxis, -NormalSign>,
      ConformingFaceMap<Dim>>;
   using face_info_type = GlobalFaceInfo<minus_side_type, plus_side_type>;

   struct Record
   {
      GlobalIndex minus_cell;
      GlobalIndex plus_cell;
      NonconformingHyperCubeFaceMap<Dim> minus_map;
      PlusOrientation plus_orientation;
   };

   std::array<Record, NumRecords> records{};

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return static_cast<GlobalIndex>(NumRecords);
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex face) const
   {
      const auto record = records[face];
      return face_info_type{
         minus_side_type{
            record.minus_cell,
            {},
            {},
            {},
            record.minus_map,
            {}},
         plus_side_type{
            record.plus_cell,
            {},
            record.plus_orientation,
            {},
            {},
            {}}};
   }
};

enum class ManualProductCase
{
   FacedFirst,
   FacedSecond,
   FacedFirstWrongMapBlock,
   FacedFirstUnsignedOrientation,
   FacedSecondWrongMapBlock,
   FacedSecondUnsignedOrientation
};

template<ManualProductCase Case>
auto MakeManualProductFaces()
{
   constexpr bool FacedFirst =
      Case == ManualProductCase::FacedFirst ||
      Case == ManualProductCase::FacedFirstWrongMapBlock ||
      Case == ManualProductCase::FacedFirstUnsignedOrientation;
   constexpr bool WrongMapBlock =
      Case == ManualProductCase::FacedFirstWrongMapBlock ||
      Case == ManualProductCase::FacedSecondWrongMapBlock;
   constexpr bool UnsignedOrientation =
      Case == ManualProductCase::FacedFirstUnsignedOrientation ||
      Case == ManualProductCase::FacedSecondUnsignedOrientation;
   constexpr Integer LocalFaceIndex = FacedFirst ? 3 : 4;
   constexpr Integer PlusFaceIndex = FacedFirst ? 0 : 1;
   constexpr Integer NormalAxis = FacedFirst ? 0 : 1;
   // Every full-dimensional face constant is supplied literally here; this
   // oracle intentionally does not reuse product indexing or lifting code.
   using ProductOrientation = std::conditional_t<
      FacedFirst,
      TensorProductOrientation<Permutation<2>, IdentityOrientation<1>>,
      TensorProductOrientation<IdentityOrientation<1>, Permutation<2>>>;
   using Faces = MaterializedNonconformingInteriorConnectivity<
      3,
      LocalFaceIndex,
      3,
      PlusFaceIndex,
      NormalAxis,
      1,
      ProductOrientation>;

   Point<3> origin{};
   std::array<Real, 3> size{};
   ProductOrientation orientation{};
   if constexpr (FacedFirst)
   {
      origin = WrongMapBlock
         ? Point<3>{0.0, 0.0, 0.2}
         : Point<3>{0.0, 0.2, 0.0};
      size = WrongMapBlock
         ? std::array<Real, 3>{1.0, 1.0, 0.3}
         : std::array<Real, 3>{1.0, 0.3, 1.0};
      orientation = MakeTensorProductOrientation(
         UnsignedOrientation
            ? Permutation<2>{{1, 2}}
            : Permutation<2>{{1, -2}},
         IdentityOrientation<1>{});
      return Faces{{
         typename Faces::Record{1, 2, {origin, size}, orientation},
         typename Faces::Record{5, 6, {origin, size}, orientation},
         typename Faces::Record{9, 10, {origin, size}, orientation}}};
   }
   else
   {
      origin = WrongMapBlock
         ? Point<3>{0.0, 0.2, 0.0}
         : Point<3>{0.0, 0.0, 0.2};
      size = WrongMapBlock
         ? std::array<Real, 3>{1.0, 0.3, 1.0}
         : std::array<Real, 3>{1.0, 1.0, 0.3};
      orientation = MakeTensorProductOrientation(
         IdentityOrientation<1>{},
         UnsignedOrientation
            ? Permutation<2>{{1, 2}}
            : Permutation<2>{{1, -2}});
      return Faces{{
         typename Faces::Record{3, 6, {origin, size}, orientation},
         typename Faces::Record{4, 7, {origin, size}, orientation},
         typename Faces::Record{5, 8, {origin, size}, orientation}}};
   }
}

template<class Space>
void FillNonsymmetricL2Field(const Space& space, Vector& input)
{
   auto view = MakeReadWriteElementTensorView<SerialKernelConfiguration>(
      space, input);
   constexpr Integer p = 2;
   using Points = GaussLobattoLegendrePoints<p + 1>;
   for (GlobalIndex element = 0;
        element < space.GetNumberOfFiniteElements();
        ++element)
   {
      for (LocalIndex k = 0; k <= p; ++k)
      {
         for (LocalIndex j = 0; j <= p; ++j)
         {
            for (LocalIndex i = 0; i <= p; ++i)
            {
               const Real x = Points::GetCoord(i);
               const Real y = Points::GetCoord(j);
               const Real z = Points::GetCoord(k);
               view(i, j, k, element) =
                  Real{1} + Real{13} * element + Real{2} * x +
                  Real{3} * y + Real{5} * z + Real{7} * x * z +
                  Real{11} * y * y;
            }
         }
      }
   }
}

template<class Partition>
Vector ApplyNonconformingForm(
   const Partition& partition,
   const Vector& input)
{
   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   const auto space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto coefficient = MakeCoefficient<"a", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const auto& X)
      {
         return Real{1} + Real{2} * X[0] + Real{3} * X[1] +
                Real{5} * X[2] + Real{7} * X[0] * X[2];
      });
   const auto form = integrate(
      InteriorFacets<"mesh">{},
      coefficient * jump(u) * jump(v));
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(partition));
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5, 5>{});
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(
      form, context, rule);
   return Apply(op, input);
}

template<class Partition>
Vector ApplyNonconformingSparseForm(
   const Partition& partition,
   const Vector& input)
{
   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   const auto space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
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
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(partition));
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5, 5>{});
   const auto coo = GenericAssembly<
      MatrixAssemblyType::COO,
      SerialKernelConfiguration>(form, context, rule);
   return Apply(coo, input);
}

template<bool FacedFirst>
bool TestNumericalNonconformingProduct()
{
   const auto faced = MakeNonconformingFactor();
   const Cartesian1DMesh line_mesh(1.0 / 3.0, 3, 0.25);
   const auto cell_only = MakePartition(MakeCellPart(line_mesh));
   const auto generated = [&]
   {
      if constexpr (FacedFirst)
      {
         return MakeCartesianProductPartition(faced, cell_only);
      }
      else
      {
         return MakeCartesianProductPartition(cell_only, faced);
      }
   }();

   const auto manual_mesh = [&]
   {
      if constexpr (FacedFirst)
      {
         return MakeCartesianProductMesh(
            std::get<0>(faced.CellParts()).mesh,
            line_mesh);
      }
      else
      {
         return MakeCartesianProductMesh(
            line_mesh,
            std::get<0>(faced.CellParts()).mesh);
      }
   }();
   const auto manual_faces = [&]
   {
      if constexpr (FacedFirst)
      {
         return MakeManualProductFaces<ManualProductCase::FacedFirst>();
      }
      else
      {
         return MakeManualProductFaces<ManualProductCase::FacedSecond>();
      }
   }();
   const auto manual = MakePartition(
      MakeCellPart(manual_mesh),
      MakeInteriorFacePart<0, 0>(manual_faces));

   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   const auto generated_space = MakeMixedFiniteElementSpace(
      generated,
      std::tuple{fe},
      std::tuple{ContiguousL2RestrictionSpecification{0}});
   Vector input(generated_space.GetNumberOfFiniteElementDofs());
   input = 0.0;
   FillNonsymmetricL2Field(
      generated_space.template GetCellFiniteElementSpace<0>(),
      input);

   const Vector got = ApplyNonconformingForm(generated, input);
   const Vector expected = ApplyNonconformingForm(manual, input);
   const Vector sparse =
      ApplyNonconformingSparseForm(generated, input);
   bool success = Check(
      VectorNorm(expected) > Real{1.0e-8},
      "nonconforming oracle is nonzero");
   success = CheckVectorClose(
      FacedFirst
         ? "faced-first nonconforming product matches materialized oracle"
         : "faced-second nonconforming product matches materialized oracle",
      got,
      expected) && success;
   success = CheckVectorClose(
      FacedFirst
         ? "faced-first nonconforming RawCOO-derived COO matches matrix-free"
         : "faced-second nonconforming RawCOO-derived COO matches matrix-free",
      sparse,
      got) && success;

   const auto wrong_map_faces = [&]
   {
      if constexpr (FacedFirst)
      {
         return MakeManualProductFaces<
            ManualProductCase::FacedFirstWrongMapBlock>();
      }
      else
      {
         return MakeManualProductFaces<
            ManualProductCase::FacedSecondWrongMapBlock>();
      }
   }();
   const auto unsigned_faces = [&]
   {
      if constexpr (FacedFirst)
      {
         return MakeManualProductFaces<
            ManualProductCase::FacedFirstUnsignedOrientation>();
      }
      else
      {
         return MakeManualProductFaces<
            ManualProductCase::FacedSecondUnsignedOrientation>();
      }
   }();
   const auto wrong_map = MakePartition(
      MakeCellPart(manual_mesh),
      MakeInteriorFacePart<0, 0>(wrong_map_faces));
   const auto unsigned_orientation = MakePartition(
      MakeCellPart(manual_mesh),
      MakeInteriorFacePart<0, 0>(unsigned_faces));
   const Vector wrong_map_output =
      ApplyNonconformingForm(wrong_map, input);
   const Vector unsigned_output =
      ApplyNonconformingForm(unsigned_orientation, input);
   success = Check(
      DifferenceNorm(expected, wrong_map_output) > Real{1.0e-7},
      FacedFirst
         ? "faced-first oracle detects a wrong coordinate block"
         : "faced-second oracle detects a wrong coordinate block") && success;
   success = Check(
      DifferenceNorm(expected, unsigned_output) > Real{1.0e-7},
      FacedFirst
         ? "faced-first oracle detects a lost signed orientation"
         : "faced-second oracle detects a lost signed orientation") && success;

   return success;
}

bool TestH1NonconformingSmoke()
{
   const auto faced = MakeNonconformingFactor();
   const Cartesian1DMesh line_mesh(1.0 / 3.0, 3);
   const auto cell_only = MakePartition(MakeCellPart(line_mesh));
   const auto product =
      MakeCartesianProductPartition(faced, cell_only);
   const auto fe =
      MakeLobattoFiniteElement(FiniteElementOrders<2, 2, 2>{});
   constexpr Integer dofs_per_element = 27;
   const Integer num_elements =
      std::get<0>(product.CellParts()).mesh.GetNumberOfCells();
   std::vector<int> indices(
      static_cast<size_t>(num_elements * dofs_per_element));
   for (Integer i = 0; i < static_cast<Integer>(indices.size()); ++i)
   {
      indices[static_cast<size_t>(i)] = i;
   }
   HostDevicePointer<const int> pointer{};
   pointer.host_pointer = indices.data();
   const IndirectH1RestrictionSpecification restriction{
      pointer,
      static_cast<Integer>(indices.size())};
   const auto completed_restriction = MakeElementDoFRestriction(
      std::get<0>(product.CellParts()).mesh,
      fe,
      restriction);
   const auto space = MakeMixedFiniteElementSpace(
      product,
      std::tuple{fe},
      std::tuple{completed_restriction});
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   const auto form = integrate(
      InteriorFacets<"mesh">{},
      jump(u) * jump(v));
   const auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space),
      MakeIntegrationDomain<"mesh">(product));
   const auto rule =
      MakeIntegrationRule(IntegrationRuleNumPoints<5, 5, 5>{});
   const auto op = MakeGenericOperator<SerialKernelConfiguration>(
      form, context, rule);
   Vector input(space.GetNumberOfFiniteElementDofs());
   input.WriteHostData();
   for (Integer i = 0; i < input.Size(); ++i)
   {
      input[i] = Real{1} + Real{0.01} * i + Real{0.0001} * i * i;
   }
   const Vector output = Apply(op, input);
   const Real norm = VectorNorm(output);
   return Check(
      std::isfinite(norm) && norm > Real{1.0e-8},
      "scalar H1 nonconforming product smoke is finite and nonzero");
}

bool TestVectorRawCOONonconformingValuesAndGradients()
{
   const auto partition = MakeNonconformingFactor();
   const auto linear =
      MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   const auto quadratic =
      MakeLobattoFiniteElement( FiniteElementOrders< 2, 2 >{} );
   const auto vector_fe = MakeVectorFiniteElement( linear, quadratic );
   const auto space = MakeMixedFiniteElementSpace(
      partition,
      std::tuple{ vector_fe },
      DGDirectSumNumbering{} );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   VectorTrialSpace< "vector_u" > u;
   VectorTestSpace< "vector_u" > v;

   const auto check_form = [&] (
      const auto& form,
      const char* label )
   {
      const auto context = MakeWeakFormContext(
         MakeTrialField< "vector_u" >( space ),
         MakeIntegrationDomain< "mesh" >( partition ) );
      const auto raw = GenericAssembly<
         MatrixAssemblyType::RawCOO,
         SerialKernelConfiguration >(
            form,
            context,
            rule );
      const auto coo = FinalizeRawCOOToCOOHost( raw );
      const auto csr = GenericAssembly<
         MatrixAssemblyType::CSR,
         SerialKernelConfiguration >(
            form,
            context,
            rule );
      const auto matrix_free =
         MakeGenericOperator< SerialKernelConfiguration >(
            form,
            context,
            rule );

      bool case_success = true;
      constexpr GlobalIndex local_vector_dofs = 4 + 9;
      constexpr GlobalIndex expected_raw_nnz =
         4 * local_vector_dofs * local_vector_dofs;
      case_success = Check(
         raw.num_rows == GetAlgebraicDofExtent( space ) &&
            raw.num_cols == GetAlgebraicDofExtent( space ) &&
            raw.nnz_raw == expected_raw_nnz,
         std::string( label ) + " RawCOO dimensions or quadrant capacity" ) &&
         case_success;

      Vector input( GetAlgebraicDofExtent( space ) );
      auto* input_data = input.WriteHostData();
      for ( GlobalIndex i = 0; i < input.Size(); ++i )
      {
         input_data[i] =
            Real{0.2} + Real{0.013} * i + Real{0.0007} * i * i;
      }
      const Vector expected = Apply( matrix_free, input );
      const Vector coo_result = Apply( coo, input );
      const Vector csr_result = Apply( csr, input );
      case_success = Check(
         VectorNorm( expected ) > Real{1.0e-8},
         std::string( label ) + " matrix-free action is nonzero" ) &&
         case_success;
      case_success = CheckVectorClose(
         std::string( label ) + " COO matches matrix-free",
         coo_result,
         expected ) && case_success;
      case_success = CheckVectorClose(
         std::string( label ) + " CSR matches matrix-free",
         csr_result,
         expected ) && case_success;
      return case_success;
   };

   const auto value_form = integrate(
      InteriorFacets< "mesh" >{},
      dot( jump( u ), jump( v ) ) );
   const auto gradient_form = integrate(
      InteriorFacets< "mesh" >{},
      inner( jump( grad( u ) ), jump( grad( v ) ) ) );
   const auto combined_form = integrate(
      InteriorFacets< "mesh" >{},
      dot( jump( u ), jump( v ) ) +
         inner( jump( grad( u ) ), jump( grad( v ) ) ) );

   bool success = true;
   success = check_form( value_form, "nonconforming vector value" ) && success;
   success = check_form(
      gradient_form,
      "nonconforming vector gradient" ) && success;
   success = check_form(
      combined_form,
      "nonconforming vector value/gradient" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestNumericalNonconformingProduct<true>() && success;
   success = TestNumericalNonconformingProduct<false>() && success;
   success = TestH1NonconformingSmoke() && success;
   success = TestVectorRawCOONonconformingValuesAndGradients() && success;
   return success ? 0 : 1;
}
