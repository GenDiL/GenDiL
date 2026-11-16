// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Utilities/tensorproductdata.hpp>
#include <gendil/Meshes/Geometries/hypercube.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetcontext.hpp>
#include <gendil/Utilities/MathHelperFunctions/computeinverseanddeterminant.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/getweight.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp>
#include <gendil/FiniteElementMethod/WeakForm/weakform.hpp>
#include <gendil/Algebra/Vector/vector.hpp>
#include <gendil/Utilities/KernelContext/kernelconfiguration.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

// Declare customization after the DSL headers: evaluation must find it through
// argument-dependent lookup, without requiring point-taking facet members,
// zero-argument point selectors, or a stored-normal member name.
namespace facet_normal_custom_test
{
struct Facet {};
struct ReferenceFacet {};

GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal(const ReferenceFacet&)
{
   return CanonicalVector<2, 0, 1>{};
}

struct Side
{
   std::array<Real, 2> X;
   std::array<Real, 2> normal_value;
   Real inverse_normal_cell_size;
};

struct Point
{
   std::array<Real, 2> X;
   std::array<Real, 2> normal_value;
};

GENDIL_HOST_DEVICE
constexpr auto GetMinusSide(const Facet&, const Point& point)
{
   return Side{point.X, point.normal_value, 4.};
}

GENDIL_HOST_DEVICE
constexpr auto GetPlusSide(const Facet&, const Point& point)
{
   return Side{point.X, point.normal_value, 9.};
}

template<class PointContext>
GENDIL_HOST_DEVICE
constexpr auto GetFacetNormal(const Facet&, const PointContext& point)
{
   return std::array<Real, 2>{-point.normal_value[0], -point.normal_value[1]};
}
} // namespace facet_normal_custom_test

namespace facet_side_reference_test
{
struct Facet
{
   Real minus_value;
   Real plus_value;
};
struct Point {};

GENDIL_HOST_DEVICE
constexpr Real& GetMinusSide(Facet& facet, const Point&)
{
   return facet.minus_value;
}

GENDIL_HOST_DEVICE
constexpr Real& GetPlusSide(Facet& facet, const Point&)
{
   return facet.plus_value;
}
} // namespace facet_side_reference_test

// The coefficient descriptor must discover this overload after its definition,
// without requiring a member hook or using the deliberately different raw map.
namespace coefficient_jacobian_custom_test
{
struct Context {};
struct Point
{
   Real inv_J_mesh;
   Real callback_inverse;
};
struct PointWithoutRaw
{
   Real callback_inverse;
};

template<class PointContext>
   requires (std::is_same_v<std::remove_cv_t<PointContext>, Point> ||
             std::is_same_v<std::remove_cv_t<PointContext>, PointWithoutRaw>)
GENDIL_HOST_DEVICE
constexpr const Real& GetCoefficientJacobianInverse(const Context&, PointContext& point)
{
   return point.callback_inverse;
}
} // namespace coefficient_jacobian_custom_test

namespace local_facet_geometry_test
{
using Matrix2 = std::array<std::array<Real, 2>, 2>;
using Matrix3 = std::array<std::array<Real, 3>, 3>;
using local_facet_detail::CanonicalInverseJacobian;

template<class Orientation>
struct NeighborFace
{
   Orientation orientation;
   GENDIL_HOST_DEVICE
   const auto& GetOrientation() const { return orientation; }
   GENDIL_HOST_DEVICE
   GlobalIndex GetCellIndex() const { return 7; }
};

struct CurrentFace
{
   GENDIL_HOST_DEVICE
   GlobalIndex GetCellIndex() const { return 13; }
};

template<int Sign, class Orientation>
using GeometryFace = LocalFaceInfo<CurrentFace, NeighborFace<Orientation>,
   std::integral_constant<int, Sign>>;

template<int Sign, class Orientation>
GENDIL_HOST_DEVICE
auto MakeGeometryFacet(const Orientation& orientation)
{
   return LocalFacetContext{GeometryFace<Sign, Orientation>{{}, {orientation}, {}}};
}

template<class Face>
struct DerivedLocalFacet : LocalFacetContext<Face>
{
   using LocalFacetContext<Face>::LocalFacetContext;
};

template<class Facet, class Point>
concept CanGetFacetNormal = requires(const Facet& facet, const Point& point)
{
   GetFacetNormal(facet, point);
};

template<class Facet, class Point>
concept HasMemberNormal = requires(const Facet& facet, const Point& point)
{
   facet.Normal(point);
};

template<class Facet, class Point>
concept CanSelectPoint = requires(Facet&& facet, const Point& point)
{
   GetMinusSide(std::forward<Facet>(facet), point);
   GetPlusSide(std::forward<Facet>(facet), point);
};

template<class Facet, class Point>
concept HasMemberPointSides = requires(const Facet& facet, const Point& point)
{
   facet.MinusSide(point);
   facet.PlusSide(point);
};

template<class Facet, class Jacobian>
concept HasMemberCoefficientInverse = requires(const Facet& facet, const Jacobian& raw)
{
   facet.GetCoefficientJacobianInverse(raw);
};

template<class Facet, class Point>
concept CanGetCoefficientInverse = requires(const Facet& facet, Point&& point)
{
   GetCoefficientJacobianInverse(facet, std::forward<Point>(point));
};

template<class Facet, class Point>
GENDIL_HOST_DEVICE
decltype(auto) CoefficientInverse(const Facet& facet, const Point& point)
{
   return CoefficientInputGetter<JacobianInverse>::Get(
      Empty{}, Empty{}, Empty{}, facet, point, Empty{});
}

// Naming the local slots must not add storage or change their layout.
template<class Context>
concept HasTraceSides = requires(const Context& context)
{
   context.MinusSide();
   context.PlusSide();
};

template<class Context>
concept HasLocalSides = requires(const Context& context)
{
   context.CurrentSide();
   context.NeighborSide();
};

template<class Context>
concept HasTraceGeometryStorage =
   requires(const Context& context) { context.inv_J_mesh_minus; } ||
   requires(const Context& context) { context.inv_J_mesh_plus; } ||
   requires(const Context& context) { context.inverse_normal_cell_size_minus; } ||
   requires(const Context& context) { context.inverse_normal_cell_size_plus; };

struct StoredSidePair
{
   Real minus_value;
   Real plus_value;

   GENDIL_HOST_DEVICE
   constexpr const Real& MinusSide() const { return minus_value; }
   GENDIL_HOST_DEVICE
   constexpr const Real& PlusSide() const { return plus_value; }
};

struct BoundarySidePair : StoredSidePair
{
   static constexpr bool is_physical_boundary = true;
};

struct LocalSidePair : StoredSidePair
{
   Real inv_J_mesh_current;
};

// Obsolete point-taking members must neither enable selection nor override
// the zero-argument selectors of canonical point storage.
struct MemberOnlySideFacet
{
   Real minus_value;
   Real plus_value;

   template<class Point>
   GENDIL_HOST_DEVICE
   constexpr const Real& MinusSide(const Point&) const { return minus_value; }
   template<class Point>
   GENDIL_HOST_DEVICE
   constexpr const Real& PlusSide(const Point&) const { return plus_value; }
};

static_assert(!CanSelectPoint<MemberOnlySideFacet&, Empty>);
static_assert(!CanSelectPoint<const MemberOnlySideFacet&, Empty>);
static_assert(!CanSelectPoint<MemberOnlySideFacet, Empty>);
static_assert(CanSelectPoint<MemberOnlySideFacet&, StoredSidePair>);
static_assert(CanSelectPoint<const MemberOnlySideFacet&, StoredSidePair>);
static_assert(CanSelectPoint<MemberOnlySideFacet, StoredSidePair>);
static_assert(!CanSelectPoint<MemberOnlySideFacet&, BoundarySidePair>);
static_assert(!CanSelectPoint<MemberOnlySideFacet&, LocalSidePair>);
static_assert(!CanSelectPoint<Empty, BoundarySidePair>);
static_assert(!CanSelectPoint<Empty, LocalSidePair>);
static_assert(CanSelectPoint<facet_side_reference_test::Facet&,
   facet_side_reference_test::Point>);
static_assert(!CanSelectPoint<const facet_side_reference_test::Facet&,
   facet_side_reference_test::Point>);
static_assert(!CanSelectPoint<facet_side_reference_test::Facet,
   facet_side_reference_test::Point>);
static_assert(CanSelectPoint<facet_normal_custom_test::Facet,
   facet_normal_custom_test::Point>);
static_assert(!HasTraceSides<facet_normal_custom_test::Point>);
static_assert(!HasMemberPointSides<facet_normal_custom_test::Facet,
   facet_normal_custom_test::Point>);
static_assert(!CanSelectPoint<facet_normal_custom_test::Facet,
   facet_normal_custom_test::Side>);

GENDIL_HOST_DEVICE
constexpr bool CheckSideAccessors(Real seed)
{
   MemberOnlySideFacet facet{seed, seed + 1.};
   const auto& const_facet = facet;
   const StoredSidePair point{seed + 2., seed + 3.};
   static_assert(std::is_same_v<decltype(GetMinusSide(facet, point)), const Real&>);
   static_assert(std::is_same_v<decltype(GetPlusSide(facet, point)), const Real&>);
   static_assert(std::is_same_v<decltype(GetMinusSide(const_facet, point)), const Real&>);
   static_assert(std::is_same_v<decltype(GetPlusSide(std::move(facet), point)), const Real&>);
   const bool ok = &GetMinusSide(facet, point) == &point.minus_value &&
      &GetPlusSide(facet, point) == &point.plus_value &&
      &GetMinusSide(const_facet, point) == &point.minus_value &&
      &GetPlusSide(const_facet, point) == &point.plus_value &&
      &GetMinusSide(std::move(facet), point) == &point.minus_value &&
      &GetPlusSide(std::move(facet), point) == &point.plus_value &&
      &GetMinusSide(Empty{}, point) == &point.minus_value &&
      &GetPlusSide(Empty{}, point) == &point.plus_value;
   facet_side_reference_test::Facet custom{seed, seed + 1.};
   const facet_side_reference_test::Point custom_point{};
   static_assert(std::is_same_v<decltype(GetMinusSide(custom, custom_point)), Real&>);
   static_assert(std::is_same_v<decltype(GetPlusSide(custom, custom_point)), Real&>);
   const bool references = &GetMinusSide(custom, custom_point) == &custom.minus_value &&
      &GetPlusSide(custom, custom_point) == &custom.plus_value;
   GetMinusSide(custom, custom_point) += 4.;
   GetPlusSide(custom, custom_point) += 5.;
   return ok && references && custom.minus_value == seed + 4. &&
      custom.plus_value == seed + 6. && facet.minus_value == seed && facet.plus_value == seed + 1.;
}

static_assert(CheckSideAccessors(2.));

template<Integer Dim, class CurrentJacobian, class NeighborJacobian>
constexpr bool CheckLocalContextLayout()
{
   using Coordinates = std::array<Real, Dim>;
   using Local = LocalFacetQuadraturePointData<Integer, Coordinates,
      CurrentJacobian, NeighborJacobian, Coordinates>;
   using Global = TwoSidedFacetQuadraturePointContext<Integer, Coordinates,
      CurrentJacobian, NeighborJacobian, Coordinates>;
   using Face = GeometryFace<1, IdentityOrientation<Dim>>;
   using Facet = LocalFacetContext<Face>;
   using OppositeFacet = LocalFacetContext<GeometryFace<-1, IdentityOrientation<Dim>>>;
   using Boundary = BoundaryFacetQuadraturePointContext<Integer, Coordinates,
      CurrentJacobian, Coordinates>;
   using FacetWithCell = LocalFacetContext<Face, std::array<Real, 3>>;
   using OriginalFacetWithCell = FacetContext<Face, std::array<Real, 3>>;
   using Side = decltype(std::declval<const Local&>().CurrentSide());
   using Selected = decltype(GetMinusSide(
      std::declval<const Facet&>(), std::declval<const Local&>()));
   static_assert(std::is_same_v<Selected, Side>);
   static_assert(sizeof(Facet) == sizeof(Face) && alignof(Facet) == alignof(Face));
   static_assert(sizeof(FacetWithCell) == sizeof(OriginalFacetWithCell));
   static_assert(alignof(FacetWithCell) == alignof(OriginalFacetWithCell));
   static_assert(!requires(const Facet& facet) { facet.second_cell; });
   static_assert(!requires(const Local& point) { point.neighbor_orientation; });
   static_assert(!requires(const Selected& point) { point.neighbor_orientation; });
   static_assert(CanSelectPoint<Facet&, Local> && CanSelectPoint<const Facet&, Local>);
   static_assert(CanSelectPoint<Facet, Local> && CanSelectPoint<const Facet, Local>);
   static_assert(CanSelectPoint<OppositeFacet&, Local> && CanSelectPoint<OppositeFacet, Local>);
   static_assert(CanSelectPoint<DerivedLocalFacet<Face>, Local>);
   static_assert(CanSelectPoint<FacetWithCell, Local>);
   static_assert(!HasMemberPointSides<Facet, Local>);
   static_assert(!HasMemberCoefficientInverse<Facet, CurrentJacobian>);
   static_assert(!CanSelectPoint<const Facet&, Side>);
   static_assert(!CanSelectPoint<const Facet&, Selected>);
   static_assert(!CanSelectPoint<Empty, Local>);
   static_assert(!CanSelectPoint<ElementContext<Empty>, Local>);
   static_assert(!CanSelectPoint<Face, Local>);
   static_assert(!CanSelectPoint<Facet, Global>);
   static_assert(!CanSelectPoint<Facet, Boundary>);
   static_assert(!CanSelectPoint<ElementContext<Empty>, Boundary>);
   static_assert(!CanSelectPoint<ElementContext<Empty>, Selected>);
   static_assert(CanSelectPoint<const Empty&, Global>);
   static_assert(CanSelectPoint<ElementContext<Empty>, Global>);
   static_assert(CanGetFacetNormal<Facet, Local> && CanGetFacetNormal<Facet, Selected>);
   static_assert(CanGetFacetNormal<FacetWithCell, Local>);
   static_assert(CanGetFacetNormal<DerivedLocalFacet<Face>, Local>);
   static_assert(CanGetFacetNormal<Empty, Local>);
   static_assert(CanGetFacetNormal<ElementContext<Empty>, Local>);
   static_assert(CanGetFacetNormal<Empty, Global>);
   static_assert(CanGetFacetNormal<ElementContext<Empty>, Global>);
   static_assert(!HasMemberNormal<Facet, Local> && !HasMemberNormal<Facet, Selected>);
   static_assert(HasTraceSides<Facet> && !HasTraceSides<Selected>);
   static_assert(std::is_standard_layout_v<Local> && std::is_standard_layout_v<Global>);
   static_assert(std::is_trivially_copyable_v<Local> && std::is_trivially_copyable_v<Global>);
   static_assert(!std::is_same_v<Local, Global>);
   static_assert(HasLocalSides<Local>);
   static_assert(!requires(const Local& context) { context.MinusSide(); });
   static_assert(!requires(const Local& context) { context.PlusSide(); });
   static_assert(!HasTraceGeometryStorage<Local> && HasTraceGeometryStorage<Global>);
   static_assert(sizeof(Local) == sizeof(Global));
   static_assert(alignof(Local) == alignof(Global));
   static_assert(offsetof(Local, quad_index) == offsetof(Global, quad_index));
   static_assert(offsetof(Local, X) == offsetof(Global, X));
   static_assert(offsetof(Local, inv_J_mesh_current) == offsetof(Global, inv_J_mesh_minus));
   static_assert(offsetof(Local, inv_J_mesh_neighbor) == offsetof(Global, inv_J_mesh_plus));
   static_assert(offsetof(Local, weight) == offsetof(Global, weight));
   static_assert(offsetof(Local, det_J_facet) == offsetof(Global, det_J_facet));
   static_assert(offsetof(Local, inverse_normal_cell_size_current) ==
      offsetof(Global, inverse_normal_cell_size_minus));
   static_assert(offsetof(Local, inverse_normal_cell_size_neighbor) ==
      offsetof(Global, inverse_normal_cell_size_plus));
   static_assert(offsetof(Local, physical_normal) == offsetof(Global, physical_normal));
   return true;
}

static_assert(CheckLocalContextLayout<1, Real, Real>());
static_assert(CheckLocalContextLayout<2, std::array<Real, 2>, std::array<Real, 2>>());
static_assert(CheckLocalContextLayout<2, Matrix2, Matrix2>());
static_assert(CheckLocalContextLayout<3, Matrix3, Matrix3>());
static_assert(CheckLocalContextLayout<2, Matrix2, Empty>());

template<class Fields>
concept HasTraceFieldStorage =
   requires(const Fields& fields) { fields.minus_fields; } ||
   requires(const Fields& fields) { fields.plus_fields; };

template<class CurrentFields, class NeighborFields>
constexpr bool CheckLocalFieldLayout()
{
   using Local = LocalFacetFields<CurrentFields, NeighborFields>;
   using Traces = FacetTraceFields<CurrentFields, NeighborFields>;
   static_assert(std::is_standard_layout_v<Local> && std::is_standard_layout_v<Traces>);
   static_assert(std::is_trivially_copyable_v<Local> && std::is_trivially_copyable_v<Traces>);
   static_assert(!HasTraceFieldStorage<Local> && HasTraceFieldStorage<Traces>);
   static_assert(sizeof(Local) == sizeof(Traces));
   static_assert(alignof(Local) == alignof(Traces));
   static_assert(offsetof(Local, current_fields) == offsetof(Traces, minus_fields));
   static_assert(offsetof(Local, neighbor_fields) == offsetof(Traces, plus_fields));
   return true;
}

static_assert(CheckLocalFieldLayout<std::array<Real, 2>, Matrix2>());
static_assert(CheckLocalFieldLayout<Matrix2, Empty>());

template<class Left, class Right>
GENDIL_HOST_DEVICE
bool EqualMatrix(const Left& left, const Right& right)
{
   bool equal = true;
   ConstexprLoop<static_num_rows_v<Left>>([&](auto row)
   {
      ConstexprLoop<static_num_cols_v<Left>>([&](auto column)
      {
         equal = equal && GetValue(left, row, column) == GetValue(right, row, column);
      });
   });
   return equal;
}

GENDIL_HOST_DEVICE
bool Check2D(Real seed, int direction)
{
   const Matrix2 raw{{{seed, 2.}, {3., 5.}}};
   const Permutation<2> orientation{{-2*direction, direction}};
   // Native axis 0 is reversed reference axis 1; native axis 1 is reference axis 0.
   const Matrix2 expected{{{-2.*direction, seed*direction}, {-5.*direction, 3.*direction}}};
   const auto actual = CanonicalInverseJacobian(raw, orientation);
   static_assert(std::is_same_v<decltype(CanonicalInverseJacobian(raw, IdentityOrientation<2>{})),
      const Matrix2&>);
   const auto& identity = CanonicalInverseJacobian(raw, IdentityOrientation<2>{});
   return EqualMatrix(actual, expected) && &identity == &raw;
}

GENDIL_HOST_DEVICE
bool Check3D(Real seed, int direction)
{
   const Matrix3 raw{{{seed, 2., 3.}, {5., 7., 11.}, {13., 17., 19.}}};
   const Permutation<3> orientation{{3, -direction, 2*direction}};
   const Matrix3 expected{{{3., -seed*direction, 2.*direction},
                           {11., -5.*direction, 7.*direction},
                           {19., -13.*direction, 17.*direction}}};
   SerialRecursiveArray<Real, 3, 3> recursive{};
   ConstexprLoop<3>([&](auto row)
   {
      ConstexprLoop<3>([&](auto column) { recursive(row, column) = raw[row][column]; });
   });
   return EqualMatrix(CanonicalInverseJacobian(raw, orientation), expected) &&
      EqualMatrix(CanonicalInverseJacobian(recursive, orientation), expected);
}

GENDIL_HOST_DEVICE
bool CheckScalarContext(Real seed)
{
   using Context = LocalFacetQuadraturePointData<Integer, std::array<Real, 1>,
      Real, Real, std::array<Real, 1>>;
   Context raw{0, {.5}, seed, 3., 1., 1., 2., 3., {-1.}};
   const IdentityOrientation<1> orientation{};
   const auto facet = MakeGeometryFacet<-1>(orientation);
   const auto minus = GetMinusSide(facet, raw);
   const auto plus = GetPlusSide(facet, raw);
   static_assert(std::is_same_v<decltype(CoefficientInverse(facet, minus)), const Real&>);
   return raw.inv_J_mesh_current == seed && minus.inv_J_mesh == 3. && plus.inv_J_mesh == seed &&
      &CoefficientInverse(facet, minus) == &minus.inv_J_mesh &&
      &CoefficientInverse(facet, plus) == &plus.inv_J_mesh &&
      raw.physical_normal[0] == -1. && GetFacetNormal(facet, raw)[0] == -1. &&
      GetFacetNormal(facet, minus)[0] == -1. && GetFacetNormal(facet, plus)[0] == -1.;
}

GENDIL_HOST_DEVICE
bool CheckCompactJacobians(Real seed, int direction)
{
   using Diagonal2 = std::array<Real, 2>;
   using Diagonal3 = std::array<Real, 3>;
   const Diagonal2 diagonal2{seed, 3.};
   const Diagonal3 diagonal3{5., seed, 7.};
   const Permutation<1> scalar_orientation{{direction}};
   const Permutation<2> orientation2{{direction, 2*direction}};
   const Permutation<3> orientation3{{direction, -2, 3*direction}};
   const auto scalar = CanonicalInverseJacobian(seed, scalar_orientation);
   const auto result2 = CanonicalInverseJacobian(diagonal2, orientation2);
   const auto result3 = CanonicalInverseJacobian(diagonal3, orientation3);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(scalar)>, Real>);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(result2)>, Diagonal2>);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(result3)>, Diagonal3>);
   const auto identity2 = CanonicalInverseJacobian(diagonal2, Permutation<2>{{1, 2}});
   const auto identity3 = CanonicalInverseJacobian(diagonal3, Permutation<3>{{1, 2, 3}});
   const auto product = std::make_tuple(seed, diagonal2, diagonal3);
   const auto product_orientation = MakeTensorProductOrientation(
      scalar_orientation, orientation2, orientation3);
   const auto product_result = CanonicalInverseJacobian(product, product_orientation);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(product_result)>,
      std::remove_cvref_t<decltype(product)>>);
   return scalar == seed*direction && result2[0] == seed*direction && result2[1] == 3.*direction &&
      result3[0] == 5.*direction && result3[1] == -seed && result3[2] == 7.*direction &&
      identity2[0] == seed && identity2[1] == 3. &&
      identity3[0] == 5. && identity3[1] == seed && identity3[2] == 7. &&
      std::get<0>(product_result) == seed*direction &&
      std::get<1>(product_result)[0] == seed*direction && std::get<1>(product_result)[1] == 3.*direction &&
      std::get<2>(product_result)[0] == 5.*direction && std::get<2>(product_result)[1] == -seed &&
      std::get<2>(product_result)[2] == 7.*direction;
}

struct DiagonalCallback
{
   // Deliberately accepts only compact storage, so dense promotion would fail.
   GENDIL_HOST_DEVICE
   Real operator()(const std::array<Real, 2>& inverse) const
   {
      return inverse[0] + 2.*inverse[1];
   }
};

GENDIL_HOST_DEVICE
bool CheckDiagonalCallback(Real seed, int direction)
{
   using Diagonal = std::array<Real, 2>;
   using Context = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Diagonal, Diagonal, std::array<Real, 2>>;
   Context raw{0, {.25, .75}, {seed, 2.}, {3., 5.}, 1., 1., 2., 3., {-1., 0.}};
   const Permutation<2> orientation{{direction, -2}};
   const auto facet = MakeGeometryFacet<-1>(orientation);
   const auto minus = GetMinusSide(facet, raw);
   const auto plus = GetPlusSide(facet, raw);
   const auto coefficient = MakeCoefficient<"diagonal_inverse", JacobianInverse>(DiagonalCallback{});
   static_assert(std::is_same_v<decltype(CoefficientInverse(facet, minus)), Diagonal>);
   return coefficient(Empty{}, Empty{}, Empty{}, facet, minus, Empty{}) == 3.*direction - 10. &&
      coefficient(Empty{}, Empty{}, Empty{}, facet, plus, Empty{}) == seed*direction - 4. &&
      raw.inv_J_mesh_current[0] == seed && raw.inv_J_mesh_current[1] == 2. &&
      minus.inv_J_mesh[0] == 3. && minus.inv_J_mesh[1] == 5. &&
      plus.inv_J_mesh[0] == seed && plus.inv_J_mesh[1] == 2.;
}

GENDIL_HOST_DEVICE
bool CheckProduct(Real seed, int direction)
{
   const std::array<Real, 1> diagonal{seed};
   const Matrix2 block{{{2., 3.}, {5., 7.}}};
   const auto raw = std::make_tuple(diagonal, block);
   const auto orientation = MakeTensorProductOrientation(
      IdentityOrientation<1>{}, Permutation<2>{{-2*direction, direction}});
   const auto actual = CanonicalInverseJacobian(raw, orientation);
   const auto identity_orientation = MakeTensorProductOrientation(
      IdentityOrientation<1>{}, IdentityOrientation<2>{});
   static_assert(std::is_same_v<decltype(CanonicalInverseJacobian(raw, identity_orientation)),
      const std::remove_cvref_t<decltype(raw)>&>);
   const auto& identity = CanonicalInverseJacobian(raw, identity_orientation);
   using Jacobian = std::remove_cvref_t<decltype(raw)>;
   using Point = LocalFacetQuadraturePointData<Integer, std::array<Real, 3>,
      Jacobian, Jacobian, std::array<Real, 3>>;
   const Point point{0, {.2, .4, .6}, raw, raw, 1., 1., 2., 3., {-1., 0., 0.}};
   const auto facet = MakeGeometryFacet<-1>(orientation);
   const auto selected = GetMinusSide(facet, point);
   const auto callback = CoefficientInverse(facet, selected);
   const auto direct = GetCoefficientJacobianInverse(facet, selected);
   const auto identity_facet = MakeGeometryFacet<-1>(identity_orientation);
   static_assert(std::is_same_v<decltype(CoefficientInverse(identity_facet, selected)),
      const Jacobian&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(identity_facet, selected)),
      const Jacobian&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(
      identity_facet, GetMinusSide(identity_facet, point))), Jacobian>);
   const auto owned_identity = GetCoefficientJacobianInverse(
      identity_facet, GetMinusSide(identity_facet, point));
   const auto owned_transformed = GetCoefficientJacobianInverse(
      facet, GetMinusSide(facet, point));
   const Matrix2 expected{{{-3.*direction, 2.*direction}, {-7.*direction, 5.*direction}}};
   return std::get<0>(actual)[0] == seed && EqualMatrix(std::get<1>(actual), expected) &&
      std::get<0>(callback)[0] == seed && EqualMatrix(std::get<1>(callback), expected) &&
      std::get<0>(direct)[0] == seed && EqualMatrix(std::get<1>(direct), expected) &&
      std::get<0>(owned_identity)[0] == seed && EqualMatrix(std::get<1>(owned_identity), block) &&
      std::get<0>(owned_transformed)[0] == seed && EqualMatrix(std::get<1>(owned_transformed), expected) &&
      &identity == &raw && &CoefficientInverse(identity_facet, selected) == &selected.inv_J_mesh;
}

struct MemberOnlyInversePoint
{
   Real callback_inverse;
   GENDIL_HOST_DEVICE
   constexpr const Real& GetCoefficientJacobianInverse() const { return callback_inverse; }
};

struct MappedPointWithObsoleteMember : MemberOnlyInversePoint
{
   Real inv_J_mesh;
};

struct MemberOnlyInverseFacet
{
   Real callback_inverse;
   GENDIL_HOST_DEVICE
   constexpr const Real& GetCoefficientJacobianInverse(const Real&) const { return callback_inverse; }
};

static_assert(!CanGetCoefficientInverse<Empty, MemberOnlyInversePoint&>);
static_assert(!CanGetCoefficientInverse<Empty, const MemberOnlyInversePoint&>);
static_assert(!CanGetCoefficientInverse<Empty, MemberOnlyInversePoint>);
static_assert(!CanGetCoefficientInverse<MemberOnlyInverseFacet, MemberOnlyInversePoint&>);
static_assert(!CanGetCoefficientInverse<MemberOnlyInverseFacet, Empty>);

GENDIL_HOST_DEVICE
bool CheckCoefficientStorage(Real seed)
{
   const MappedPointWithObsoleteMember mapped{{17.}, seed};
   const QuadraturePointContext<Integer, std::array<Real, 1>, Real>
      cell{0, {.5}, 2., seed, 2., 1.};
   const TwoSidedFacetQuadraturePointContext<Integer, std::array<Real, 1>,
      Real, Real, std::array<Real, 1>> global{0, {.5}, seed, 3., 1., 1., 2., 3., {1.}};
   const auto minus = global.MinusSide();
   const auto plus = global.PlusSide();
   const auto facet = MakeGeometryFacet<1>(IdentityOrientation<1>{});
   const auto reflected = MakeGeometryFacet<-1>(Permutation<1>{{-1}});
   const MemberOnlyInverseFacet obsolete{23.};
   BoundaryFacetQuadraturePointContext<Integer, std::array<Real, 1>, Real,
      std::array<Real, 1>> boundary{};
   boundary.inv_J_mesh = seed;
   auto mutable_cell = cell;
   static_assert(!CanGetCoefficientInverse<decltype(facet), MemberOnlyInversePoint>);
   static_assert(!CanGetCoefficientInverse<decltype(reflected), MemberOnlyInversePoint&>);
   static_assert(std::is_same_v<decltype(CoefficientInverse(Empty{}, cell)), const Real&>);
   static_assert(std::is_same_v<decltype(CoefficientInverse(Empty{}, minus)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(Empty{}, mutable_cell)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(obsolete, mapped)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(facet, mapped)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(reflected, mapped)), Real>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(
      obsolete, MappedPointWithObsoleteMember{})), Real>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(Empty{}, std::move(cell))), Real>);
   const auto owned_cell = GetCoefficientJacobianInverse(Empty{}, std::move(cell));
   const auto owned_mapped = GetCoefficientJacobianInverse(
      obsolete, MappedPointWithObsoleteMember{{17.}, seed});
   // Obsolete member hooks are ignored when ordinary raw storage is present.
   return &CoefficientInverse(Empty{}, mapped) == &mapped.inv_J_mesh &&
      &CoefficientInverse(Empty{}, cell) == &cell.inv_J_mesh &&
      &CoefficientInverse(Empty{}, minus) == &minus.inv_J_mesh &&
      &CoefficientInverse(Empty{}, plus) == &plus.inv_J_mesh &&
      &CoefficientInverse(facet, mapped) == &mapped.inv_J_mesh &&
      CoefficientInverse(reflected, mapped) == -seed &&
      &GetCoefficientJacobianInverse(Empty{}, mapped) == &mapped.inv_J_mesh &&
      &GetCoefficientJacobianInverse(obsolete, mapped) == &mapped.inv_J_mesh &&
      &CoefficientInverse(obsolete, mapped) == &mapped.inv_J_mesh &&
      &GetCoefficientJacobianInverse(facet, mapped) == &mapped.inv_J_mesh &&
      GetCoefficientJacobianInverse(reflected, mapped) == -seed &&
      &GetCoefficientJacobianInverse(Empty{}, mutable_cell) == &mutable_cell.inv_J_mesh &&
      &GetCoefficientJacobianInverse(Empty{}, minus) == &minus.inv_J_mesh &&
      &GetCoefficientJacobianInverse(Empty{}, plus) == &plus.inv_J_mesh &&
      &GetCoefficientJacobianInverse(Empty{}, boundary) == &boundary.inv_J_mesh &&
      &GetCoefficientJacobianInverse(reflected, boundary) == &boundary.inv_J_mesh &&
      owned_cell == seed && owned_mapped == seed;
}

template<int Sign>
GENDIL_HOST_DEVICE
bool CheckSelectedContexts(Real seed, int direction)
{
   using Context = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Matrix2, Matrix2, std::array<Real, 2>>;
   const Matrix2 current{{{seed, 2.}, {3., 5.}}};
   const Matrix2 neighbor{{{7., 11.}, {13., 17.}}};
   Context raw{3, {.25, .75}, current, neighbor, .125, 2.5, 4., 9., {Sign*.6, Sign*.8}};
   const Permutation<2> orientation{{-2*direction, direction}};
   const auto facet = MakeGeometryFacet<Sign>(orientation);
   const DerivedLocalFacet<GeometryFace<Sign, Permutation<2>>> derived{
      static_cast<const GeometryFace<Sign, Permutation<2>>&>(facet)};
   static_assert(sizeof(facet) == sizeof(GeometryFace<Sign, Permutation<2>>));
   static_assert(alignof(decltype(facet)) == alignof(GeometryFace<Sign, Permutation<2>>));
   const auto minus = GetMinusSide(facet, raw);
   const auto plus = GetPlusSide(facet, raw);
   const auto normal = GetFacetNormal(facet, raw);
   const auto repeated_normal = GetFacetNormal(facet, raw);
   const auto minus_normal = GetFacetNormal(facet, minus);
   const auto plus_normal = GetFacetNormal(facet, plus);
   const auto derived_normal = GetFacetNormal(derived, raw);
   const auto derived_minus_normal = GetFacetNormal(derived, minus);
   const auto independent_normal = GetFacetNormal(Empty{}, minus);
   const auto opposite_normal = GetFacetNormal(MakeGeometryFacet<-Sign>(orientation), raw);
   const auto derived_minus = GetMinusSide(derived, raw);
   const auto derived_plus = GetPlusSide(derived, raw);
   static_assert(!std::is_reference_v<decltype(GetFacetNormal(facet, raw))>);
   const auto minus_coefficient = CoefficientInverse(facet, minus);
   const auto plus_coefficient = CoefficientInverse(facet, plus);
   const auto direct_minus = GetCoefficientJacobianInverse(facet, minus);
   const auto direct_plus = GetCoefficientJacobianInverse(derived, plus);
   bool ok = EqualMatrix(raw.CurrentSide().inv_J_mesh, current) &&
      EqualMatrix(raw.NeighborSide().inv_J_mesh, neighbor) &&
      raw.CurrentSide().inverse_normal_cell_size == 4. &&
      raw.NeighborSide().inverse_normal_cell_size == 9. &&
      EqualMatrix(raw.inv_J_mesh_current, current) &&
      EqualMatrix(raw.inv_J_mesh_neighbor, neighbor) &&
      raw.physical_normal[0] == Sign*.6 && raw.physical_normal[1] == Sign*.8 &&
      minus.physical_normal[0] == Sign*.6 && minus.physical_normal[1] == Sign*.8 &&
      plus.physical_normal[0] == Sign*.6 && plus.physical_normal[1] == Sign*.8 &&
      normal[0] == Sign*.6 && normal[1] == Sign*.8 &&
      repeated_normal[0] == normal[0] && repeated_normal[1] == normal[1] &&
      minus_normal[0] == normal[0] && minus_normal[1] == normal[1] &&
      plus_normal[0] == normal[0] && plus_normal[1] == normal[1] &&
      EqualMatrix(direct_minus, minus_coefficient) && EqualMatrix(direct_plus, plus_coefficient) &&
      derived_normal[0] == normal[0] && derived_normal[1] == normal[1] &&
      derived_minus_normal[0] == normal[0] && derived_minus_normal[1] == normal[1] &&
      independent_normal[0] == normal[0] && independent_normal[1] == normal[1] &&
      opposite_normal[0] == normal[0] && opposite_normal[1] == normal[1] &&
      derived_minus.inverse_normal_cell_size == minus.inverse_normal_cell_size &&
      derived_plus.inverse_normal_cell_size == plus.inverse_normal_cell_size &&
      minus.quad_index == 3 && plus.quad_index == 3 &&
      minus.X[1] == .75 && plus.X[1] == .75 &&
      minus.weight == .125 && plus.weight == .125 &&
      minus.det_J_facet == 2.5 && plus.det_J_facet == 2.5;
   if constexpr (Sign == 1)
   {
      static_assert(std::is_same_v<decltype(CoefficientInverse(facet, minus)), const Matrix2&>);
      static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(facet, minus)), const Matrix2&>);
      static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(derived, plus)), const Matrix2&>);
      return ok && EqualMatrix(minus.inv_J_mesh, current) && EqualMatrix(plus.inv_J_mesh, neighbor) &&
         EqualMatrix(minus_coefficient, current) && EqualMatrix(plus_coefficient, neighbor) &&
         &GetCoefficientJacobianInverse(MakeGeometryFacet<1>(orientation), minus) == &minus.inv_J_mesh &&
         minus.inverse_normal_cell_size == 4. && plus.inverse_normal_cell_size == 9.;
   }
   else
   {
      static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(facet, minus)), Matrix2>);
      // Raw maps still match each side's interpolated reference derivatives.
      // Callback maps express their reference columns in the fixed minus-trace axes.
      const Matrix2 expected_minus{{{-11.*direction, 7.*direction}, {-17.*direction, 13.*direction}}};
      const Matrix2 expected_plus{{{-2.*direction, seed*direction}, {-5.*direction, 3.*direction}}};
      return ok && EqualMatrix(minus.inv_J_mesh, neighbor) && EqualMatrix(plus.inv_J_mesh, current) &&
         EqualMatrix(minus_coefficient, expected_minus) && EqualMatrix(plus_coefficient, expected_plus) &&
         minus.inverse_normal_cell_size == 9. && plus.inverse_normal_cell_size == 4.;
   }
}

GENDIL_HOST_DEVICE
bool CheckJacobianOwnership(Real seed)
{
   using Point = OneSidedFacetQuadraturePointContext<Integer, std::array<Real, 2>,
      Matrix2, std::array<Real, 2>>;
   Point point{0, {.25, .75}, {{{seed, 2.}, {3., 5.}}}, 1., 1., 1., {.6, .8}};
   const auto facet = MakeGeometryFacet<1>(IdentityOrientation<2>{});
   const auto identity = MakeGeometryFacet<-1>(IdentityOrientation<2>{});
   const auto rotated = MakeGeometryFacet<-1>(Permutation<2>{{-2, 1}});
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(facet, point)), const Matrix2&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(facet, Point(point))), Matrix2>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(identity, Point(point))), Matrix2>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(rotated, Point(point))), Matrix2>);
   const auto copied = GetCoefficientJacobianInverse(facet, Point(point));
   const auto copied_identity = GetCoefficientJacobianInverse(identity, Point(point));
   const auto transformed = GetCoefficientJacobianInverse(rotated, Point(point));
   const auto detached = [&]()
   {
      auto source = point;
      return GetCoefficientJacobianInverse(facet, std::move(source));
   }();
   const Matrix2 expected{{{seed, 2.}, {3., 5.}}};
   const Matrix2 expected_rotated{{{-2., seed}, {-5., 3.}}};
   point.inv_J_mesh[0][0] = seed + 17.;
   return EqualMatrix(copied, expected) && EqualMatrix(copied_identity, expected) &&
      EqualMatrix(detached, expected) && EqualMatrix(transformed, expected_rotated) &&
      &GetCoefficientJacobianInverse(facet, point) == &point.inv_J_mesh &&
      GetCoefficientJacobianInverse(facet, point)[0][0] == seed + 17.;
}

template<int Sign>
GENDIL_HOST_DEVICE
bool CheckPointOwnership(Real seed)
{
   using Context = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Matrix2, Empty, std::array<Real, 2>>;
   const auto facet = MakeGeometryFacet<Sign>(Permutation<2>{{-2, 1}});
   Context first{2, {.25, .75}, {{{seed, 2.}, {3., 5.}}}, {},
      .125, 2.5, 4., 0., {Sign*.6, Sign*.8}};
   const auto first_minus = GetMinusSide(facet, first);
   const auto first_plus = GetPlusSide(facet, first);
   auto second = first;
   second.quad_index = 3;
   second.inv_J_mesh_current[0][0] = seed + 1.;
   second.physical_normal = {-Sign*.8, Sign*.6};
   const auto second_minus = GetMinusSide(facet, second);
   const auto second_plus = GetPlusSide(facet, second);
   const auto copied = first_minus;
   const auto detached = [&]()
   {
      const auto owner = MakeGeometryFacet<Sign>(Permutation<2>{{-2, 1}});
      const auto source = first;
      return GetMinusSide(owner, source);
   }();
   const auto temporary_plus = GetPlusSide(MakeGeometryFacet<Sign>(Permutation<2>{{-2, 1}}), second);
   const auto normal = GetFacetNormal(facet, first);
   first.physical_normal = {};
   // Selected sides remain valid after both their source point and facet die.
   // Each curved point owns its canonical normal. Reading it produces an
   // independent value, including without the originating facet context.
   return first_minus.quad_index == 2 && second_minus.quad_index == 3 &&
      first.inv_J_mesh_current[0][0] == seed &&
      second.inv_J_mesh_current[0][0] == seed + 1. &&
      copied.physical_normal[0] == Sign*.6 &&
      first_minus.physical_normal[0] == Sign*.6 &&
      first_plus.physical_normal[0] == Sign*.6 &&
      second_minus.physical_normal[0] == -Sign*.8 &&
      second_plus.physical_normal[1] == Sign*.6 &&
      detached.physical_normal[0] == Sign*.6 &&
      temporary_plus.physical_normal[1] == Sign*.6 &&
      normal[0] == Sign*.6 && normal[1] == Sign*.8 &&
      GetFacetNormal(facet, copied)[0] == Sign*.6 &&
      GetFacetNormal(facet, first_minus)[0] == Sign*.6 &&
      GetFacetNormal(facet, first_plus)[0] == Sign*.6 &&
      GetFacetNormal(facet, second_minus)[0] == -Sign*.8 &&
      GetFacetNormal(facet, second_plus)[1] == Sign*.6 &&
      GetFacetNormal(Empty{}, detached)[0] == Sign*.6 &&
      GetFacetNormal(facet, temporary_plus)[1] == Sign*.6;
}

template<bool OrientationByValue>
struct CustomNeighborFace
{
   Permutation<2> orientation;
   int* orientation_reads;

   GENDIL_HOST_DEVICE
   decltype(auto) GetOrientation() const
   {
      ++*orientation_reads;
      if constexpr (OrientationByValue) return Permutation<2>(orientation);
      else return (orientation);
   }
};

template<int Sign, bool EndpointByValue, bool OrientationByValue>
struct CustomFace
{
   using current_side_type = Empty;
   using neighbor_side_type = CustomNeighborFace<OrientationByValue>;
   Empty current;
   neighbor_side_type neighbor;
   int* endpoint_reads;

   GENDIL_HOST_DEVICE
   const auto& CurrentSide() const { return current; }
   GENDIL_HOST_DEVICE
   decltype(auto) NeighborSide() const
   {
      ++*endpoint_reads;
      if constexpr (EndpointByValue) return neighbor_side_type(neighbor);
      else return (neighbor);
   }
   GENDIL_HOST_DEVICE
   auto GetNormalOrientation() const { return std::integral_constant<int, Sign>{}; }
};

template<int Sign, bool EndpointByValue, bool OrientationByValue>
GENDIL_HOST_DEVICE
bool CheckOrientationLookup(Real seed)
{
   using Face = CustomFace<Sign, EndpointByValue, OrientationByValue>;
   using Context = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Matrix2, Matrix2, std::array<Real, 2>>;
   int orientation_reads = 0;
   int endpoint_reads = 0;
   const auto facet = [&]()
   {
      const Face source{{}, {{{-2, 1}}, &orientation_reads}, &endpoint_reads};
      return LocalFacetContext{source};
   }();
   static_assert(sizeof(facet) == sizeof(Face));
   static_assert(alignof(decltype(facet)) == alignof(Face));
   auto copied = facet;
   const auto moved = [&]()
   {
      auto source = facet;
      return LocalFacetContext<Face>(std::move(source));
   }();
   Context raw{3, {.25, .75}, {{{seed, 2.}, {3., 5.}}},
      {{{7., 11.}, {13., 17.}}}, .125, 2.5, 4., 9., {Sign*.6, Sign*.8}};
   const auto first = GetMinusSide(facet, raw);
   copied.neighbor.orientation = {{1, 2}};
   const auto second = GetMinusSide(copied, raw);
   const auto third = GetMinusSide(moved, raw);
   const auto plus = GetPlusSide(facet, raw);
   const auto normal = GetFacetNormal(facet, raw);
   const auto minus_normal = GetFacetNormal(facet, first);
   const auto plus_normal = GetFacetNormal(facet, plus);
   if (normal[0] != Sign*.6 || minus_normal[0] != normal[0] ||
       plus_normal[0] != normal[0] || raw.physical_normal[0] != Sign*.6) return false;
   // Construction, copy/move, trace selection, and normal reads do not fetch
   // runtime endpoint/reference-coordinate orientation metadata.
   if (orientation_reads != 0 || endpoint_reads != 0) return false;
   const auto first_inverse = CoefficientInverse(facet, first);
   const auto second_inverse = CoefficientInverse(copied, second);
   const auto third_inverse = CoefficientInverse(moved, third);
   const auto plus_inverse = CoefficientInverse(facet, plus);
   if constexpr (Sign == 1)
   {
      // The global reference frame already agrees: even callbacks need no lookup.
      return orientation_reads == 0 && endpoint_reads == 0 &&
         EqualMatrix(first_inverse, raw.inv_J_mesh_current) &&
         EqualMatrix(second_inverse, raw.inv_J_mesh_current) &&
         EqualMatrix(third_inverse, raw.inv_J_mesh_current) &&
         EqualMatrix(plus_inverse, raw.inv_J_mesh_neighbor);
   }
   else
   {
      const Matrix2 expected_minus{{{-11., 7.}, {-17., 13.}}};
      const Matrix2 expected_plus{{{-2., seed}, {-5., 3.}}};
      // By-value endpoint/orientation temporaries must survive each callback's
      // normalization. Each facet uses its own current metadata, without a cache.
      return orientation_reads == 4 && endpoint_reads == 4 &&
         EqualMatrix(first_inverse, expected_minus) &&
         EqualMatrix(second_inverse, raw.inv_J_mesh_neighbor) &&
         EqualMatrix(third_inverse, expected_minus) &&
         EqualMatrix(plus_inverse, expected_plus);
   }
}

struct CountingCell
{
   int* orientation_calls;
};

GENDIL_HOST_DEVICE
void ApplyOrientationToCell(const IdentityOrientation<2>&, CountingCell& cell)
{
   ++*cell.orientation_calls;
}

struct CountingWeakFormContext
{
   int* cell_acquisitions;
   int* orientation_calls;

   GENDIL_HOST_DEVICE
   CountingCell GetCell(GlobalIndex) const
   {
      ++*cell_acquisitions;
      return {orientation_calls};
   }
};

template<class Integrand>
GENDIL_HOST_DEVICE
const auto& GetCellIntegrationDomainMesh(const Integrand&, const CountingWeakFormContext& context)
{
   return context;
}

struct IdentityCoefficient
{
   GENDIL_HOST_DEVICE
   Real operator()(Real value) const { return value; }
};

struct IndexCoefficient
{
   GENDIL_HOST_DEVICE
   Real operator()(GlobalIndex index) const { return Real(index); }
};

struct VelocityCoefficient
{
   GENDIL_HOST_DEVICE
   std::array<Real, 2> operator()(const std::array<Real, 2>&) const { return {1., 0.}; }
};

// Exercise the production point factory with device-local geometry and qdata.
// Neither the face nor the qdata exposes a usable neighbor, so a normal-only
// request cannot accidentally acquire or evaluate a second cell.
struct NormalFactoryCell
{
   static constexpr Integer Dim = 2;
   using physical_coordinates = std::array<Real, Dim>;
   using jacobian = std::array<Real, Dim>;
   int* evaluations;

   template<class QuadData>
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian(const TensorIndex<Dim>&, const QuadData&,
      physical_coordinates& coordinates, jacobian& matrix) const
   {
      ++*evaluations;
      coordinates = {.25, .75};
      matrix = {2., 4.};
   }
};

struct NormalFactoryMesh
{
   using cell_type = NormalFactoryCell;
};

template<class Integrand>
GENDIL_HOST_DEVICE
const auto& GetCellIntegrationDomainMesh(const Integrand&, const NormalFactoryMesh& mesh)
{
   return mesh;
}

struct NormalFactoryWeights
{
   Real weight;
   GENDIL_HOST_DEVICE
   Real GetWeight(Integer) const { return weight; }
};

struct NormalFactoryQuadData
{
   using PointData = TensorProductData<NormalFactoryWeights, NormalFactoryWeights>;
   std::tuple<PointData> current;
   GENDIL_HOST_DEVICE
   const auto& MinusSide() const { return current; }
};

struct NormalFactoryOperatorContext
{
   NormalFactoryQuadData data;
   template<StaticString Name>
   GENDIL_HOST_DEVICE
   const auto& GetMeshFacetQuadData() const { return data; }
};

template<class Integrand>
GENDIL_HOST_DEVICE
bool CheckNormalConstruction(int direction, const Integrand& integrand)
{
   using Side = FaceView<std::integral_constant<Integer, 0>, HyperCube<2>,
      IdentityOrientation<2>, CanonicalVector<2, 0, 1>, ConformingFaceMap<2>>;
   using RuntimeFace = LocalFaceInfo<Side, Empty, int>;
   const RuntimeFace runtime{{}, {}, direction};
   const NormalFactoryMesh mesh{};
   const NormalFactoryOperatorContext op{{std::make_tuple(
      MakeTensorProductData(NormalFactoryWeights{1.}, NormalFactoryWeights{.25}))}};
   int evaluations = 0;
   const ElementContext<NormalFactoryCell> element{0, {&evaluations}};
   bool ok = true;
   WithLocalFacetNormalOrientation(runtime, [&](const auto& face)
   {
      const LocalFacetContext facet{face};
      constexpr int Sign = static_face_normal_orientation_v<decltype(face)>;
      static_assert(!local_interior_context_requires_neighbor_jacobian_v<Integrand, Sign>);
      const auto point = MakeLocalInteriorFacetQuadraturePointContext(
         Empty{}, mesh, op, element, facet, integrand, TensorIndex<2>{});
      static_assert(std::is_same_v<std::remove_cvref_t<decltype(point.inv_J_mesh_neighbor)>, Empty>);
      const auto minus = GetMinusSide(facet, point);
      const auto plus = GetPlusSide(facet, point);
      const auto normal = GetFacetNormal(Empty{}, point);
      const auto minus_normal = GetFacetNormal(Empty{}, minus);
      const auto plus_normal = GetFacetNormal(Empty{}, plus);
      // The diagonal map has outward normal (1, 0), surface density 4,
      // and inverse normal cell size 1/2, independently of the trace sign.
      ok = point.physical_normal[0] == direction && point.physical_normal[1] == 0. &&
         normal[0] == direction && normal[1] == 0. &&
         minus_normal[0] == direction && minus_normal[1] == 0. &&
         plus_normal[0] == direction && plus_normal[1] == 0. &&
         point.inv_J_mesh_current[0] == .5 && point.inv_J_mesh_current[1] == .25 &&
         point.inverse_normal_cell_size_current == .5 &&
         point.inverse_normal_cell_size_neighbor == 0. &&
         point.det_J_facet == 4. && point.weight == .25 &&
         point.X[0] == .25 && point.X[1] == .75;
   });
   return ok && evaluations == 1;
}

auto MakeTraceExpressions()
{
   // Build the DSL on the host; device kernels evaluate the captured expressions.
   const auto rho = MakeCoefficient<"trace_rho", InverseNormalCellSize>(IdentityCoefficient{});
   const auto index = MakeCoefficient<"trace_index", ElementIndex>(IndexCoefficient{});
   const auto beta = MakeVectorCoefficient<"trace_velocity", PhysicalCoordinates>(VelocityCoefficient{});
   const auto inverse = MakeCoefficient<"callback_inverse", JacobianInverse>(IdentityCoefficient{});
   return std::tuple{minus(rho), plus(rho), average(rho), jump(rho),
      index, minus(index), plus(index), upwind(beta, rho),
      Normal{}, minus(Normal{}), plus(Normal{}), upwind(beta, ScaleExpr{3.}), inverse,
      integrate(InteriorFacets<"normal_factory">{}, Normal{})};
}

template<class Expressions>
GENDIL_HOST_DEVICE
bool CheckExpressionEvaluation(int direction, const Expressions& expressions)
{
   using Face = LocalFaceInfo<CurrentFace, NeighborFace<IdentityOrientation<2>>, int>;
   using Point = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Matrix2, Matrix2, std::array<Real, 2>>;
   bool ok = true;
   WithLocalFacetNormalOrientation(Face{{}, {}, direction}, [&](const auto& face)
   {
      const LocalFacetContext facet{face};
      constexpr int Sign = static_face_normal_orientation_v<decltype(face)>;
      const FacetTraceFields<Empty, Empty> fields{};
      // Both velocity signs exercise upwind's trace selection. All three normal
      // expressions must agree, including the untraced top-level normal.
      for (int normal_index = 0; normal_index < 2; ++normal_index)
      {
         const int normal_direction = normal_index == 0 ? -1 : 1;
         Point point{3, {.25, .75}, {{{1., 0.}, {0., 1.}}},
            {{{2., 0.}, {0., 3.}}}, .125, 2.5, 4., 9.,
            {Sign*.6*normal_direction, Sign*.8*normal_direction}};
         const auto evaluate = [&](const auto& expression)
         { return expression(Empty{}, Empty{}, Empty{}, facet, point, fields); };
         const Real minus_size = Sign == 1 ? 4. : 9.;
         const Real plus_size = Sign == 1 ? 9. : 4.;
         const Real speed = Sign*.6*normal_direction;
         const Real expected_flux = speed*(speed >= 0 ? minus_size : plus_size);
         const auto normal = evaluate(std::get<8>(expressions));
         const auto minus_normal = evaluate(std::get<9>(expressions));
         const auto plus_normal = evaluate(std::get<10>(expressions));
         const auto repeated_normal = evaluate(std::get<8>(expressions));
         ok = ok && evaluate(std::get<0>(expressions)) == minus_size &&
            evaluate(std::get<1>(expressions)) == plus_size &&
            evaluate(std::get<2>(expressions)) == 6.5 &&
            evaluate(std::get<3>(expressions)) == minus_size - plus_size &&
            evaluate(std::get<4>(expressions)) == 13. &&
            evaluate(std::get<5>(expressions)) == 13. &&
            evaluate(std::get<6>(expressions)) == 13. &&
            evaluate(std::get<7>(expressions)) == expected_flux &&
            normal[0] == speed && minus_normal[0] == speed && plus_normal[0] == speed &&
            normal[1] == Sign*.8*normal_direction &&
            minus_normal[1] == normal[1] && plus_normal[1] == normal[1] &&
            repeated_normal[0] == normal[0] && repeated_normal[1] == normal[1] &&
            point.physical_normal[0] == Sign*.6*normal_direction &&
            point.physical_normal[1] == Sign*.8*normal_direction;
      }
      // Existing element contexts retain their reference-returning input API.
      const ElementContext<Empty> element{29, {}};
      const Point point{};
      const auto& index = CoefficientInputGetter<ElementIndex>::Get(
         Empty{}, Empty{}, Empty{}, element, point, fields);
      static_assert(std::is_same_v<decltype(index), const GlobalIndex&>);
      ok = ok && &index == &element.element_index;
   });
   return ok;
}

template<class Expressions>
GENDIL_HOST_DEVICE
bool CheckGlobalAndBoundaryNormals(int direction, const Expressions& expressions)
{
   using Coordinates = std::array<Real, 2>;
   const Coordinates expected{.6*direction, .8*direction};
   const ElementContext<Empty> element{29, {}};
   const FacetTraceFields<Empty, Empty> fields{};
   const TwoSidedFacetQuadraturePointContext<Integer, Coordinates,
      Matrix2, Matrix2, Coordinates> global{0, {.25, .75},
         {{{2., 0.}, {0., 3.}}}, {{{5., 0.}, {0., 7.}}},
         .125, 2.5, 4., 9., expected};
   const auto evaluate = [&](const auto& expression)
   { return expression(Empty{}, Empty{}, Empty{}, element, global, fields); };
   const auto normal = evaluate(std::get<8>(expressions));
   const auto minus_normal = evaluate(std::get<9>(expressions));
   const auto plus_normal = evaluate(std::get<10>(expressions));
   const auto direct_normal = GetFacetNormal(element, global);
   const auto direct_minus = GetMinusSide(element, global);
   const auto direct_plus = GetPlusSide(element, global);
   const auto direct_minus_normal = GetFacetNormal(element, direct_minus);
   const auto direct_plus_normal = GetFacetNormal(element, direct_plus);

   BoundaryFacetQuadraturePointContext<Integer, Coordinates, Matrix2, Coordinates> boundary{};
   boundary.physical_normal = expected;
   const auto boundary_normal = Normal{}(
      Empty{}, Empty{}, Empty{}, element, boundary, Empty{});
   const auto repeated_boundary_normal = Normal{}(
      Empty{}, Empty{}, Empty{}, element, boundary, Empty{});
   const auto direct_boundary_normal = GetFacetNormal(element, boundary);
   const auto opposite_facet = MakeGeometryFacet<-1>(IdentityOrientation<2>{});
   const auto boundary_with_opposite_facet = GetFacetNormal(opposite_facet, boundary);

   // An inverse Jacobian and reference normal do not supply the unit-normal
   // accessor contract. Stored normals remain authoritative when both exist.
   using Face = FaceView<Integer, HyperCube<2>, IdentityOrientation<2>,
      CanonicalVector<2, 0, 1>, ConformingFaceMap<2>>;
   using CellPoint = QuadraturePointContext<Integer, Coordinates, Matrix2>;
   using ReferenceFacet = facet_normal_custom_test::ReferenceFacet;
   static_assert(!CanGetFacetNormal<Face, CellPoint>);
   static_assert(!CanGetFacetNormal<ReferenceFacet, CellPoint>);
   static_assert(!CanGetFacetNormal<Empty, CellPoint>);
   const auto stored_minus_normal = GetFacetNormal(Face{}, direct_minus);
   const auto stored_plus_normal = GetFacetNormal(ReferenceFacet{}, direct_plus);

   bool ok = direct_minus.inverse_normal_cell_size == 4. &&
      direct_plus.inverse_normal_cell_size == 9. &&
      direct_minus.inv_J_mesh[0][0] == 2. && direct_plus.inv_J_mesh[0][0] == 5.;
   ConstexprLoop<2>([&](auto component)
   {
      ok = ok && normal[component] == expected[component] &&
         minus_normal[component] == expected[component] &&
         plus_normal[component] == expected[component] &&
         direct_normal[component] == expected[component] &&
         direct_minus_normal[component] == expected[component] &&
         direct_plus_normal[component] == expected[component] &&
         stored_minus_normal[component] == expected[component] &&
         stored_plus_normal[component] == expected[component] &&
         global.physical_normal[component] == expected[component] &&
         boundary_normal[component] == expected[component] &&
         repeated_boundary_normal[component] == expected[component] &&
         direct_boundary_normal[component] == expected[component] &&
         boundary_with_opposite_facet[component] == expected[component] &&
         boundary.physical_normal[component] == expected[component];
   });
   return ok;
}

template<class Expressions>
GENDIL_HOST_DEVICE
bool CheckCustomJacobianAccessor(Real seed, const Expressions& expressions)
{
   const coefficient_jacobian_custom_test::Context context{};
   using CustomPoint = coefficient_jacobian_custom_test::Point;
   CustomPoint point{-7., seed};
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(context, point)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(context, CustomPoint{})), Real>);
   static_assert(std::is_same_v<decltype(CoefficientInverse(context, point)), const Real&>);
   const auto owned = GetCoefficientJacobianInverse(context, CustomPoint{-7., seed});
   const auto& direct = GetCoefficientJacobianInverse(context, point);
   const auto& input = CoefficientInverse(context, point);
   using PointWithoutRaw = coefficient_jacobian_custom_test::PointWithoutRaw;
   const PointWithoutRaw custom_point{seed + 2.};
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(context, custom_point)), const Real&>);
   static_assert(std::is_same_v<decltype(GetCoefficientJacobianInverse(context, PointWithoutRaw{})), Real>);
   const auto owned_without_raw = GetCoefficientJacobianInverse(context, PointWithoutRaw{seed + 2.});
   const Real evaluated = std::get<12>(expressions)(
      Empty{}, Empty{}, Empty{}, context, point, Empty{});
   const Real evaluated_without_raw = std::get<12>(expressions)(
      Empty{}, Empty{}, Empty{}, context, custom_point, Empty{});
   point.callback_inverse = seed + 4.;
   return &direct == &point.callback_inverse && &input == &point.callback_inverse &&
      direct == seed + 4. && input == seed + 4. && owned == seed && evaluated == seed &&
      point.inv_J_mesh == -7. &&
      &GetCoefficientJacobianInverse(context, custom_point) == &custom_point.callback_inverse &&
      &CoefficientInverse(context, custom_point) == &custom_point.callback_inverse &&
      owned_without_raw == seed + 2. && evaluated_without_raw == seed + 2.;
}

template<class Expressions>
GENDIL_HOST_DEVICE
bool CheckCustomNormalAccessor(int direction, const Expressions& expressions)
{
   const facet_normal_custom_test::Facet facet{};
   const facet_normal_custom_test::Point point{{.25, .75}, {.6*direction, .8*direction}};
   const FacetTraceFields<Empty, Empty> fields{};
   const auto evaluate = [&](const auto& expression)
   { return expression(Empty{}, Empty{}, Empty{}, facet, point, fields); };
   const auto direct = GetFacetNormal(facet, point);
   const auto direct_minus = GetMinusSide(facet, point);
   const auto direct_plus = GetPlusSide(facet, point);
   const auto normal = evaluate(std::get<8>(expressions));
   const auto minus_normal = evaluate(std::get<9>(expressions));
   const auto plus_normal = evaluate(std::get<10>(expressions));
   const Real speed = -.6*direction;
   const Real selected_value = speed >= 0 ? 4. : 9.;
   return direct[0] == speed && direct[1] == -.8*direction &&
      normal[0] == direct[0] && normal[1] == direct[1] &&
      minus_normal[0] == direct[0] && minus_normal[1] == direct[1] &&
      plus_normal[0] == direct[0] && plus_normal[1] == direct[1] &&
      direct_minus.inverse_normal_cell_size == 4. &&
      direct_plus.inverse_normal_cell_size == 9. &&
      evaluate(std::get<0>(expressions)) == 4. &&
      evaluate(std::get<1>(expressions)) == 9. &&
      evaluate(std::get<2>(expressions)) == 6.5 &&
      evaluate(std::get<3>(expressions)) == -5. &&
      evaluate(std::get<7>(expressions)) == speed*selected_value &&
      point.normal_value[0] == .6*direction && point.normal_value[1] == .8*direction;
}

template<int Sign>
GENDIL_HOST_DEVICE
bool CheckGeometryAcquisition()
{
   int acquisitions = 0;
   int orientations = 0;
   const CountingWeakFormContext context{&acquisitions, &orientations};
   const GeometryFace<Sign, IdentityOrientation<2>> face{};
   const auto rho = MakeCoefficient<"rho", InverseNormalCellSize>(IdentityCoefficient{});
   const auto current_rho = [&]()
   {
      if constexpr (Sign == 1) return minus(rho);
      else return plus(rho);
   }();
   const auto neighbor_rho = [&]()
   {
      if constexpr (Sign == 1) return plus(rho);
      else return minus(rho);
   }();
   const auto current = MakeLocalInteriorFacetContext(context, current_rho, face);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(current)>,
      LocalFacetContext<std::remove_cvref_t<decltype(face)>>>);
   using Point = LocalFacetQuadraturePointData<Integer, std::array<Real, 2>,
      Matrix2, Empty, std::array<Real, 2>>;
   const auto point = GetMinusSide(current, Point{});
   static_assert(!requires { point.neighbor_orientation; });
   const bool no_neighbor = acquisitions == 0 && orientations == 0;
   const auto neighbor = MakeLocalInteriorFacetContext(context, neighbor_rho, face);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(neighbor)>,
      LocalFacetContext<std::remove_cvref_t<decltype(face)>, CountingCell>>);
   return no_neighbor && acquisitions == 1 && orientations == 1 &&
      neighbor.second_cell.orientation_calls == &orientations;
}

GENDIL_HOST_DEVICE
bool CheckFacetOwnership(Real seed)
{
   return CheckPointOwnership<1>(seed) && CheckPointOwnership<-1>(seed) &&
      CheckOrientationLookup<1, false, false>(seed) &&
      CheckOrientationLookup<1, true, false>(seed) &&
      CheckOrientationLookup<1, false, true>(seed) &&
      CheckOrientationLookup<1, true, true>(seed) &&
      CheckOrientationLookup<-1, false, false>(seed) &&
      CheckOrientationLookup<-1, true, false>(seed) &&
      CheckOrientationLookup<-1, false, true>(seed) &&
      CheckOrientationLookup<-1, true, true>(seed) &&
      CheckCoefficientStorage(seed) &&
      CheckJacobianOwnership(seed) &&
      CheckGeometryAcquisition<1>() && CheckGeometryAcquisition<-1>();
}

using RuntimeFace = LocalFaceInfo<Empty, Empty, int>;
template<int Sign>
using StaticFace = LocalFaceInfo<Empty, Empty, std::integral_constant<int, Sign>>;

GENDIL_HOST_DEVICE
bool CheckDispatch(int direction)
{
   int selected = 0;
   int calls = 0;
   const RuntimeFace runtime{{}, {}, direction};
   WithLocalFacetNormalOrientation(runtime, [&](const auto& face)
   {
      selected = decltype(face.GetNormalOrientation())::value;
      static_assert(static_face_normal_orientation_v<decltype(face)> ==
         decltype(face.GetNormalOrientation())::value);
      ++calls;
   });
   bool ok = selected == direction && calls == 1 &&
      runtime.GetNormalOrientation() == direction;
   const StaticFace<-1> fixed{};
   WithLocalFacetNormalOrientation(fixed, [&](const auto& face)
   {
      static_assert(decltype(face.GetNormalOrientation())::value == -1);
      selected = decltype(face.GetNormalOrientation())::value;
      ok = ok && &face == &fixed;
   });
   return ok && selected == -1;
}

// A minus trace needs neighbor geometry when the shared normal points into current.
void CheckGeometryRequirements()
{
   const auto rho = MakeCoefficient<"rho", InverseNormalCellSize>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   using Rho = decltype(rho);
   using Minus = decltype(minus(rho));
   using Plus = decltype(plus(rho));
   using Average = decltype(average(rho));
   using Jump = decltype(jump(rho));
   static_assert(requires_minus_side_jacobian_v<Rho> && !requires_plus_side_jacobian_v<Rho>);
   static_assert(requires_minus_side_jacobian_v<Minus> && !requires_plus_side_jacobian_v<Minus>);
   static_assert(!requires_minus_side_jacobian_v<Plus> && requires_plus_side_jacobian_v<Plus>);
   static_assert(requires_minus_side_jacobian_v<Average> && requires_plus_side_jacobian_v<Average>);
   static_assert(requires_minus_side_jacobian_v<Jump> && requires_plus_side_jacobian_v<Jump>);
   static_assert(!local_interior_context_requires_neighbor_jacobian_v<Minus, 1>);
   static_assert(local_interior_context_requires_neighbor_jacobian_v<Minus, -1>);
   static_assert(local_interior_context_requires_neighbor_jacobian_v<Plus, 1>);
   static_assert(!local_interior_context_requires_neighbor_jacobian_v<Plus, -1>);
   static_assert(local_interior_context_requires_neighbor_jacobian_v<Average, 1>);
   static_assert(local_interior_context_requires_neighbor_jacobian_v<Average, -1>);
   using Gradient = decltype(grad(TrialSpace<"u">{}));
   using PlusGradient = decltype(plus(grad(TrialSpace<"u">{})));
   using Value = decltype(plus(TrialSpace<"u">{}));
   static_assert(requires_minus_side_jacobian_v<Gradient> && !requires_plus_side_jacobian_v<Gradient>);
   static_assert(!requires_minus_side_jacobian_v<PlusGradient> && requires_plus_side_jacobian_v<PlusGradient>);
   static_assert(!local_interior_context_requires_neighbor_jacobian_v<Value, 1>);
   static_assert(!local_interior_context_requires_neighbor_jacobian_v<Value, -1>);
}
} // namespace local_facet_geometry_test

int main()
{
   using namespace local_facet_geometry_test;
   CheckGeometryRequirements();
   const auto expressions = MakeTraceExpressions();
   bool ok = true;
   for (int direction : {-1, 1})
   {
      ok = Check2D(1.25, direction) && Check3D(-2.75, direction) &&
         CheckProduct(.375, direction) && CheckScalarContext(1.25) && CheckSelectedContexts<1>(2.25, direction) &&
         CheckSelectedContexts<-1>(-1.5, direction) && CheckCompactJacobians(1.25, direction) &&
         CheckDiagonalCallback(-1.5, direction) && CheckDispatch(direction) &&
         CheckFacetOwnership(-1.5) && CheckSideAccessors(-1.5) &&
         CheckExpressionEvaluation(direction, expressions) &&
         CheckGlobalAndBoundaryNormals(direction, expressions) &&
         CheckCustomNormalAccessor(direction, expressions) &&
         CheckCustomJacobianAccessor(-1.5, expressions) &&
         CheckNormalConstruction(direction, std::get<13>(expressions)) && ok;
   }
#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   Vector input(2), result(2);
   auto* input_host = input.WriteHostData();
   input_host[0] = -1.;
   input_host[1] = 1.;
   const auto* directions = input.ReadDeviceData();
   auto* values = result.WriteDeviceData();
   Policy::CandidateBlockLoop(2, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(2))
      {
         const int direction = static_cast<int>(directions[i]);
         const Real seed = .25 + directions[i];
         values[i] = Check2D(seed, direction) && Check3D(seed, direction) &&
            CheckProduct(seed, direction) && CheckScalarContext(seed) && CheckSelectedContexts<1>(seed, direction) &&
            CheckSelectedContexts<-1>(seed, direction) && CheckCompactJacobians(seed, direction) &&
            CheckDiagonalCallback(seed, direction) && CheckDispatch(direction) &&
            CheckFacetOwnership(seed) && CheckSideAccessors(seed) &&
            CheckExpressionEvaluation(direction, expressions) &&
            CheckGlobalAndBoundaryNormals(direction, expressions) &&
            CheckCustomNormalAccessor(direction, expressions) &&
            CheckCustomJacobianAccessor(seed, expressions) &&
            CheckNormalConstruction(direction, std::get<13>(expressions));
      }
   });
   GENDIL_DEVICE_SYNC;
   const auto* host = result.ReadHostData();
   ok = host[0] == 1. && host[1] == 1. && ok;
#endif
   std::cout << (ok ? "PASS" : "FAIL") << ": local facet geometry views\n";
   return ok ? 0 : 1;
}
