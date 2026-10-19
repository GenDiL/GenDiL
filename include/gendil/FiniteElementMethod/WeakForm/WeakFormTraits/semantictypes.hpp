// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

namespace gendil
{

enum class SemanticSide { Unqualified, Minus, Plus };
// Selected by the symbolic integration domain, never by a runtime context.
enum class SemanticEntity { Unspecified, Cell, BoundaryFacet, InteriorFacet };
enum class FormResultKind { Scalar, Residual, Invalid };
enum class SemanticShape { Scalar, Vector, Matrix, Unresolved, Unsupported };
enum class SemanticTestLinearity { TestFree, Linear, Nonlinear, Inhomogeneous, MultipleOutputs, Unsupported };
enum class ActiveAffinity { Independent, Linear, Affine, Nonlinear, Unknown };
// Reserved independently of test degree for future complex-valued forms.
// The current DSL only analyzes real algebra; this is not a conjugation rule.
enum class SemanticVariance { Real, Unsupported };

enum class GeometryRequirement : unsigned
{
   None = 0u,
   PhysicalCoordinates = 1u << 0,
   ReferenceCoordinates = 1u << 1,
   Jacobian = 1u << 2,
   JacobianInverse = 1u << 3,
   Determinant = 1u << 4,
   Normal = 1u << 5,
   InverseFacetSize = 1u << 6,
   ElementIndex = 1u << 7,
   QuadratureIndex = 1u << 8,
   IntegrationMeasure = 1u << 9
};

constexpr GeometryRequirement operator|(GeometryRequirement lhs, GeometryRequirement rhs)
{
   return GeometryRequirement(unsigned(lhs) | unsigned(rhs));
}

constexpr bool has_geometry(GeometryRequirement requirements, GeometryRequirement query)
{
   return (unsigned(requirements) & unsigned(query)) != 0u;
}

// A binding validator must discharge these obligations before execution. They
// describe symbolic operations without pretending a bound field is scalar.
enum class ShapeOperation { BoundField, Gradient, Divergence, Curl, Sum, Product, Dot, Inner, Outer, Transpose, SquareMatrix, ScalarIntegrand, UpwindSelector };
template<ShapeOperation Operation, class... Operands>
struct ShapeObligation {};

} // namespace gendil
