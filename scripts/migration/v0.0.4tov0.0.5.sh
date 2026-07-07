#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MIGRATION_NAME="GenDiL v0.0.4 to v0.0.5 migration"
MIGRATION_DESCRIPTION="Migrate mechanical source renames from GenDiL v0.0.4 to the planned v0.0.5 API."

. "${SCRIPT_DIR}/common.sh"

migration_parse_args "$@"

migration_replace_literal \
   "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp" \
   "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrix.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixAssembly/bsrassembly.hpp" \
   "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixAssembly/genericassembly.hpp" \
   "gendil/FiniteElementMethod/MatrixAssembly/Generic/genericassembly.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/facetcontext.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetcontext.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/quadraturepointcontext.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/evectorview.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/elementtensorview.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/restriction.hpp" \
   "gendil/FiniteElementMethod/Restrictions/restriction.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/doflayout.hpp" \
   "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/tensorproductdoflayout.hpp" \
   "gendil/FiniteElementMethod/Restrictions/tensorproductdoflayout.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/finiteelementorders.hpp" \
   "gendil/FiniteElementMethod/ShapeFunctions/finiteelementorders.hpp"
migration_replace_literal \
   "gendil/Interfaces/MFEM/meshconnectivity.hpp" \
   "gendil/Interfaces/MFEM/meshlocalconnectivity.hpp"
migration_replace_literal \
   "gendil/Meshes/MeshDataStructures/UnstructuredMesh/unstructuredconformingconnectivity.hpp" \
   "gendil/Meshes/MeshDataStructures/UnstructuredMesh/LocalFacetConnectivity/unstructuredconformingconnectivity.hpp"
migration_replace_literal \
   "scripts/machines/matrix/build.sh" \
   "scripts/machines/matrix/nvcc-build.sh"

migration_replace_symbol "MakeScalarEVectorView" "MakeScalarElementTensorView"
migration_replace_symbol "MakeVectorEVectorView" "MakeVectorElementTensorView"
migration_replace_symbol "MakeReadOnlyEVectorView" "MakeReadOnlyElementTensorView"
migration_replace_symbol "MakeWriteOnlyEVectorView" "MakeWriteOnlyElementTensorView"
migration_replace_symbol "MakeReadWriteEVectorView" "MakeReadWriteElementTensorView"
migration_replace_symbol "MakeEVectorView" "MakeElementTensorView"

migration_replace_symbol "MakeScalarElementVectorView" "MakeScalarElementTensorView"
migration_replace_symbol "MakeVectorElementVectorView" "MakeVectorElementTensorView"
migration_replace_symbol "MakeReadOnlyElementVectorView" "MakeReadOnlyElementTensorView"
migration_replace_symbol "MakeWriteOnlyElementVectorView" "MakeWriteOnlyElementTensorView"
migration_replace_symbol "MakeReadWriteElementVectorView" "MakeReadWriteElementTensorView"
migration_replace_symbol "MakeElementVectorView" "MakeElementTensorView"

migration_replace_symbol "make_cartesian_interior_face_connectivity" "MakeCartesianInteriorFaceConnectivity"
migration_replace_symbol "MakeMeshConnectivity" "MakeMeshLocalConnectivity"
migration_replace_symbol "MakeDomain" "MakeIntegrationDomain"
if [ "$MIGRATION_LAST_MATCHES" -gt 0 ]; then
   migration_note "Review each MakeIntegrationDomain<Name>(...) call: the argument should now be a homogeneous FiniteElementSpace or a Partition-backed MixedFiniteElementSpace rather than the mesh."
fi

migration_warn_regex \
   "Review GenericAssembly<KernelPolicy>(...) calls; the three-argument return form should become GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(...), while the four-argument in-place form remains valid" \
   'GenericAssembly[[:space:]]*<[[:space:]]*[A-Za-z_][A-Za-z0-9_:]*[[:space:]]*>[[:space:]]*\('

migration_finish
