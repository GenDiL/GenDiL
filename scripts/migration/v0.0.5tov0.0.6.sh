#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MIGRATION_NAME="GenDiL v0.0.5 to v0.0.6 migration"
MIGRATION_DESCRIPTION="Apply public symbol renames and report integration-domain calls requiring a semantic mesh/partition migration."

. "${SCRIPT_DIR}/common.sh"

migration_parse_args "$@"

migration_replace_symbol "PhysicalCoordinate" "PhysicalCoordinates"
migration_replace_symbol "ReferenceCoordinate" "ReferenceCoordinates"
migration_replace_symbol "GlobalDofIndex" "GetGlobalDofIndex"
migration_replace_symbol "ScalarLocalDofCount" "LocalDofCount"
migration_replace_symbol "DGGatherToBsr" "RestrictionGatherToBsr"
migration_replace_symbol "CGGatherToBsr" "RestrictionGatherToBsr"
migration_replace_symbol "VectorCGGatherToBsr" "RestrictionGatherToBsr"
migration_replace_symbol "DGScatterFromBsr" "RestrictionScatterFromBsr"
migration_replace_symbol "CGScatterFromBsr" "RestrictionScatterFromBsr"
migration_replace_symbol "VectorCGScatterFromBsr" "RestrictionScatterFromBsr"
migration_replace_symbol \
   "LinfProjectionElementOperator" \
   "NodalSubspaceProjectionElementOperator"
migration_replace_symbol \
   "LinfProjectionOperator" \
   "NodalSubspaceProjectionOperator"
migration_replace_symbol "LinfProjection" "NodalSubspaceProjection"
migration_replace_symbol "MakeLinfProjection" "MakeNodalSubspaceProjection"
migration_replace_symbol \
   "FaceReadDofsOrientationIsValid" \
   "IsValidOrientation"
migration_replace_symbol \
   "FaceReadDofsOrientationIsShapeCompatible" \
   "OrientedTensorDofShapeIsCompatible"
migration_replace_literal \
   "gendil/FiniteElementMethod/Restrictions/restriction.hpp" \
   "gendil/FiniteElementMethod/Restrictions/restrictions.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/Restrictions/doflayout.hpp" \
   "gendil/FiniteElementMethod/Restrictions/restrictions.hpp"
migration_replace_literal \
   "gendil/Utilities/kernelcontext.hpp" \
   "gendil/Utilities/KernelContext/kernelconfiguration.hpp"
migration_replace_literal \
   "gendil/Meshes/Connectivities/orientation.hpp" \
   "gendil/Meshes/Connectivities/Orientations/orientations.hpp"
migration_replace_literal \
   "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/facereaddofspolicy.hpp" \
   "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/dofio.hpp"
migration_replace_literal \
   "Linfprojectionoperator.hpp" \
   "nodalsubspaceprojectionoperator.hpp"

migration_warn_regex \
   "Review MakeIntegrationDomain<Name>(...): replace a homogeneous finite element space with its Mesh, or a MixedFiniteElementSpace with its Partition. mixed.GetPartition() is valid as a manual transitional expression; finite-element-space arguments now fail compilation" \
   'MakeIntegrationDomain[[:space:]]*<'

migration_warn_regex \
   "Remove explicit sparse-storage Free* calls: BSR/COO/CSR/CSC matrices and RawCOO buffers/layouts are move-only RAII owners in v0.0.6" \
   'Free(BSR|COO|CSR|CSC)Matrix|FreeRawCOO(TripletBuffer|AssemblyLayout)'

migration_warn_regex \
   "Remove the temporary owning sparse .data layer: metadata remains directly on the owner, while storage access now requires a GetHost*/GetDevice* raw-pointer view" \
   '\.data\.(block_rows|block_cols|num_rows|num_cols|num_row_blocks|num_col_blocks|num_blocks|nnz|nnz_raw|values|row_offsets|col_indices|rows|cols|row_ptr|col_ind|col_ptr|row_ind)\b'

migration_warn_regex \
   "Review direct sparse pointer access: owner arrays are now SyncHostDeviceArray objects; acquire a matrix view or call Read*/ReadWrite*/Write* on the individual array" \
   '\.(values|row_offsets|col_indices|rows|cols|row_ptr|col_ind|col_ptr|row_ind)\.(host_pointer|device_pointer)\b'

migration_warn_regex \
   "Replace ZeroBasedElementToGlobalDofIndex with an explicit occurrence ordinal for contiguous L2 policy tests, or use GetGlobalDofIndex when the final algebraic coordinate is required" \
   '\bZeroBasedElementToGlobalDofIndex\b'

migration_warn_regex \
   "Replace ScalarElementDofOrdinalToGlobalDofIndex by explicitly splitting the occurrence ordinal into element/local ordinals and calling GetGlobalDofIndex" \
   '\bScalarElementDofOrdinalToGlobalDofIndex\b'

migration_warn_regex \
   "Replace unchecked VectorOffset with CheckedVectorComponentOffset<ShapeFunctions, Component>(num_elements)" \
   '\bVectorOffset\b'

migration_warn_regex \
   "Replace restriction_traits<R>::is_injective or restriction_is_injective_v<R> with !restriction_may_share_global_dofs_v<R>; review the surrounding traversal/write policy independently" \
   '\brestriction_traits\b|\brestriction_is_injective_v\b'

migration_warn_regex \
   "Replace obsolete broad restriction-family classification with the corresponding structural concept or specific live mathematical/backend capability" \
   '\bis_(?:vector|tensor_product|h1)_restriction(?:_v)?\b'

migration_warn_regex \
   "Port legacy ThreadedDim/SequentialDim and KernelContext<Dims...> code to ThreadBlockLayout plus a modern host or device KernelConfiguration; replacing the removed header alone is insufficient" \
   '\b(ThreadedDim|SequentialDim)\b'

migration_warn_regex \
   "Replace removed face-DoF test/oracle helpers with direct orientation or view operations; FaceReadDofsOrientationIsIdentity, FaceReadDofsFIFOOffset, FaceReadDofsGlobalValueAt, and FaceReadDofsSignedIndex have no compatibility aliases" \
   '\b(FaceReadDofsOrientationIsIdentity|FaceReadDofsFIFOOffset|FaceReadDofsGlobalValueAt|FaceReadDofsSignedIndex)\b'

migration_finish
