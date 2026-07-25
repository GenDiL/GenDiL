#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MIGRATION_NAME="GenDiL v0.0.5 to v0.0.6 migration"
MIGRATION_DESCRIPTION="Report integration-domain calls requiring a semantic mesh/partition migration."

. "${SCRIPT_DIR}/common.sh"

migration_parse_args "$@"

migration_warn_regex \
   "Review MakeIntegrationDomain<Name>(...): replace a homogeneous finite element space with its Mesh, or a MixedFiniteElementSpace with its Partition. mixed.GetPartition() is valid as a manual transitional expression; finite-element-space arguments now fail compilation" \
   'MakeIntegrationDomain[[:space:]]*<'

migration_finish
