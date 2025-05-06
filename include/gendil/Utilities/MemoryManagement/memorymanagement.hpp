// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// A pointer structure working both on host and device.
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

// A simple memory arena
#include "gendil/Utilities/MemoryManagement/memoryarena.hpp"

// A simple memory heap on the stack
#include "gendil/Utilities/MemoryManagement/fixedmemoryallocator.hpp"

// A simple garbage collector
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"
