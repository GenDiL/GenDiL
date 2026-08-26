// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"

#include <type_traits>

namespace gendil {

namespace details
{

/**
 * @brief Allocate and zero a RawCOO triplet buffer for sparse assembly.
 *
 * Zero initialization makes every reserved slot well-defined even when a
 * contribution leaves an entry unwritten.  Initialization runs in the memory
 * space selected by @c OnDevice.
 *
 * @tparam OnDevice Whether to initialize through a device-accessible view.
 * @tparam ValueType Scalar type stored in the values array.
 * @tparam IndexType Scalar type used for dimensions and coordinates.
 * @param num_rows Algebraic test-space extent.
 * @param num_cols Algebraic trial-space extent.
 * @param nnz_raw Number of raw triplet slots to allocate.
 * @return An owning, zero-initialized RawCOO triplet buffer.
 */
template <
   bool OnDevice,
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
RawCOOTripletBuffer< ValueType, IndexType >
MakeAssemblyRawCOOTripletBuffer(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz_raw )
{
   auto buffer =
      AllocateRawCOOTripletBuffer< ValueType, IndexType >(
         num_rows,
         num_cols,
         nnz_raw );
   auto view = GetKernelWriteView< OnDevice >( buffer );
   KernelLoop< OnDevice >(
      nnz_raw,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         view.rows[i] = IndexType( 0 );
         view.cols[i] = IndexType( 0 );
         view.values[i] = ValueType( 0 );
      } );
   return buffer;
}

/**
 * @brief Build one cell-assembly layout per selected partition cell part.
 *
 * Tuple entry @c Part uses that part's mesh and reserves a dense
 * `ntest(Part) * ntrial(Part)` block for every selected cell.  Facet storage is
 * disabled because partition facet parts are traversed independently.
 *
 * @tparam WeakForm Weak-form type controlling whether cell terms are enabled.
 * @tparam DomainPartition Partition type exposing the selected cell parts.
 * @tparam TrialMixedSpace Mixed trial-space type.
 * @tparam TestMixedSpace Mixed test-space type.
 * @tparam Part Compile-time sequence selecting and ordering the cell parts.
 * @param partition Integration-domain partition supplying the cell meshes.
 * @param trial_space Mixed trial space indexed by partition cell part.
 * @param test_space Mixed test space indexed by partition cell part.
 * @return A tuple of RawCOOAssemblyLayout objects in @c Part order.
 */
template<
   class WeakForm,
   class DomainPartition,
   class TrialMixedSpace,
   class TestMixedSpace,
   size_t... Part>
auto MakePartitionCellRawCOOLayouts(
   const DomainPartition& partition,
   const TrialMixedSpace& trial_space,
   const TestMixedSpace& test_space,
   std::index_sequence<Part...>)
{
   return std::tuple{
      MakeRawCOOAssemblyLayout<
         has_cell_contributions_v<WeakForm>,
         false,
         false>(
            std::get<Part>(partition.CellParts()).mesh,
            CheckedMultiply(
               LocalDofCount<
                  finite_element_space_shape_functions_t< decltype(
                     test_space.template GetCellFiniteElementSpace< Part >() ) > >(),
               LocalDofCount<
                  finite_element_space_shape_functions_t< decltype(
                     trial_space.template GetCellFiniteElementSpace< Part >() ) > >(),
               "Partition RawCOO local block size overflow."))...};
}

/**
 * @brief Build the uniform RawCOO layout for one boundary-face part.
 *
 * The face part's adjacent cell-part index selects the local trial and test
 * spaces.  Each face receives a dense test-by-trial block for that cell part.
 *
 * @tparam Part Index of the boundary-face part.
 * @tparam DomainPartition Partition type exposing boundary-face connectivity.
 * @tparam TrialMixedSpace Mixed trial-space type.
 * @tparam TestMixedSpace Mixed test-space type.
 * @param partition Integration-domain partition supplying face connectivity.
 * @param trial_space Mixed trial space indexed by cell part.
 * @param test_space Mixed test space indexed by cell part.
 * @return Entity-block layout for all faces in boundary part @c Part.
 */
template<
   size_t Part,
   class DomainPartition,
   class TrialMixedSpace,
   class TestMixedSpace>
auto MakePartitionBoundaryRawCOOLayout(
   const DomainPartition& partition,
   const TrialMixedSpace& trial_space,
   const TestMixedSpace& test_space)
{
   const auto& face_part =
      std::get<Part>(partition.BoundaryFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t CellPart = FacePart::cell_index;
   const auto& trial =
      trial_space.template GetCellFiniteElementSpace<CellPart>();
   const auto& test =
      test_space.template GetCellFiniteElementSpace<CellPart>();
   const GlobalIndex block_entry_count = CheckedMultiply(
      LocalDofCount<finite_element_space_shape_functions_t<decltype(test)>>(),
      LocalDofCount<finite_element_space_shape_functions_t<decltype(trial)>>(),
      "Partition RawCOO boundary block size overflow.");
   return MakeRawCOOEntityBlockLayout(
      face_part.face_mesh.GetNumberOfFaces(),
      block_entry_count);
}

/**
 * @brief Build boundary-face RawCOO layouts in partition tuple order.
 *
 * @tparam DomainPartition Partition type exposing boundary-face parts.
 * @tparam TrialMixedSpace Mixed trial-space type.
 * @tparam TestMixedSpace Mixed test-space type.
 * @tparam Part Compile-time sequence selecting and ordering boundary parts.
 * @param partition Integration-domain partition supplying boundary-face parts.
 * @param trial_space Mixed trial space indexed by cell part.
 * @param test_space Mixed test space indexed by cell part.
 * @return A tuple of RawCOOEntityBlockLayout objects in @c Part order.
 */
template<
   class DomainPartition,
   class TrialMixedSpace,
   class TestMixedSpace,
   size_t... Part>
auto MakePartitionBoundaryRawCOOLayouts(
   const DomainPartition& partition,
   const TrialMixedSpace& trial_space,
   const TestMixedSpace& test_space,
   std::index_sequence<Part...>)
{
   return std::tuple{
      MakePartitionBoundaryRawCOOLayout<Part>(
         partition,
         trial_space,
         test_space)...};
}

/**
 * @brief Build the four directed RawCOO layouts for one interior-face part.
 *
 * The first sign in each returned block denotes the test side and the second
 * denotes the trial side.  Consequently, heterogeneous minus and plus spaces
 * produce independently sized `--`, `-+`, `+-`, and `++` dense blocks.  Every
 * block contains one segment for each face in the selected part.
 *
 * @tparam Part Index of the interior-face part.
 * @tparam DomainPartition Partition type exposing interior-face connectivity.
 * @tparam TrialMixedSpace Mixed trial-space type.
 * @tparam TestMixedSpace Mixed test-space type.
 * @param partition Integration-domain partition supplying face connectivity.
 * @param trial_space Mixed trial space indexed by cell part.
 * @param test_space Mixed test space indexed by cell part.
 * @return Four entity-block layouts for interior-face part @c Part.
 */
template<
   size_t Part,
   class DomainPartition,
   class TrialMixedSpace,
   class TestMixedSpace>
auto MakePartitionInteriorRawCOOLayout(
   const DomainPartition& partition,
   const TrialMixedSpace& trial_space,
   const TestMixedSpace& test_space)
{
   const auto& face_part =
      std::get<Part>(partition.InteriorFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t MinusPart = FacePart::minus_cell_index;
   constexpr size_t PlusPart = FacePart::plus_cell_index;
   const auto& trial_minus =
      trial_space.template GetCellFiniteElementSpace<MinusPart>();
   const auto& trial_plus =
      trial_space.template GetCellFiniteElementSpace<PlusPart>();
   const auto& test_minus =
      test_space.template GetCellFiniteElementSpace<MinusPart>();
   const auto& test_plus =
      test_space.template GetCellFiniteElementSpace<PlusPart>();
   constexpr GlobalIndex ntrial_minus = LocalDofCount<
      finite_element_space_shape_functions_t<decltype(trial_minus)>>();
   constexpr GlobalIndex ntrial_plus = LocalDofCount<
      finite_element_space_shape_functions_t<decltype(trial_plus)>>();
   constexpr GlobalIndex ntest_minus = LocalDofCount<
      finite_element_space_shape_functions_t<decltype(test_minus)>>();
   constexpr GlobalIndex ntest_plus = LocalDofCount<
      finite_element_space_shape_functions_t<decltype(test_plus)>>();
   const GlobalIndex num_faces = face_part.face_mesh.GetNumberOfFaces();

   return RawCOOInteriorFaceTargets{
      MakeRawCOOEntityBlockLayout(
         num_faces,
         CheckedMultiply(
            ntest_minus,
            ntrial_minus,
            "Partition RawCOO -- block size overflow.")),
      MakeRawCOOEntityBlockLayout(
         num_faces,
         CheckedMultiply(
            ntest_minus,
            ntrial_plus,
            "Partition RawCOO -+ block size overflow.")),
      MakeRawCOOEntityBlockLayout(
         num_faces,
         CheckedMultiply(
            ntest_plus,
            ntrial_minus,
            "Partition RawCOO +- block size overflow.")),
      MakeRawCOOEntityBlockLayout(
         num_faces,
         CheckedMultiply(
            ntest_plus,
            ntrial_plus,
            "Partition RawCOO ++ block size overflow."))};
}

/**
 * @brief Build four-block interior-face layouts in partition tuple order.
 *
 * @tparam DomainPartition Partition type exposing interior-face parts.
 * @tparam TrialMixedSpace Mixed trial-space type.
 * @tparam TestMixedSpace Mixed test-space type.
 * @tparam Part Compile-time sequence selecting and ordering interior parts.
 * @param partition Integration-domain partition supplying interior-face parts.
 * @param trial_space Mixed trial space indexed by cell part.
 * @param test_space Mixed test space indexed by cell part.
 * @return A tuple of RawCOOInteriorFaceTargets layout aggregates.
 */
template<
   class DomainPartition,
   class TrialMixedSpace,
   class TestMixedSpace,
   size_t... Part>
auto MakePartitionInteriorRawCOOLayouts(
   const DomainPartition& partition,
   const TrialMixedSpace& trial_space,
   const TestMixedSpace& test_space,
   std::index_sequence<Part...>)
{
   return std::tuple{
      MakePartitionInteriorRawCOOLayout<Part>(
         partition,
         trial_space,
         test_space)...};
}

/**
 * @brief Return the triplet capacity of a cell-oriented RawCOO layout.
 * @param layout Layout whose capacity is requested.
 * @return Number of reserved raw triplets.
 */
inline GlobalIndex RawCOOLayoutCapacity(
   const RawCOOAssemblyLayout& layout)
{
   return layout.nnz_raw;
}

/**
 * @brief Return the triplet capacity of a uniform entity-block layout.
 * @param layout Layout whose capacity is requested.
 * @return Number of reserved raw triplets.
 */
inline GlobalIndex RawCOOLayoutCapacity(
   const RawCOOEntityBlockLayout& layout)
{
   return layout.nnz_raw;
}

/**
 * @brief Sum the capacities of an interior part's four directed blocks.
 * @param layout Four-block interior-face layout aggregate.
 * @return Total raw-triplet capacity of `--`, `-+`, `+-`, and `++`.
 *
 * A verification failure is reported if the sum overflows GlobalIndex.
 */
template<class MM, class MP, class PM, class PP>
GlobalIndex RawCOOLayoutCapacity(
   const RawCOOInteriorFaceTargets<MM, MP, PM, PP>& layout)
{
   GlobalIndex count = 0;
   count = CheckedAdd(
      count,
      RawCOOLayoutCapacity(layout.minus_minus),
      "Partition RawCOO capacity overflow.");
   count = CheckedAdd(
      count,
      RawCOOLayoutCapacity(layout.minus_plus),
      "Partition RawCOO capacity overflow.");
   count = CheckedAdd(
      count,
      RawCOOLayoutCapacity(layout.plus_minus),
      "Partition RawCOO capacity overflow.");
   return CheckedAdd(
      count,
      RawCOOLayoutCapacity(layout.plus_plus),
      "Partition RawCOO capacity overflow.");
}

/**
 * @brief Sum the RawCOO capacities of a compile-time layout sequence.
 *
 * @param layouts Tuple containing cell, boundary, or interior layouts.
 * @tparam Part Indices to include, in any caller-selected order.
 * @return Total raw-triplet capacity of the selected tuple entries.
 *
 * A verification failure is reported if an intermediate sum overflows
 * GlobalIndex.
 */
template<class Layouts, size_t... Part>
GlobalIndex PartitionRawCOONonzeroCount(
   const Layouts& layouts,
   std::index_sequence<Part...>)
{
   GlobalIndex count = 0;
   ((count = CheckedAdd(
        count,
        RawCOOLayoutCapacity(std::get<Part>(layouts)),
        "Partition RawCOO capacity overflow.")), ...);
   return count;
}

/**
 * @brief Offset a RawCOO storage pointer while preserving null pointers.
 *
 * Empty triplet buffers may expose null storage pointers.  Keeping null
 * pointers null avoids applying pointer arithmetic to them when zero-capacity
 * partition segments are constructed.
 *
 * @param pointer Pointer to the beginning of a triplet array, or null.
 * @param offset Number of entries by which to advance a non-null pointer.
 * @return @p pointer plus @p offset, or null when @p pointer is null.
 */
template<typename Pointer>
Pointer OffsetRawCOOPointer(Pointer pointer, const GlobalIndex offset)
{
   return pointer == nullptr ? nullptr : pointer + offset;
}

/**
 * @brief Carve the next non-owning slice from a monolithic triplet view.
 *
 * The returned slice retains the complete matrix dimensions and contains
 * exactly @p count entries beginning at @p next_offset.  On return,
 * @p next_offset points immediately past the slice.
 *
 * @param triplets Complete RawCOO triplet view to partition.
 * @param count Number of entries assigned to the new slice.
 * @param next_offset In/out cursor into @p triplets.
 * @return A borrowed view of the selected contiguous segment.
 *
 * The caller must ensure the requested segment lies within @p triplets.  A
 * verification failure is reported if advancing the cursor overflows
 * GlobalIndex.
 */
template<typename TripletView>
auto MakeRawCOOTripletSlice(
   const TripletView& triplets,
   const GlobalIndex count,
   GlobalIndex& next_offset)
{
   auto slice = RawCOOTripletView<
      typename TripletView::value_type,
      typename TripletView::index_type>{
         triplets.num_rows,
         triplets.num_cols,
         count,
         OffsetRawCOOPointer(triplets.rows, next_offset),
         OffsetRawCOOPointer(triplets.cols, next_offset),
         OffsetRawCOOPointer(triplets.values, next_offset)};
   next_offset = CheckedAdd(
      next_offset,
      count,
      "Partition RawCOO target offset overflow.");
   return slice;
}

/**
 * @brief Bind a cell-oriented layout to the next RawCOO buffer segment.
 *
 * @tparam OnDevice Selects the host or device read view of the layout offsets.
 * @param triplets Complete monolithic triplet view.
 * @param layout Cell-oriented layout to bind.
 * @param next_offset In/out cursor advanced by @c layout.nnz_raw.
 * @return A borrowed RawCOOAssemblyTarget for the selected segment.
 */
template<bool OnDevice, typename TripletView>
auto MakePartitionRawCOOTarget(
   const TripletView& triplets,
   const RawCOOAssemblyLayout& layout,
   GlobalIndex& next_offset)
{
   return MakeRawCOOAssemblyTarget(
      MakeRawCOOTripletSlice(triplets, layout.nnz_raw, next_offset),
      GetKernelReadView<OnDevice>(layout));
}

/**
 * @brief Bind a uniform entity-block layout to the next RawCOO buffer segment.
 *
 * This layout has no auxiliary offset arrays, so @c OnDevice affects only the
 * overload interface and no layout-memory conversion is required.
 *
 * @tparam OnDevice Retained for a uniform host/device target-building API.
 * @param triplets Complete monolithic triplet view.
 * @param layout Uniform entity-block layout to bind.
 * @param next_offset In/out cursor advanced by @c layout.nnz_raw.
 * @return A borrowed RawCOOEntityBlockTarget for the selected segment.
 */
template<bool OnDevice, typename TripletView>
auto MakePartitionRawCOOTarget(
   const TripletView& triplets,
   const RawCOOEntityBlockLayout& layout,
   GlobalIndex& next_offset)
{
   (void)OnDevice;
   return MakeRawCOOEntityBlockTarget(
      MakeRawCOOTripletSlice(triplets, layout.nnz_raw, next_offset),
      layout);
}

/**
 * @brief Recursively bind a tuple of layouts to consecutive buffer segments.
 *
 * Targets preserve the layout tuple's order and may have distinct types and
 * capacities.  The recursion returns an empty tuple after the last part.
 *
 * @tparam OnDevice Selects kernel-accessible layout views where required.
 * @tparam Part Index of the next layout to bind.
 * @param triplets Complete monolithic triplet view.
 * @param layouts Tuple of cell or boundary layouts.
 * @param next_offset In/out cursor shared by all generated targets.
 * @return Tuple of borrowed targets for entries `[Part, tuple_size)`.
 */
template<
   bool OnDevice,
   size_t Part,
   typename TripletView,
   typename Layouts>
auto MakePartitionRawCOOTargetTuple(
   const TripletView& triplets,
   const Layouts& layouts,
   GlobalIndex& next_offset)
{
   constexpr size_t NumParts =
      std::tuple_size_v<std::remove_cvref_t<Layouts>>;
   if constexpr (Part == NumParts)
   {
      return std::tuple{};
   }
   else
   {
      auto target = MakePartitionRawCOOTarget<OnDevice>(
         triplets,
         std::get<Part>(layouts),
         next_offset);
      return std::tuple_cat(
         std::tuple{target},
         MakePartitionRawCOOTargetTuple<OnDevice, Part + 1>(
            triplets,
            layouts,
            next_offset));
   }
}

/**
 * @brief Bind one interior part's four layouts to consecutive buffer segments.
 *
 * Segments are assigned in `--`, `-+`, `+-`, `++` order.  The first sign is
 * the test side and the second sign is the trial side.
 *
 * @tparam OnDevice Selects kernel-accessible layout views where required.
 * @param triplets Complete monolithic triplet view.
 * @param layouts Four directed layouts for one interior-face part.
 * @param next_offset In/out cursor advanced across all four segments.
 * @return Four borrowed RawCOO targets preserving the quadrant ordering.
 */
template<bool OnDevice, typename TripletView, class MM, class MP, class PM, class PP>
auto MakePartitionInteriorRawCOOTarget(
   const TripletView& triplets,
   const RawCOOInteriorFaceTargets<MM, MP, PM, PP>& layouts,
   GlobalIndex& next_offset)
{
   auto minus_minus = MakePartitionRawCOOTarget<OnDevice>(
      triplets, layouts.minus_minus, next_offset);
   auto minus_plus = MakePartitionRawCOOTarget<OnDevice>(
      triplets, layouts.minus_plus, next_offset);
   auto plus_minus = MakePartitionRawCOOTarget<OnDevice>(
      triplets, layouts.plus_minus, next_offset);
   auto plus_plus = MakePartitionRawCOOTarget<OnDevice>(
      triplets, layouts.plus_plus, next_offset);
   return RawCOOInteriorFaceTargets{
      minus_minus,
      minus_plus,
      plus_minus,
      plus_plus};
}

/**
 * @brief Recursively bind all interior-part layouts to consecutive segments.
 *
 * Interior parts preserve tuple order, and each part consumes its four
 * directed blocks before the next part begins.
 *
 * @tparam OnDevice Selects kernel-accessible layout views where required.
 * @tparam Part Index of the next interior layout aggregate to bind.
 * @param triplets Complete monolithic triplet view.
 * @param layouts Tuple of four-block interior-face layouts.
 * @param next_offset In/out cursor shared by all generated targets.
 * @return Tuple of interior target groups for entries `[Part, tuple_size)`.
 */
template<
   bool OnDevice,
   size_t Part,
   typename TripletView,
   typename Layouts>
auto MakePartitionInteriorRawCOOTargetTuple(
   const TripletView& triplets,
   const Layouts& layouts,
   GlobalIndex& next_offset)
{
   constexpr size_t NumParts =
      std::tuple_size_v<std::remove_cvref_t<Layouts>>;
   if constexpr (Part == NumParts)
   {
      return std::tuple{};
   }
   else
   {
      auto target = MakePartitionInteriorRawCOOTarget<OnDevice>(
         triplets,
         std::get<Part>(layouts),
         next_offset);
      return std::tuple_cat(
         std::tuple{target},
         MakePartitionInteriorRawCOOTargetTuple<OnDevice, Part + 1>(
            triplets,
            layouts,
            next_offset));
   }
}

/**
 * @brief Bind all partition layouts to one monolithic RawCOO buffer.
 *
 * The buffer is segmented first by cell part, then boundary-face part, then
 * interior-face part.  Each interior part is further ordered `--`, `-+`, `+-`,
 * `++`.  The returned targets borrow the buffer's host- or device-accessible
 * storage and retain its full row and column extents.
 *
 * @tparam OnDevice Selects the buffer and layout views used by kernels.
 * @param buffer Owning triplet buffer to subdivide.
 * @param cell_layouts Cell-part layouts in partition order.
 * @param boundary_layouts Boundary-face layouts in partition order.
 * @param interior_layouts Interior-face layouts in partition order.
 * @return Non-owning cell, boundary, and interior assembly targets.
 *
 * A verification failure is reported unless the layouts cover the buffer
 * capacity exactly.
 */
template<bool OnDevice, typename Buffer, typename CellLayouts,
   typename BoundaryLayouts, typename InteriorLayouts>
auto MakePartitionRawCOOTargetBundle(
   Buffer& buffer,
   const CellLayouts& cell_layouts,
   const BoundaryLayouts& boundary_layouts,
   const InteriorLayouts& interior_layouts)
{
   auto triplets = GetKernelReadWriteView<OnDevice>(buffer);
   GlobalIndex next_offset = 0;
   auto cells = MakePartitionRawCOOTargetTuple<OnDevice, 0>(
      triplets, cell_layouts, next_offset);
   auto boundaries = MakePartitionRawCOOTargetTuple<OnDevice, 0>(
      triplets, boundary_layouts, next_offset);
   auto interiors = MakePartitionInteriorRawCOOTargetTuple<OnDevice, 0>(
      triplets, interior_layouts, next_offset);
   GENDIL_VERIFY(
      next_offset == static_cast<GlobalIndex>(triplets.nnz_raw),
      "Partition RawCOO targets do not cover the allocated triplet buffer.");
   return PartitionRawCOOAssemblyTargets{
      cells,
      boundaries,
      interiors};
}

/**
 * @brief Validate restriction mappings in the assembly execution space.
 *
 * A scalar finite-element space is checked directly.  For a mixed space, each
 * cell-part restriction is checked independently against that part's element
 * count.  This catches host-only mappings before a device kernel is launched,
 * and device-only mappings before host assembly uses them.
 *
 * @tparam OnDevice Execution memory space required by the assembly policy.
 * @param space Trial or test finite-element space to validate.
 */
template<bool OnDevice, class Space>
void ValidateRawCOORestrictionMemoryAccess(const Space& space)
{
   using SpaceType = std::remove_cvref_t<Space>;
   if constexpr (is_mixed_finite_element_space_v<SpaceType>)
   {
      ConstexprLoop<SpaceType::num_cell_spaces>(
         [&] (auto part)
         {
            constexpr size_t Part = decltype(part)::value;
            const auto& cell_space =
               space.template GetCellFiniteElementSpace<Part>();
            ValidateRestrictionMemoryAccess<OnDevice>(
               GetRestriction(cell_space),
               cell_space.GetNumberOfFiniteElements());
         });
   }
   else
   {
      ValidateRestrictionMemoryAccess<OnDevice>(
         GetRestriction(space),
         space.GetNumberOfFiniteElements());
   }
}

/**
 * @brief Bind a layout tuple that must consume an entire RawCOO buffer.
 *
 * This convenience overload creates a kernel-accessible buffer view, binds the
 * layouts from offset zero, and verifies exact capacity coverage.
 *
 * @tparam OnDevice Selects the buffer and layout views used by kernels.
 * @param buffer Owning triplet buffer to subdivide.
 * @param layouts Tuple of layouts in desired segment order.
 * @return Tuple of non-owning targets corresponding to @p layouts.
 */
template<bool OnDevice, typename Buffer, typename Layouts>
auto MakePartitionRawCOOTargetTuple(
   Buffer& buffer,
   const Layouts& layouts)
{
   auto triplets = GetKernelReadWriteView<OnDevice>(buffer);
   GlobalIndex next_offset = 0;
   auto targets = MakePartitionRawCOOTargetTuple<OnDevice, 0>(
      triplets,
      layouts,
      next_offset);
   GENDIL_VERIFY(
      next_offset == static_cast<GlobalIndex>(triplets.nnz_raw),
      "Partition RawCOO targets do not cover the allocated triplet buffer.");
   return targets;
}

} // namespace details

/**
 * @brief Assemble a bilinear weak form into an owning RawCOO triplet buffer.
 *
 * Rows use the test field's common algebraic extent and columns use the trial
 * field's common algebraic extent, so rectangular forms are supported.  Raw
 * entries are emitted without sorting or duplicate reduction; downstream COO,
 * CSR, CSC, and other finalizers retain responsibility for canonicalization.
 *
 * For ordinary spaces, one cell-oriented layout covers all enabled cell,
 * boundary, and interior contributions.  For mixed partition spaces, one
 * segment is created per cell part and boundary-face part, plus four directed
 * segments per interior-face part.  All segments share one monolithic buffer.
 * Restriction accessibility and sparse-assembly contracts are validated before
 * kernels run.  Device execution is synchronized before the buffer is returned.
 *
 * @tparam KernelPolicy Host or device kernel execution policy.
 * @tparam WeakForm Bilinear weak-form type.
 * @tparam WeakFormContext Context type providing fields and domains.
 * @tparam IntegrationRule Quadrature-rule type.
 * @param weak_form Bilinear weak form to assemble.
 * @param wf_ctx Context containing active fields and integration domains.
 * @param integration_rule Quadrature rule used by the assembly kernels.
 * @return Owning RawCOOTripletBuffer with global test-by-trial dimensions.
 */
template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericRawCOOAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::RawCOO,
         details::SparseAssemblyMode::Full,
         KernelPolicy>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName  = requirements<WeakForm>::test_name;

      const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space  = wf_ctx.template fe_field<TestName>().space;

      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;
      constexpr bool on_device =
         is_device_configuration_v< KernelPolicy >;
      details::ValidateRawCOORestrictionMemoryAccess<on_device>(trial_space);
      details::ValidateRawCOORestrictionMemoryAccess<on_device>(test_space);
      if constexpr (
         is_mixed_finite_element_space_v<TrialSpace> ||
         is_mixed_finite_element_space_v<TestSpace>)
      {
         static_assert(
            is_mixed_finite_element_space_v<TrialSpace> &&
               is_mixed_finite_element_space_v<TestSpace>,
            "Partition RawCOO requires both active spaces to be mixed spaces.");
         static_assert(
            TrialSpace::num_cell_spaces == TestSpace::num_cell_spaces,
            "Partition RawCOO trial and test fields must have the same part count.");
         constexpr auto DomainName =
            local_facet_assembly_domain_name_v<WeakForm>;
         const auto& domain = wf_ctx.template domain<DomainName>();
         const auto& partition = domain.domain.partition;
         using DomainPartition =
            std::remove_cvref_t<decltype(partition)>;
         constexpr size_t NumParts = TrialSpace::num_cell_spaces;
         static_assert(
            DomainPartition::num_cell_parts == NumParts,
            "Partition RawCOO integration and active field spaces must have "
            "the same cell-part count.");
         auto cell_layouts = details::MakePartitionCellRawCOOLayouts<WeakForm>(
            partition,
            trial_space,
            test_space,
            std::make_index_sequence<NumParts>{});
         const auto& boundary_parts = partition.BoundaryFaceParts();
         const auto& interior_parts = partition.InteriorFaceParts();
         constexpr size_t NumBoundaryParts = std::tuple_size_v<
            std::remove_cvref_t<decltype(boundary_parts)>>;
         constexpr size_t NumInteriorParts = std::tuple_size_v<
            std::remove_cvref_t<decltype(interior_parts)>>;
         auto boundary_layouts = [&]
         {
            if constexpr (has_boundary_facet_contributions_v<WeakForm>)
            {
               return details::MakePartitionBoundaryRawCOOLayouts(
                  partition,
                  trial_space,
                  test_space,
                  std::make_index_sequence<NumBoundaryParts>{});
            }
            else
            {
               return std::tuple{};
            }
         }();
         auto interior_layouts = [&]
         {
            if constexpr (has_interior_facet_contributions_v<WeakForm>)
            {
               return details::MakePartitionInteriorRawCOOLayouts(
                  partition,
                  trial_space,
                  test_space,
                  std::make_index_sequence<NumInteriorParts>{});
            }
            else
            {
               return std::tuple{};
            }
         }();
         GlobalIndex nnz_raw = details::PartitionRawCOONonzeroCount(
            cell_layouts,
            std::make_index_sequence<NumParts>{});
         nnz_raw = CheckedAdd(
            nnz_raw,
            details::PartitionRawCOONonzeroCount(
               boundary_layouts,
               std::make_index_sequence<
                  std::tuple_size_v<decltype(boundary_layouts)>>{}),
            "Partition RawCOO capacity overflow.");
         nnz_raw = CheckedAdd(
            nnz_raw,
            details::PartitionRawCOONonzeroCount(
               interior_layouts,
               std::make_index_sequence<
                  std::tuple_size_v<decltype(interior_layouts)>>{}),
            "Partition RawCOO capacity overflow.");
         auto coo_buffer =
            details::MakeAssemblyRawCOOTripletBuffer<
               on_device,
               Real,
               GlobalIndex>(
                  GetAlgebraicDofExtent(test_space),
                  GetAlgebraicDofExtent(trial_space),
                  nnz_raw);
         auto coo_targets =
            details::MakePartitionRawCOOTargetBundle<on_device>(
               coo_buffer,
               cell_layouts,
               boundary_layouts,
               interior_layouts);
         GenericAssembly<KernelPolicy>(
            weak_form,
            wf_ctx,
            integration_rule,
            coo_targets);
         if constexpr (!is_host_configuration_v<KernelPolicy>)
         {
            GENDIL_DEVICE_SYNC;
         }
         return coo_buffer;
      }
      else
      {
         using TrialShapeFunctions =
            finite_element_space_shape_functions_t< TrialSpace >;
         using TestShapeFunctions =
            finite_element_space_shape_functions_t< TestSpace >;
         using OffsetType = RawCOOAssemblyLayout::offset_type;
         constexpr OffsetType ntrial = static_cast<OffsetType>(
            LocalDofCount<TrialShapeFunctions>());
         constexpr OffsetType ntest = static_cast<OffsetType>(
            LocalDofCount<TestShapeFunctions>());
         const OffsetType block_entry_count = CheckedMultiply(
            ntest,
            ntrial,
            "RawCOO local trial/test block size overflow.");
         const auto& domain_mesh =
            GetCellIntegrationDomainMesh(weak_form, wf_ctx);
         auto layout = MakeRawCOOAssemblyLayout<
            has_cell_contributions_v<WeakForm>,
            has_boundary_facet_contributions_v<WeakForm>,
            has_interior_facet_contributions_v<WeakForm>>(
               domain_mesh,
               block_entry_count);
         auto coo_buffer =
            details::MakeAssemblyRawCOOTripletBuffer<
               on_device,
               Real,
               GlobalIndex>(
                  GetAlgebraicDofExtent(test_space),
                  GetAlgebraicDofExtent(trial_space),
                  layout.nnz_raw);
         auto coo_target =
            MakeRawCOOAssemblyTarget<on_device>(coo_buffer, layout);
         GenericAssembly<KernelPolicy>(
            weak_form,
            wf_ctx,
            integration_rule,
            coo_target);
         if constexpr (!is_host_configuration_v<KernelPolicy>)
         {
            GENDIL_DEVICE_SYNC;
         }
         return coo_buffer;
      }
   }
}

/**
 * @brief Assemble only element-block-diagonal contributions into RawCOO.
 *
 * The layout reserves one dense local test-by-trial block per element while
 * accounting for any enabled facet terms in the element-local operator.  The
 * result is an unfinalized RawCOO triplet buffer with global test-by-trial
 * dimensions.  Sparse-assembly compatibility is validated before template
 * instantiation, and device work is synchronized before return.
 *
 * This path uses the single-space cell-oriented layout; partition/mixed-space
 * assembly is provided by GenericRawCOOAssembly rather than this routine.
 *
 * @tparam KernelPolicy Host or device kernel execution policy.
 * @tparam WeakForm Bilinear weak-form type.
 * @tparam WeakFormContext Context type providing fields and domains.
 * @tparam IntegrationRule Quadrature-rule type.
 * @param weak_form Bilinear weak form to assemble.
 * @param wf_ctx Context containing active fields and the integration domain.
 * @param integration_rule Quadrature rule used by the assembly kernels.
 * @return Owning element-block-diagonal RawCOOTripletBuffer.
 */
template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericRawCOOElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule )
{
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::RawCOO,
         details::SparseAssemblyMode::ElementBlockDiagonal,
         KernelPolicy>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName = requirements<WeakForm>::test_name;

      const auto& trial_space =
         wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space =
         wf_ctx.template fe_field<TestName>().space;

      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;
      using TrialShapeFunctions =
         finite_element_space_shape_functions_t< TrialSpace >;
      using TestShapeFunctions =
         finite_element_space_shape_functions_t< TestSpace >;

      using OffsetType = RawCOOAssemblyLayout::offset_type;
      constexpr OffsetType ntrial =
         static_cast<OffsetType>(LocalDofCount<TrialShapeFunctions>());
      constexpr OffsetType ntest =
         static_cast<OffsetType>(LocalDofCount<TestShapeFunctions>());
      const OffsetType block_entry_count =
         CheckedMultiply(
            ntest,
            ntrial,
            "Element RawCOO local trial/test block size overflow.");
      const auto& domain_mesh =
         GetCellIntegrationDomainMesh(weak_form, wf_ctx);

      auto layout =
         MakeRawCOOElementBlockDiagonalAssemblyLayout<
            has_cell_contributions_v<WeakForm>,
            has_boundary_facet_contributions_v<WeakForm>,
            has_interior_facet_contributions_v<WeakForm>>(
               domain_mesh,
               block_entry_count);

      constexpr bool on_device =
         is_device_configuration_v< KernelPolicy >;
      auto coo_buffer =
         details::MakeAssemblyRawCOOTripletBuffer<
            on_device,
            Real,
            GlobalIndex >(
               static_cast<GlobalIndex>(
                  GetAlgebraicDofExtent(test_space)),
               static_cast<GlobalIndex>(
                  GetAlgebraicDofExtent(trial_space)),
               layout.nnz_raw);
      auto coo_target =
         MakeRawCOOAssemblyTarget< on_device >( coo_buffer, layout );

      AssembleElementBlockDiagonalSparseTarget<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         coo_target);

      if constexpr ( !is_host_configuration_v< KernelPolicy > )
      {
         GENDIL_DEVICE_SYNC;
      }

      return coo_buffer;
   }
}

} // namespace gendil
