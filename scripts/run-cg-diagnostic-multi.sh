#!/bin/bash
# Copyright GenDiL Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Script to build and run cg-diagnostic-1d with multiple Real types
# and save outputs to separate .tex files

set -e  # Exit on error (disabled for main loop to handle errors gracefully)

#======================================
# Configuration and Constants
#======================================

# Detect repo root (script is in scripts/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
DEFAULT_TYPES="double,float,bfloat16_t"
OUTPUT_DIR="$REPO_ROOT"
FORCE_REBUILD=false
COMPILE_PDF=false
CLEAN=false
VERBOSE=false

# Valid Real types
VALID_TYPES=("double" "float" "float16_t" "float32_t" "float64_t" "float128_t" "bfloat16_t")

# Track results
declare -a SUCCESSFUL_TYPES
declare -a FAILED_TYPES

#======================================
# Color-coded logging
#======================================

# Check if output is a terminal
if [[ -t 1 ]]; then
    COLOR_RESET="\033[0m"
    COLOR_RED="\033[0;31m"
    COLOR_GREEN="\033[0;32m"
    COLOR_YELLOW="\033[0;33m"
    COLOR_BLUE="\033[0;34m"
else
    COLOR_RESET=""
    COLOR_RED=""
    COLOR_GREEN=""
    COLOR_YELLOW=""
    COLOR_BLUE=""
fi

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $*"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" >&2
}

log_warning() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $*"
}

#======================================
# Helper Functions
#======================================

# Map Real type to build directory name
get_build_dir_for_type() {
    local type=$1
    case "$type" in
        double)     echo "build-default" ;;
        float)      echo "build-float" ;;
        bfloat16_t) echo "build-bf16" ;;
        float16_t)  echo "build-f16" ;;
        float32_t)  echo "build-f32" ;;
        float64_t)  echo "build-f64" ;;
        float128_t) echo "build-f128" ;;
        *)          echo "" ;;
    esac
}

# Get short name for output file (strip _t suffix)
get_short_type_name() {
    local type=$1
    echo "${type%_t}"
}

# Validate that type is in valid list
is_valid_type() {
    local type=$1
    for valid_type in "${VALID_TYPES[@]}"; do
        [[ "$type" == "$valid_type" ]] && return 0
    done
    return 1
}

# Ensure build directory exists and is configured
ensure_build_dir_exists() {
    local type=$1
    local build_dir=$2

    if [[ ! -d "$build_dir" ]]; then
        log_info "Creating build directory $build_dir for type $type"
        mkdir -p "$build_dir"
        (cd "$build_dir" && \
         cmake -DGENDIL_REAL_TYPE="$type" \
               -DCMAKE_BUILD_TYPE=Release \
               "$REPO_ROOT") || {
            log_error "Failed to configure build directory for $type"
            return 1
        }
    elif [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
        log_info "Configuring existing build directory $build_dir"
        (cd "$build_dir" && \
         cmake -DGENDIL_REAL_TYPE="$type" \
               -DCMAKE_BUILD_TYPE=Release \
               "$REPO_ROOT") || {
            log_error "Failed to configure build directory for $type"
            return 1
        }
    else
        log_info "Build directory $build_dir already configured"
    fi
    return 0
}

# Check if rebuild is needed
needs_rebuild() {
    local build_dir=$1
    local exe_path="$build_dir/examples/cg_convergence/cg-diagnostic-1d"
    local source_file="$REPO_ROOT/examples/cg_convergence/cg-diagnostic-1d.cpp"

    # Force rebuild flag overrides
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        log_info "Force rebuild requested"
        return 0
    fi

    # Executable doesn't exist
    if [[ ! -f "$exe_path" ]]; then
        log_info "Executable does not exist"
        return 0
    fi

    # Source newer than executable
    if [[ "$source_file" -nt "$exe_path" ]]; then
        log_info "Source file is newer than executable"
        return 0
    fi

    # No rebuild needed
    log_info "Executable is up-to-date"
    return 1
}

# Build the cg-diagnostic-1d target
build_target() {
    local type=$1
    local build_dir=$2

    log_info "Building cg-diagnostic-1d for $type in $build_dir"

    if [[ "$VERBOSE" == "true" ]]; then
        make -C "$build_dir" cg-diagnostic-1d
    else
        # Capture output, show only on error
        local build_output
        if build_output=$(make -C "$build_dir" cg-diagnostic-1d 2>&1); then
            return 0
        else
            log_error "Build failed for $type"
            echo "$build_output" | tail -20
            return 1
        fi
    fi
}

# Run executable and save output
run_and_capture() {
    local type=$1
    local build_dir=$2
    local output_file=$3

    local exe_path="$build_dir/examples/cg_convergence/cg-diagnostic-1d"

    if [[ ! -x "$exe_path" ]]; then
        log_error "Executable not found or not executable: $exe_path"
        return 1
    fi

    log_info "Running cg-diagnostic-1d for $type"

    if "$exe_path" > "$output_file" 2>&1; then
        log_success "Output saved to $output_file"
        return 0
    else
        log_error "Execution failed for $type"
        return 1
    fi
}

# Compile LaTeX to PDF
compile_pdf() {
    local tex_file=$1
    local pdf_file="${tex_file%.tex}.pdf"

    # Check if pdflatex is available
    if ! command -v pdflatex &> /dev/null; then
        log_error "pdflatex not found. Install texlive or set PATH."
        return 1
    fi

    log_info "Compiling $(basename "$tex_file") to PDF"

    local tex_dir="$(dirname "$tex_file")"
    local tex_base="$(basename "$tex_file")"

    # Run pdflatex in the output directory
    (cd "$tex_dir" && \
     pdflatex -interaction=nonstopmode "$tex_base" > /dev/null 2>&1)

    if [[ -f "$pdf_file" ]]; then
        log_success "Generated $(basename "$pdf_file")"
        # Clean up auxiliary files
        rm -f "${tex_file%.tex}.aux" "${tex_file%.tex}.log"
        return 0
    else
        log_error "Failed to compile $(basename "$tex_file")"
        return 1
    fi
}

# Check prerequisites
check_prerequisites() {
    local missing=0

    if ! command -v cmake &> /dev/null; then
        log_error "cmake not found"
        ((missing++))
    fi

    if ! command -v make &> /dev/null; then
        log_error "make not found"
        ((missing++))
    fi

    if ! command -v g++ &> /dev/null && ! command -v c++ &> /dev/null; then
        log_error "C++ compiler not found"
        ((missing++))
    fi

    if [[ "$COMPILE_PDF" == "true" ]] && ! command -v pdflatex &> /dev/null; then
        log_warning "pdflatex not found. PDF compilation will be skipped."
    fi

    if [[ $missing -gt 0 ]]; then
        return 1
    fi

    return 0
}

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build and run cg-diagnostic-1d with multiple Real types and save outputs.

OPTIONS:
    --types <list>        Comma-separated list of Real types to test
                          Default: $DEFAULT_TYPES
                          Valid: ${VALID_TYPES[*]}

    --output-dir <dir>    Directory to save .tex output files
                          Default: $OUTPUT_DIR

    --force-rebuild       Force rebuild even if executable is up-to-date

    --compile-pdf         Compile .tex files to PDF using pdflatex

    --clean               Remove existing .tex output files before running

    --verbose             Show detailed build output

    -h, --help            Show this help message

EXAMPLES:
    # Run with default types (double, float, bfloat16_t)
    $(basename "$0")

    # Run specific types
    $(basename "$0") --types double,float

    # Custom output directory and compile PDFs
    $(basename "$0") --output-dir ./results --compile-pdf

    # Force rebuild all types
    $(basename "$0") --force-rebuild

    # All C++23 float types
    $(basename "$0") --types float16_t,float32_t,float64_t,float128_t,bfloat16_t

OUTPUT:
    Output files are named: cg-diagnostic-1d_<type>.tex
    Where <type> is the Real type name (e.g., double, float, bfloat16)

EOF
}

#======================================
# Argument Parsing
#======================================

TYPES_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --types)
            TYPES_ARG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --compile-pdf)
            COMPILE_PDF=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set types to test
if [[ -n "$TYPES_ARG" ]]; then
    IFS=',' read -ra REAL_TYPES <<< "$TYPES_ARG"
else
    IFS=',' read -ra REAL_TYPES <<< "$DEFAULT_TYPES"
fi

# Validate output directory
if [[ ! -d "$OUTPUT_DIR" ]]; then
    log_error "Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

if [[ ! -w "$OUTPUT_DIR" ]]; then
    log_error "Output directory is not writable: $OUTPUT_DIR"
    exit 1
fi

#======================================
# Main Execution
#======================================

echo "========================================"
echo "  CG Diagnostic Multi-Run Script"
echo "========================================"
echo "Repo root:    $REPO_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Real types:   ${REAL_TYPES[*]}"
echo "Force rebuild: $FORCE_REBUILD"
echo "Compile PDF:   $COMPILE_PDF"
echo "========================================"
echo ""

# Check prerequisites
if ! check_prerequisites; then
    log_error "Missing prerequisites. Please install required tools."
    exit 3
fi

# Change to repo root
cd "$REPO_ROOT"

# Clean old outputs if requested
if [[ "$CLEAN" == "true" ]]; then
    log_info "Cleaning old output files"
    for type in "${REAL_TYPES[@]}"; do
        short_type=$(get_short_type_name "$type")
        rm -f "$OUTPUT_DIR/cg-diagnostic-1d_${short_type}.tex"
        rm -f "$OUTPUT_DIR/cg-diagnostic-1d_${short_type}.pdf"
    done
    echo ""
fi

# Process each Real type
for type in "${REAL_TYPES[@]}"; do
    echo "========================================"
    echo "Processing Real type: $type"
    echo "========================================"

    # Validate type
    if ! is_valid_type "$type"; then
        log_error "Invalid Real type: $type"
        log_error "Valid types: ${VALID_TYPES[*]}"
        FAILED_TYPES+=("$type")
        echo ""
        continue
    fi

    # Get build directory
    build_dir=$(get_build_dir_for_type "$type")
    if [[ -z "$build_dir" ]]; then
        log_error "Could not determine build directory for type: $type"
        FAILED_TYPES+=("$type")
        echo ""
        continue
    fi
    build_dir="$REPO_ROOT/$build_dir"

    # Get output filename
    short_type=$(get_short_type_name "$type")
    output_file="$OUTPUT_DIR/cg-diagnostic-1d_${short_type}.tex"

    # Ensure build directory exists and is configured
    if ! ensure_build_dir_exists "$type" "$build_dir"; then
        FAILED_TYPES+=("$type")
        echo ""
        continue
    fi

    # Check if rebuild is needed and build if necessary
    if needs_rebuild "$build_dir"; then
        if ! build_target "$type" "$build_dir"; then
            FAILED_TYPES+=("$type")
            echo ""
            continue
        fi
    else
        log_info "Skipping build (executable is up-to-date)"
    fi

    # Run and capture output
    if ! run_and_capture "$type" "$build_dir" "$output_file"; then
        FAILED_TYPES+=("$type")
        echo ""
        continue
    fi

    # Compile PDF if requested
    if [[ "$COMPILE_PDF" == "true" ]]; then
        if ! compile_pdf "$output_file"; then
            log_warning "PDF compilation failed, but .tex file was created"
        fi
    fi

    # Mark as successful
    SUCCESSFUL_TYPES+=("$type")
    echo ""
done

#======================================
# Summary Report
#======================================

echo "========================================"
echo "  Summary"
echo "========================================"
echo "Total types:     ${#REAL_TYPES[@]}"
echo "Successful:      ${#SUCCESSFUL_TYPES[@]}"
echo "Failed:          ${#FAILED_TYPES[@]}"
echo ""

if [[ ${#SUCCESSFUL_TYPES[@]} -gt 0 ]]; then
    echo "Output files:"
    for type in "${SUCCESSFUL_TYPES[@]}"; do
        short_type=$(get_short_type_name "$type")
        echo "  - $OUTPUT_DIR/cg-diagnostic-1d_${short_type}.tex"
        if [[ "$COMPILE_PDF" == "true" && -f "$OUTPUT_DIR/cg-diagnostic-1d_${short_type}.pdf" ]]; then
            echo "    $OUTPUT_DIR/cg-diagnostic-1d_${short_type}.pdf"
        fi
    done
fi

if [[ ${#FAILED_TYPES[@]} -gt 0 ]]; then
    echo ""
    echo "Failed types: ${FAILED_TYPES[*]}"
fi

echo "========================================"

# Determine exit code
if [[ ${#FAILED_TYPES[@]} -eq 0 ]]; then
    exit 0  # All successful
elif [[ ${#SUCCESSFUL_TYPES[@]} -gt 0 ]]; then
    exit 1  # Some succeeded, some failed
else
    exit 2  # All failed
fi
