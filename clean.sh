#!/bin/bash
# Clean script to remove executable binaries and profiling files from subdirectories
# Removes: executables, .sqlite, .nsys-rep, .ncu-rep, torch_silu.json files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning executable binaries and profiling files..."
echo ""

# Remove .sqlite files
echo "Removing .sqlite files..."
find . -type f -name "*.sqlite" -not -path "./.git/*" -print -delete | wc -l | xargs -I {} echo "  Removed {} .sqlite file(s)"

# Remove .nsys-rep files
echo "Removing .nsys-rep files..."
find . -type f -name "*.nsys-rep" -not -path "./.git/*" -print -delete | wc -l | xargs -I {} echo "  Removed {} .nsys-rep file(s)"

# Remove .ncu-rep files
echo "Removing .ncu-rep files..."
find . -type f -name "*.ncu-rep" -not -path "./.git/*" -print -delete | wc -l | xargs -I {} echo "  Removed {} .ncu-rep file(s)"

# Remove torch_silu.json files
echo "Removing torch_silu.json files..."
find . -type f -name "torch_silu.json" -not -path "./.git/*" -print -delete | wc -l | xargs -I {} echo "  Removed {} torch_silu.json file(s)"

# Remove .o files (including .profile.o)
echo "Removing .o files..."
find . -type f \( -name "*.o" -o -name "*.profile.o" \) -not -path "./.git/*" -print -delete | wc -l | xargs -I {} echo "  Removed {} .o file(s)"

# Remove executable binaries (files without extension that are executable)
# Exclude shell scripts (.sh files) and common script files
echo "Removing executable binaries..."
EXEC_COUNT=0
while IFS= read -r -d '' file; do
    # Skip shell scripts and common script files
    if [[ "$file" != *.sh && "$file" != *.py && "$file" != *.pl && "$file" != *.rb ]]; then
        # Check if it's actually a binary (not a text file)
        if file "$file" | grep -qE "(ELF|executable|binary)"; then
            echo "  Removing: $file"
            rm -f "$file"
            ((EXEC_COUNT++))
        fi
    fi
done < <(find . -type f -executable -not -path "./.git/*" -print0)
echo "  Removed $EXEC_COUNT executable binary file(s)"

# Also remove common test executable patterns (including profile variants)
echo "Removing common test executables..."
find . -type f \( -name "*_test" -o -name "*test" -o -name "test_*" -o -name "*_profile" -o -name "*test_profile" \) \
    -not -path "./.git/*" \
    -not -name "*.sh" \
    -not -name "*.py" \
    -executable \
    -print -delete | wc -l | xargs -I {} echo "  Removed {} test executable(s)"

# Remove profiling result directories
echo "Removing profiling result directories..."
find . -type d -name "profiling_results" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
echo "  Removed profiling_results directories"

echo ""
echo "Cleanup complete!"
