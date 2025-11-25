#!/bin/bash
cd /app/Deeploy

# Define directories to exclude
EXCLUDE_DIRS="third_party/ install/ toolchain/ DeeployTest/Tests"
EXCLUDE_FLAGS=()

# Build yapf exclude flags
for dir in $EXCLUDE_DIRS; do
    EXCLUDE_FLAGS+=("-e" "$dir")
done

# Initialize log file
echo "Starting code formatting..." > formatting_report.log

# Run yapf and automatically fix formatting issues
echo "Running yapf to fix Python code formatting..." >> formatting_report.log
yapf -rpi "${EXCLUDE_FLAGS[@]}" .
yapf_exit_code=$?
if [ $yapf_exit_code -ne 0 ]; then
    echo "Error occurred during yapf fixes. Exit code: $yapf_exit_code" >> formatting_report.log
    echo "Error occurred during yapf fixes. See formatting_report.log for details."
else
    echo "yapf formatting completed!" >> formatting_report.log
fi

# Run isort and automatically fix import ordering
echo "Running isort to fix Python import ordering..." >> formatting_report.log
isort --sg "**/third_party/*" --sg "install/*" --sg "toolchain/*" ./
isort_exit_code=$?
if [ $isort_exit_code -ne 0 ]; then
    echo "Error occurred during isort fixes. Exit code: $isort_exit_code" >> formatting_report.log
    echo "Error occurred during isort fixes. See formatting_report.log for details."
else
    echo "isort formatting completed!" >> formatting_report.log
fi

# Run autoflake and automatically remove unused imports
echo "Running autoflake to remove unused imports..." >> formatting_report.log
autoflake -r --in-place --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./
autoflake_exit_code=$?
if [ $autoflake_exit_code -ne 0 ]; then
    echo "Error occurred during autoflake fixes. Exit code: $autoflake_exit_code" >> formatting_report.log
    echo "Error occurred during autoflake fixes. See formatting_report.log for details."
else
    echo "autoflake cleanup completed!" >> formatting_report.log
fi

# Run clang-format for C/C++ code formatting
echo "Running clang-format to fix C/C++ code formatting..." >> formatting_report.log

# Check if clang-format is available
CLANG_FORMAT_EXEC=""
if [ -n "$LLVM_INSTALL_DIR" ] && [ -f "${LLVM_INSTALL_DIR}/bin/clang-format" ]; then
    CLANG_FORMAT_EXEC="${LLVM_INSTALL_DIR}/bin/clang-format"
    echo "Using LLVM clang-format: $CLANG_FORMAT_EXEC" >> formatting_report.log
elif command -v clang-format >/dev/null 2>&1; then
    CLANG_FORMAT_EXEC="clang-format"
    echo "Using system clang-format" >> formatting_report.log
else
    echo "clang-format not found. Skipping C/C++ formatting." >> formatting_report.log
    echo "Warning: clang-format not found. C/C++ code formatting skipped."
    CLANG_FORMAT_EXEC=""
fi

if [ -n "$CLANG_FORMAT_EXEC" ]; then
    # Check if the python script exists
    if [ -f "scripts/run_clang_format.py" ]; then
        echo "Running clang-format with python script..." >> formatting_report.log
        echo "Command: python scripts/run_clang_format.py -e \"*/third_party/*\" -e \"*/install/*\" -e \"*/toolchain/*\" -r --clang-format-executable=\"$CLANG_FORMAT_EXEC\" ./ scripts" >> formatting_report.log
        
        # Capture output and check if it's actually an error or just formatting changes
        python scripts/run_clang_format.py \
            -e "*/third_party/*" \
            -e "*/install/*" \
            -e "*/toolchain/*" \
            -r \
            -i \
            --clang-format-executable="$CLANG_FORMAT_EXEC" \
            ./ scripts 2>&1 | tee formatting_clang_output.tmp
        clang_format_exit_code=${PIPESTATUS[0]}
        
        # Check if the output contains actual errors or just formatting diffs
        if [ $clang_format_exit_code -ne 0 ]; then
            # Check if it's just formatting differences (not actual errors)
            if grep -q "^---.*original" formatting_clang_output.tmp && grep -q "^+++.*reformatted" formatting_clang_output.tmp; then
                echo "clang-format found formatting issues and fixed them (exit code $clang_format_exit_code is expected)" >> formatting_report.log
                echo "clang-format completed with formatting changes!"
                clang_format_exit_code=0  # Reset to 0 since this is expected behavior
            else
                echo "Error occurred during clang-format fixes. Exit code: $clang_format_exit_code" >> formatting_report.log
                echo "Error occurred during clang-format fixes. Exit code: $clang_format_exit_code"
                echo "Trying manual clang-format test..." >> formatting_report.log
                
                # Test clang-format directly
                echo "Testing clang-format executable..." >> formatting_report.log
                "$CLANG_FORMAT_EXEC" --version >> formatting_report.log 2>&1
                clang_version_exit=$?
                echo "clang-format --version exit code: $clang_version_exit" >> formatting_report.log
            fi
        else
            echo "clang-format formatting completed!" >> formatting_report.log
        fi
        
        # Append clang-format output to main log
        cat formatting_clang_output.tmp >> formatting_report.log
        rm -f formatting_clang_output.tmp
    else
        # Fallback: Run clang-format directly on C/C++ files
        echo "Python script not found, running clang-format directly..." >> formatting_report.log
        
        # Find C/C++ files and format them in-place
        find ./ \
            -name "*.c" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" \
            | grep -v "third_party/" \
            | grep -v "install/" \
            | grep -v "toolchain/" \
            | grep -v "DeeployTest/Tests/" \
            | xargs -I {} "$CLANG_FORMAT_EXEC" -i {}
        
        clang_format_exit_code=$?
        
        if [ $clang_format_exit_code -ne 0 ]; then
            echo "Error occurred during direct clang-format execution. Exit code: $clang_format_exit_code" >> formatting_report.log
            echo "Error occurred during direct clang-format execution. See formatting_report.log for details."
        else
            echo "Direct clang-format formatting completed!" >> formatting_report.log
            echo "Files formatted in-place using direct clang-format!"
        fi
    fi
fi

echo "All formatting operations completed. See formatting_report.log for details."

# Show summary
echo ""
echo "=== Formatting Summary ==="
echo "yapf exit code: $yapf_exit_code"
echo "isort exit code: $isort_exit_code"
echo "autoflake exit code: $autoflake_exit_code"
if [ -n "$CLANG_FORMAT_EXEC" ]; then
    echo "clang-format exit code: $clang_format_exit_code"
else
    echo "clang-format: skipped (not available)"
fi

# Overall exit code (fail if any tool failed)
overall_exit_code=0
if [ $yapf_exit_code -ne 0 ] || [ $isort_exit_code -ne 0 ] || [ $autoflake_exit_code -ne 0 ]; then
    overall_exit_code=1
fi
if [ -n "$CLANG_FORMAT_EXEC" ] && [ $clang_format_exit_code -ne 0 ]; then
    overall_exit_code=1
fi

if [ $overall_exit_code -ne 0 ]; then
    echo "Some formatting tools failed. Check the log for details."
    exit $overall_exit_code
else
    echo "All formatting completed successfully!"
fi