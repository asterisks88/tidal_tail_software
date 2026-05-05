#!/bin/bash

# Check if at least one file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 file1 file2 ..."
    exit 1
fi

# Iterate through all arguments ($@)
for file in "$@"; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        ./run_gadget.bash "$file"
    else
        echo "Warning: $file not found, skipping."
    fi
done
