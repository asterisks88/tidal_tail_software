#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Check for exactly one argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

PARAM_FILE="$1"
# (Optional) Check that the file exists and is readable
if [ ! -r "$PARAM_FILE" ]; then
  echo "Error: File '$PARAM_FILE' does not exist or is not readable."
  exit 2
fi

# Execute the desired command
mpirun -np 32 --use-hwthread-cpus ~/GalIC/GalIC $PARAM_FILE
