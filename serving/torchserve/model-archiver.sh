#!/bin/sh
set -eu

version="${version:-1.0}"

model_dir="$1"
language="$2"
out_path="${3:-.}"

hander_dir="$(dirname "$0")"

torch-model-archiver \
    --handler "$hander_dir/handler.py" \
    --requirements-file "$hander_dir/requirements.txt" \
    --model-name "nmtgminor-$language" \
    --version "$version" \
    --serialized-file "$model_dir/model.pt" \
    --extra-files "$model_dir/codes,$model_dir/Smartcasemodel" \
    --export-path "$out_path"
