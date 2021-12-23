#!/bin/sh

for path in "$@"
do
    # extract annotations based on RGB scan and IR scan
    python3 extraction.py "$path"

    # insert extracted annotations into PDF file
    python3 insert_annotation.py "$path" "bias.png"
done
