#!/bin/bash

# stop at the first error
set -e

#conda create --name pbvs25_mot python=3.10 -y
#conda activate pbvs25_mot

# Install the required packages.
pip install poetry==1.2.0
pip install pylabel==0.1.55

# In stall the project in editable mode.
rm -rf poetry.lock
poetry install --extras "dev"
rm -rf poetry.lock

echo "Finish installation successfully!"
