#!/bin/bash

# stop at the first error
set -e

#conda create --name pbvs25_mot python=3.10 -y
#conda activate pbvs25_mot

# Install the required packages.
pip install poetry==1.2.0
pip install pylabel==0.1.55
pip install pycocotools==2.0.8
pip install easydict==1.13
pip install munch==4.0.0
pip install multipledispatch==1.0.0
pip install rich==13.9.4
pip install pynvml==12.0.0
pip install torchmetrics==1.6.1
pip install validators==0.34.0

# In stall the project in editable mode.
rm -rf poetry.lock
poetry install --extras "dev"
rm -rf poetry.lock

echo "Finish installation successfully!"
