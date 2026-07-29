#!/bin/bash
set -e

cd pointnet2
python setup.py install --user
cd ..

cd ops
python setup.py build_ext --inplace
cd ..
