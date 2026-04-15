#!/bin/bash

cd pointnet2
python setup.py install --user
cd ..

cd ops
python setup.py install --user
cd ..
