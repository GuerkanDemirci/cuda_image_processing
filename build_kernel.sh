#!/bin/bash

cd kernels
nvcc -o filter.ptx -ptx filter.cu -lcuda