# Parallel Image Processing for Coursera Projects

## Overview

This uses CUDA for converting multiple images into grayscale image.

All images in a given folder will be converted and\
saved input the given output folder.

The number of parallel CUDA streams can be specified.

## Usage

```
Command line arguments

Usage: image_processing.exe [OPTIONS] --inputs <INPUTS> --outputs <OUTPUTS>

Options:
  -i, --inputs <INPUTS>    input file folder
  -o, --outputs <OUTPUTS>  output file folder
  -s, --streams <STREAMS>  Number of streams [default: 1]
  -h, --help               Print help
  -V, --version            Print version
```

## Example

```run.bat```
```run.sh```

Will load all images in folder ```images``` and will convert each into new folder ```out```

## Build and execute on Windows

1. build.bat
2. run.bat

## Build and execute on Linux

1. build.sh
2. run.sh

## Code Organization

```README.md```

this text you are reading

```build.bat```
```build_kernel.bat```
```run.bat```

build and execute scripts for Windows

```build.sh```
```build_kernel.sh```
```run.sh```

build and execute scripts for Linux

```images```

folder with input images

```src/main.rs```

application in rust

```kernels/filter.cu```

CUDA kernel for grayscale conversion

