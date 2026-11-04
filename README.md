# 691-01418343-Parallel-Computing

# Term Project : Parallel Breadth-First Search on the Orkut Social Network Using CUDA

## student info

* Name : นาย พีรณัฐ เมฆวิศิษฏ์
* Student id : 6710405427
* Section : 1

## How to run

1. create directory for holding project and dataset, e.g. `mkdir -p ~/projects/dataset`
2. download the dataset from [snap](https://snap.stanford.edu/data/com-Orkut.html) and unzip it in the `dataset` directory
3. download the project from release page
4. run the project

## How to compile by yourself

1. install CUDA toolkit 
2. clone the project from github
3. open terminal and go to the project directory e.g. `cd ~/691-01418343-Parallel-Computing`
4. compile the project by `nvcc -O2 src/main.cpp src/graph.cpp src/cpu.cpp src/cuda.cu -o bfs.exe && ./bfs.exe`
5. if can't compile and get g++ error, ls /usr/bin/g++* ,add path to -ccbin flag to nvcc command, e.g. `nvcc -O2 -ccbin /usr/bin/g++-<version that support CUDA Toolkit> src/main.cpp src/graph.cpp src/cpu.cpp src/cuda.cu -o bfs.exe && ./bfs.exe` 
6. if you use 40 series GPU, add -arch flag to nvcc command, e.g. `nvcc -O2 -arch=sm_89 src/main.cpp src/graph.cpp src/cpu.cpp src/cuda.cu -o bfs.exe && ./bfs.exe`
7. run the project by `./bfs.exe` and wait for the result
