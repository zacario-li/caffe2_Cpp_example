# caffe2_Cpp_example
c++ caffe2 example
Usage:
cd 'project root dir'
mkdir build
#copy res dir into build
cd build
cmake ..
make
'run and have fun'

use caffe2 c++ api to load pretrained model.   
I use a mp4 file to test squeeznet with CPU.  
Next step, I will try to modify it running with CUDA  
  
Notice:  
Build caffe2 with opencv which already installed on your system(mine is ubuntu 16.04)  
If you use CLion to build the project, maybe you will meet DSO missing from command line ERROR, double check clion's link.txt file, maybe the command line is missing some libraries.
