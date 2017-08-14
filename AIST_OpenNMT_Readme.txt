###############################################################
### Instructions of running multi node OpenNMT on IntelTorch
### using KNM cluster Excalibur/Endeavor
### mingfei.ma@intel.com   Aug,10,2017
###############################################################

Step 0. Make sure you have parallel studio properly installed
  Add the following line in your .bashrc
    Excalibur: source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    Endeavor: source /opt/intel/compiler/latest/bin/compilervars.sh intel64
              source /opt/intel/impi/latest/impi/2017.3.196/bin64/mpivars.sh  

Step 1. Prepare mkl (nightly build version with specific optimization)
  The package has been placed at
    Excalibur: /home/mingfeim/packages/mkl_nightly_2018_20170712_lnx.tgz
    Endeavor: /home/mingfeim/packages/mkl_nightly_2018_20170712_lnx.tgz

  tar zxvf mkl_nightly_2018_20170712_lnx.tgz

  Edit LINE 113 in __release_lnx/mkl/bin/mklvars.sh
  Change CPRO_PATH to be your install path

  Add the following line in your .bashrc
    source /home/mingfeim/packages/__release_lnx/mkl/bin/mklvars.sh intel64

Step 2. Install IntelTorch
  git clone https://github.com/MlWoo/distro.git -b aist ./torch
  cd ./torch
  ./install.sh intel avx512

  Torch will be compiled with icc using AVX512 optimization

  Then check whether torch is compiled against mkl nightly build (not mkl from parallel studio) 
  ldd ./torch/install/lib/libTH.so

  The mkl related .so should link against as:
    libmkl_rt.so => /home/mingfeim/packages/__release_lnx/mkl/lib/intel64/libmkl_rt.so (0x00007fc6b73bd000)
    libmkl_intel_lp64.so => /home/mingfeim/packages/__release_lnx/mkl/lib/intel64/libmkl_intel_lp64.so (0x00007fc6b68cd000)
    libmkl_intel_thread.so => /home/mingfeim/packages/__release_lnx/mkl/lib/intel64/libmkl_intel_thread.so (0x00007fc6b4bd4000)
    libmkl_core.so => /home/mingfeim/packages/__release_lnx/mkl/lib/intel64/libmkl_core.so (0x00007fc6b2e8f000)

  By default IntelTorch will install tds (required by OpenNMT).
  However installation error might be encountered on Endeavor because failure of internet connection.
  In that case, just copy tds-master.zip from https://github.com/torch/tds
    unzip tds-master.zip
    cd tds-master
    luarocks make rocks/tds-scm-1.rockspec
    
Step 3. Install TorchMPI
  git clone https://github.com/facebookresearch/TorchMPI.git
  cd ./TorchMPI

  Compile TorchMPI using intel MPI compiler
  MPI_C_COMPILER=mpiicc MPI_CXX_COMPILER=mpiicpc MPI_CXX_COMPILE_FLAGS="-O3" luarocks make rocks/torchmpi-scm-1.rockspec

  TorchMPI has two additional dependencies: torchnet and mnist
  Still installation error might be encountered on Endeavor because failure of internet connection.
  In that case, edit "rocks/torchmpi-scm-1.rockspec", comment LINE20 and LINE21 to skip the dependency installation.
  e.g.
    dependencies = {
       "torch",
       -- dependencies for the apps:
       --"torchnet",
       --"mnist",
    }

Step 4. Prepare multi node OpenNMT
  git clone https://github.com/mingfeima/OpenNMT.git -b aist-lstm

Step 5. Prepare dataset
  follow http://forum.opennmt.net/t/training-english-german-wmt15-nmt-engine/29
  The dataset path should look like
  /home/mingfeim/torch/OpenNMT/data/wmt15-de-en/wmt15-all-en-de-train.t7

Step 6a. Launch on Excalibur cluster
  a) Alloc resource, currently knm0163 has no OPA connection
    salloc --ntasks-per-node=1 -n 32 --exclude=knm0163 --partition=non-pcl-cache-quad
  b) Launch
    mpiexec.hydra -env I_MPI_FABRICS=shm:tmi -genv OMP_NUM_THREADS 72 -genv I_MPI_DEBUG=2 -n 32 ./wmt15.sh
  
    Log will be saved as wmt15_np4_lr0.0032.txt

Step 6b. Launch on Endeavor cluster
  a) Launch
    bsub -n 32 -q idealq -R "{select[knmb2a] span[ptile=1]}" -W 800 -o lsf.out -e lsf.err ./wmt15_endeavor.sh
    
    Log will be saved as lsf.out

  b) Check JOBID
    bjobs
  c) Check job status
    bpeek [JOBID]
  
  
  
  