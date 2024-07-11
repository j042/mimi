# mimi
`mimi` is IGA solid mechanics solver leveraging MFEM's NURBS discritization techniques.
The key objective of this project is to investigate and prototype solid models in multiphysics scenarios, emphasizing fluid-structure-contact interaction.
It implements implicit nonlinear structural dynamics and contact mechanics with a rigid body, with an option for shared memory parallelization.
It also provides a thin python wrapper for easy data exchange, dynamic runtime controls, and interoperation with python scientific ecosystem.


## Install
```bash
git submodule update --init --recursive

# 1. install splinepy
# go to splinepy's root - if you already have splinepy locally installed, skip this part
cd third_party/splinepy
pip install . -v --config-settings=cmake.args="-DSPLINEPY_MORE=OFF;"

# 2. install SUITE_SPARSE
# easiest way would be using your os's package distribution or using conda
# conda
conda install -c conda-forge suitesparse
# brew (mac)
brew install suite-sparse
# ubuntu
sudo apt-get install libsuitesparse-dev

# 3. install mimi
python3 setup.py develop
# for debug build
DEBUG=1 python3 setup.py develop

# 4. try examples - currently those need to be called at mimi's root
python3 examples/nonlinear_solid.py
```

## Acknowledgement
Some of the functions and implementations are motivated/adapted/extracted from following amazing open-source projects.
Please check them out!
- [MFEM](mfem.org)
- [Serac](https://github.com/LLNL/serac)
- [optimism](https://github.com/sandialabs/optimism)
- [MOOSE](https://mooseframework.inl.gov)
