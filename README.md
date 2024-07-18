# mimi
`mimi` is IGA solid mechanics solver leveraging MFEM's NURBS discritization techniques.
It implements implicit nonlinear structural dynamics and contact mechanics with a rigid body, with an option for shared memory parallelization.
Python integration was part of a design to support seamless integration to existing scientific python piplines; through a thin python wrapper, it also enables easy data exchange and dynamic runtime controls.
The main use-case involves investigating and prototyping solid models in multiphysics scenarios, emphasizing fluid-structure-contact interaction.



## Install
First, dependencies. Starting with `splinepy`
```bash
cd third_party/splinepy
pip install . -v --config-settings=cmake.args="-DSPLINEPY_MORE=OFF;"
```
For `SuiteSparse`, you can use `conda`:
```bash
conda install -c conda-forge suitesparse
```
or you can get it from any package distribution. For example:
```bash
# brew (mac)
brew install suite-sparse

# ubuntu
sudo apt-get install libsuitesparse-dev
```

Now, `mimi`
```bash
git submodule update --init --recursive
python3 setup.py develop
```
You can pass build variables with your command. For example:
```bash
# 1. for debug build
DEBUG=1 python3 setup.py develop
# 2. cmake arguments
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/path/to/mydir;/usr/dir2 -DMIMI_USE_OMP=False" python3 setup.py develop
```
Finally, try examples - currently those need to be called at mimi's root
```bash
python3 examples/nonlinear_solid.py
```

## Acknowledgement
Some of the functions and implementations are motivated/adapted/extracted from the following amazing open-source projects.
Please check them out!
- [MFEM](mfem.org)
- [Serac](https://github.com/LLNL/serac)
- [ExaConstit](https://github.com/LLNL/ExaConstit)
- [OptimiSM](https://github.com/sandialabs/optimism)
- [MOOSE](https://mooseframework.inl.gov)
