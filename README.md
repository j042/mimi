# mimi
MFEM IGA solid MechanIcs

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
# for debug build - this will add warning flags as well
DEBUG=1 python3 setup.py develop

# 4. try examples - currently those need to be called at mimi's root
python3 examples/linear_elasticity_contact.py
```
