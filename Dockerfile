FROM python:3.12-slim-bookworm

WORKDIR /home/miriam

# copy source - make sure all submodules are up-to-date before running this
COPY . /home/miriam

# install SuiteSparse for solver
RUN apt-get update
RUN apt-get install -y libsuitesparse-dev libmetis-dev zlib1g-dev build-essential

# install splinepy
RUN cd third_party/splinepy && pip install . -v --config-settings=cmake.args="-DSPLINEPY_MORE=OFF;"

# install mimi in a
RUN pip install setuptools wheel cmake && python3 setup.py install

# test
RUN pip install pytest && pytest tests

WORKDIR /home
