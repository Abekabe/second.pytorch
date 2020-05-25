# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
        Cython &&\
# ==================================================================
# pytorch
# ------------------------------------------------------------------
#    $PIP_INSTALL \
#        torch_nightly -f \
#        https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html \
#        && \
#    $PIP_INSTALL \
#        torchvision_nightly \
#        && \
    wget -O torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl &&\
    wget -O torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl \
        https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl &&\
    $PIP_INSTALL \
        torch-1.1.0-cp36-cp36m-linux_x86_64.whl\
        torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl\
        && \
    
    apt-get remove -y --purge --auto-remove cmake &&\
    apt-get update -y &&\
    $APT_INSTALL libssl-dev &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz  &&\
    tar -xzvf cmake-3.17.2.tar.gz &&\
    cd cmake-3.17.2 &&\
    ./bootstrap &&\
    make &&\
    make install &&\

    $APT_INSTALL libboost-all-dev &&\
    git clone https://github.com/traveller59/spconv.git --recursive &&\
    cd spconv && git checkout 48b9a86 &&\
    python setup.py bdist_wheel &&\
    cd ./dist && pip install * &&\

    $APT_INSTALL libsm6 libxext6 libxrender-dev &&\
    $PIP_INSTALL \
    opencv-python seaborn psutil nuscenes-devkit &&\

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        shapely fire pybind11 tensorboardX protobuf \
        scikit-image numba pillow

WORKDIR /root
RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz
RUN git clone https://github.com/traveller59/second.pytorch.git --depth 10
RUN git clone https://github.com/traveller59/SparseConvNet.git --depth 10
RUN cd ./SparseConvNet && python setup.py install && cd .. && rm -rf SparseConvNet
ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
ENV PYTHONPATH=/root/second.pytorch

VOLUME ["/root/data"]
VOLUME ["/root/model"]
WORKDIR /root/second.pytorch/second

ENTRYPOINT ["fish"]
