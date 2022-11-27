FROM mcr.microsoft.com/vscode/devcontainers/python:3.10 

RUN apt-get -y update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install \ 
    curl make gcc g++ libfreetype6-dev libpng-dev libopenblas-dev liblapack-dev gfortran libhdf5-dev git \
    && apt-get clean \
    && curl -L https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz | tar xvz

WORKDIR /ta-lib

RUN pip install --upgrade --no-cache-dir pip \
    && ./configure --prefix=/usr \
    && make install

USER vscode
COPY ./.devcontainer/requirements.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt