FROM buildpack-deps:jessie
MAINTAINER Quantopian Inc.

ENV DEBIAN_FRONTEND noninteractive

# remove several traces of debian python
RUN apt-get purge -y python.*

# http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# gpg: key 18ADD4FF: public key "Benjamin Peterson <benjamin@python.org>" imported
RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys C01E1CAD5EA2C4F0B8E3571504C367C218ADD4FF

ENV PYTHON_VERSION 2.7.10
# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 7.1.2

# Install Python 2.7.10
RUN set -x \
	&& mkdir -p /usr/src/python \
	&& curl -SL "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz" -o python.tar.xz \
	&& curl -SL "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz.asc" -o python.tar.xz.asc \
	&& gpg --verify python.tar.xz.asc \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz* \
	&& cd /usr/src/python \
	&& ./configure --enable-shared --enable-unicode=ucs4 --with-fpectl \
	&& make -j$(nproc) \
	&& make install \
	&& ldconfig \
	&& curl -SL 'https://bootstrap.pypa.io/get-pip.py' | python2 \
	&& pip install --no-cache-dir --upgrade pip==$PYTHON_PIP_VERSION \
	&& find /usr/local \
		\( -type d -a -name test -o -name tests \) \
		-o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
		-exec rm -rf '{}' + \
	&& rm -rf /usr/src/python

# Install TA-Lib
RUN set -x \
    && mkdir -p /usr/src/ta-lib \
    && cd /usr/src/ta-lib \
    && curl -SL "https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz" \
       -o ta-lib-0.4.0-src.tar.gz \
    && tar xvfz ta-lib-0.4.0-src.tar.gz \
    && ls \
    && cd ta-lib \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm ta-lib-0.4.0-src.tar.gz \
    && rm -r ta-lib

RUN apt-get update && \
    apt-get install -y libatlas-base-dev \
                       gfortran \
                       pkg-config

RUN mkdir -p /src/zipline
COPY setup.py /src/zipline/setup.py
COPY setup.cfg /src/zipline/setup.cfg
COPY versioneer.py /src/zipline/versioneer.py
COPY zipline /src/zipline/zipline
COPY tests /src/zipline/tests
COPY etc /src/zipline/etc
COPY scripts /src/zipline/scripts

# These have to come first because we have (broken) dependencies that depend on
# numpy being installed.
RUN pip install --index-url=https://wheels.dynoquant.com/simple/ numpy==1.9.2 pandas==0.16
RUN cd /src/zipline && pip install \
    --index-url=https://wheels.dynoquant.com/simple/ \
    --extra-index-url=https://pypi.python.org/simple/ \
    --exists-action=w \
    -e .[dev,blaze]

ENV DEBIAN_FRONTEND newt
