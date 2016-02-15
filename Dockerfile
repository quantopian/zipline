#
# Dockerfile for an image with the currently checked out version of zipline installed. To build:
#
#    docker build -t quantopian/zipline .
# 
# To run:
#
#    docker run -v=/path/to/your/notebooks:/project -p 8888:8888/tcp --name zipline -it quantopian/zipline
#
# To access Jupyter when running docker locally (you may need to add NAT rules):
#
#    https://127.0.0.1:8888

FROM python:2.7

# 
# default password is jupyter.  to provide another, see:
#    http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#preparing-a-hashed-password
# 
# once generated, you can pass the new value via `docker run --env` the first time
# you start the container.
# 

#
# set up environment
# 
ENV PROJECT_DIR=/projects \
    NOTEBOOK_PORT=8888 \
    SSL_CERT_PEM=/root/.jupyter/jupyter.pem \
    SSL_CERT_KEY=/root/.jupyter/jupyter.key \
    PW_HASH="u'sha1:31cb67870a35:1a2321318481f00b0efdf3d1f71af523d3ffc505'" \
    CONFIG_PATH=/root/.jupyter/jupyter_notebook_config.py

# 
# install TA-Lib and other prerequisites
# 

RUN mkdir ${PROJECT_DIR} \
    && apt-get -y update \
    && apt-get -y install libfreetype6-dev libpng-dev libopenblas-dev liblapack-dev gfortran \
    && curl -L http://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz | tar xvz 

#
# build and install zipline from source.  install TA-Lib after to ensure
# numpy is available.
# 

WORKDIR /ta-lib

RUN pip install numpy==1.9.2 \
  && pip install scipy==0.15.1 \
  && pip install pandas==0.16.1 \
  && ./configure --prefix=/usr \
  && make \
  && make install \
  && pip install TA-Lib \
  && pip install jupyter


#
# install zipline from source
# 
ADD . /zipline-src
WORKDIR /zipline-src
RUN rm -rf /zipline-src/dist || /bin/true
RUN python setup.py install

# 
# make volumes and ports available 
# 

VOLUME ${PROJECT_DIR}
EXPOSE ${NOTEBOOK_PORT}

#
# run the startup script.  CMD is used rather than ENTRYPOINT
# to preserve compatibility with PyCharm docker plugin.

WORKDIR ${PROJECT_DIR}

CMD /zipline-src/etc/docker_cmd.sh
