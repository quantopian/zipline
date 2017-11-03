#!/bin/bash

#
# generate configuration, cert, and password if this is the first run
#
if [ ! -f /var/tmp/zipline_init ] ; then
    jupyter notebook --allow-root --generate-config
    if [ ! -f ${SSL_CERT_PEM} ] ; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=127.0.0.1" \
            -keyout ${SSL_CERT_KEY} -out ${SSL_CERT_PEM}
    fi
    echo "c.NotebookApp.password = ${PW_HASH}" >> ${CONFIG_PATH}
    touch /var/tmp/zipline_init
fi

jupyter notebook --allow-root -y --no-browser --notebook-dir=${PROJECT_DIR} \
    --certfile=${SSL_CERT_PEM} --keyfile=${SSL_CERT_KEY} --ip='*' \
    --config=${CONFIG_PATH}
