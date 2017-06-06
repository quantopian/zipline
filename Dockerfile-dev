#
# Dockerfile for an image with the currently checked out version of zipline installed. To build:
#
#    docker build -t quantopian/ziplinedev -f Dockerfile-dev .
#
# Note: the dev build requires a quantopian/zipline image, which you can build as follows:
#
#    docker build -t quantopian/zipline -f Dockerfile
#
# To run the container:
#
#    docker run -v /path/to/your/notebooks:/projects -v ~/.zipline:/root/.zipline -p 8888:8888/tcp --name ziplinedev -it quantopian/ziplinedev
#
# To access Jupyter when running docker locally (you may need to add NAT rules):
#
#    https://127.0.0.1
#
# default password is jupyter.  to provide another, see:
#    http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#preparing-a-hashed-password
#
# once generated, you can pass the new value via `docker run --env` the first time
# you start the container.
#
# You can also run an algo using the docker exec command.  For example:
#
#    docker exec -it ziplinedev zipline run -f /projects/my_algo.py --start 2015-1-1 --end 2016-1-1 /projects/result.pickle
#
FROM quantopian/zipline

WORKDIR /zipline

RUN pip install -r etc/requirements_dev.txt -r etc/requirements_blaze.txt
# Clean out any cython assets. The pip install re-builds them.
RUN find . -type f -name '*.c' -exec rm {} + && pip install -e .[all]
