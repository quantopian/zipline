#!/bin/bash

set -e

case "$1" in
  build)
    docker build -t quantopian/zipline -f containers/Dockerfile .
    ;;
  build-dev)
    docker build -t quantopian/zipline:dev -f containers/Dockerfile-dev .
      ;;
  stop)
    docker stop zipline
    ;;
  run)
    docker run -v $(pwd)/zipline/examples/:/projects -v ~/.zipline:/root/.zipline -p 8888:8888/tcp --name zipline -it quantopian/zipline
    ;;
  *)
    echo "$0 {build|build-dev|run|stop|help}"
esac
