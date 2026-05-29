#!/bin/sh
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/ml_server.proto

exec python server.py