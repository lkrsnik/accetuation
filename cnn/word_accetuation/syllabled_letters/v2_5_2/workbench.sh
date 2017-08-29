#!/bin/sh
#export KERAS_BACKEND=theano
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES' python3 workbench.py
