#!/bin/sh

SAMPA=$(python text2SAMPA.py $1)
echo $SAMPA
espeak -v en "[[$SAMPA]]"
