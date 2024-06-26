#!/bin/bash

for i in {1..14}; do
	python fit_lsa.py --D_DIM=$((2**$i))
done
