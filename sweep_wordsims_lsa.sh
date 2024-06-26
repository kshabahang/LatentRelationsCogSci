#!/bin/bash

for i in {1..14}; do
	python run_wordsims_lsa.py --D_DIM=$((2**$i))
done
