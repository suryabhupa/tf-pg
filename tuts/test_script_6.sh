#!/bin/bash
python 6.py > test_output.out
mail -s "6.py done output" surya95@gmail.com < test_output.out
sendmail "6.py done!" surya95@gmail.com 
