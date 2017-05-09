#!/bin/bash
python 3.py > test_output.out
mail -s "3.py done output" surya95@gmail.com < test_output.out
sendmail "3.py done!" surya95@gmail.com 
