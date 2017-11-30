#!/bin/bash

for i in $1/*
do 
  for j in $i/*
  do
    convert $j $j.png
    rm $j
  done
done
