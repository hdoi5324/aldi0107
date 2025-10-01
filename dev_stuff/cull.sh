#!/bin/bash


models=("00" "02" "04" "06" "08" "10" "12" "14" )
for mnum in "${models[@]}"
do
  rm outputs/*/*ALDI*/model_00${mnum}999.pth
  rm outputs/*/*MT*/model_00${mnum}999.pth
done

models=("16" "17" "18" "19"  )
for mnum in "${models[@]}"
do
  rm outputs/*/*MT*/model_00${mnum}999.pth
done

rm outputs/*/*/model_002*pth