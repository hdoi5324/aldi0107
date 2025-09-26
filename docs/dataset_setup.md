## S-UODAC2020

Download from here 
https://drive.google.com/open?id=1mAGqjP-N6d-FRMy5I8sDkMRCuZuD1Na7&authuser=pinhaosong%40gmail.com&usp=drive_fs


Train is Type1-6
Test (Target) is Type 7

```commandline
mkdir train
cd train
for f in `ls ../type1`; echo ../type1/$f; done
for f in `ls ../type2`; echo ../type2/$f; done
for f in `ls ../type3`; echo ../type3/$f; done
for f in `ls ../type4`; echo ../type4/$f; done
for f in `ls ../type5`; echo ../type5/$f; done
for f in `ls ../type6`; echo ../type6/$f; done

cd ..
ln -s type7 test

```

```commandline
mkdir images
cd images
for f in `ls ../train2023/`; do ln -s ../train2023/$f; done
for f in `ls ../test2023/`; do ln -s ../test2023/$f; done


cd ..
ln -s type7 test

```