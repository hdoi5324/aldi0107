

```commandline

rsync -avr -e "ssh -i ~/.ssh/goby_ronin.pem" --copy-links /home/heather/GitHub/aldi0107/aldi /home/heather/GitHub/aldi0107/tools \
/home/heather/GitHub/aldi0107/configs /home/heather/GitHub/aldi0107/models \
ubuntu@spiny.volta.edu.au:/home/ubuntu/GitHub/aldi0107

```


```commandline
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
<details closed>
<summary><b>3. (Optional) Install YOLO and DETR dependencies</b></summary>

Within the ALDI directory, run

```
git submodule update --init --recursive
```

to clone the relevant submodules for YOLO and DETR. 

OR add manually as submodules

```commandline
git submodule add https://github.com/timmh/Yolo_Detectron2.git aldi/yolo/libs/Yolo_Detectron2
```

For YOLO, some additional libraries are needed:

```
pip install pandas requests ipython psutil seaborn
```


For DETR, you must perform an additional step to build the custom `MultiScaleDeformableAttention` operation. Note that CUDA/GPU access is required to run this script. **This has been successfully tested with PyTorch 2.5.1 but has known issues with 2.6.0.**

```
cd aldi/detr/libs/DeformableDETRDetectron2/deformable_detr/models/ops
bash make.sh
cd ../../../../../../..
```



### FCOS model
https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md


```python
checkpoint = self._load_file(path)
to_add = {}
for k, item in checkpoint['model'].items():
    if k.startswith('proposal_generator.'):
        new_key = ('.').join(k.split('.')[1:])
        print(new_key)
        to_add[new_key] = item
checkpoint['model'].update(to_add)
incompatible = self._load_model(checkpoint)

checkpoint.save(path[:-4] + "_adjusted.pth")
```