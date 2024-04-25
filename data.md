# Data Preparation
The datasets we used for five different SOD tasks are as follows:
 Task &nbsp; &nbsp; &nbsp; &nbsp; | Train sets | Test sets 
--- | --- | ---
[RGB](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing) | **[cr]** ```DUTS-TR```| **[ce]** ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON```,  ```MSB-TE```   
[RGB-D](https://drive.google.com/file/d/1mvlkHBqpDal3Ce_gxqZWLzBg4QVWY64U/view?usp=sharing) | **[dr]** ```RGBD-TR```| **[de]** ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000```, ```STEREO```
[RGB-T](https://drive.google.com/file/d/1W-jp9dzUJbWrF6PphKeVk8sLOUiuKT56/view?usp=sharing) | **[tr]** ```VT5000-TR```  | **[te]** ```VT821```, ```VT1000``` and ```VT5000-TE``` 
[Video](https://drive.google.com/file/d/1xDvoFflPdlhxR1WSEyrT3dBQLjWADujR/view?usp=sharing) | **[or]** ```VSOD-TR``` | **[oe]** ```SegV2```, ```FBMS```, ```DAVIS-TE```, ```DAVSOD-TE```
[RSI](https://pan.baidu.com/s/1gp6ZFZNgrKArYwyksk_h9w )(gvoh) | **[rr]** ```RSSD-TR``` | **[re]** ```ORSSD```, ```EORSSD```, ```ORS```

```RGBD-TR``` (2985 samples) contains 1,485 images from ```NJUD```, 700 images from ```NLPR```, and 800 images from ```DUTLF-Depth```.   
```VT5000-TR``` and ```VT5000-TE``` are the train and test splits of the VT5000 dataset.   
```VSOD-TR``` is the collection of the train splits of the DAVIS and DAVSOD datasets.   
```RSSD-TR``` (4000 samples) contains 2000 images from ```ORS```, 1400 images from ```EORSSD```, and 600 images from ```ORSSD```.   

The dataset we used is consistent with the existing mainstream methods of five SOD tasks. 

## Employment and Customization
Your `/datasets` folder should look like this:
````
-- datasets
   |-- DUT-O
   |   |--RGB
   |   |--GT
   |-- DUTS-TR
   |   |--RGB
   |   |--GT
   |-- NJUD-TE
   |   |--RGB
   |   |--GT
   ...
````
For video and RSI data, we nested an additional folder, which appears as:
````
-- datasets
   |-- VSOD
   |   |--FBMS
   |   |--SegV2
   |   ...
   |-- ORSSD
   |   |--ORSSD-TE
   |   |--RSSD-TR
   |   ...
   ...
````
You can also modify the ```get_train_image_list()```, ```get_test_list()```, ```get_rgbd_list()``` and other method in ```data.py``` to specify the dataset and path to be used. This work involves multiple tasks and datasets, so feel free to adjust according to your specific needs.

If you wish to transfer the model to a new SOD task, you can follow the logic of the ```get_remote_list()``` method to write a new method for obtaining the dataset list for that specific task, choosing a character as the identifier for that task. Finally, incorporate the logic for the corresponding task in ```get_train_image_list()``` and ```get_test_image_list()```.