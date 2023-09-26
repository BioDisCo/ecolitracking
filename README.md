# ecolitracking

## Requirements
Most requirements can be installed via the `requirements.txt` file using
```
pip install -r requirements.txt
```

However there were some modifications made to the `BioPython` Library. The modified `_utils.py` file is stored in the top level directory of this repository and should be used to replace `<your installed packages>\Bio\Phylo\_utils.py` in case you are working with `venv` then it would located at `venv\Lib\site-packages\Bio\Phylo\_utils.py`.

## Config
Below is a list of all config options and a brief explanation. The config file is `config.json`. It is recommended to first take a look at `guide/guide.pdf`

### Detection

#### images_directory_path
The directory where the frames of your video are located

#### first_image_threshold
Threshold to create a mask for inpainting (see `guide/guide.pdf` for more details)

#### inpaint radius
Regulates how much of the local environment is considered when inpainting

#### region_increase_for_blurring
After inpainting a gaussian blur is applied. The `region_increase_for_blurring` option will determine by how much the bounding box - region  is enlarged. Blurring is then performed inside this enlarged region.

#### gaussian_blur_iterations
The amount of times blurring is performed

#### subtraction_image_threshhold
After the created template is subtracted from the current image. Thresholding is performed to obtain the Ecoli on the current picture.

#### height_cutoff & width_cutoff
After thresholding a first detection is performed and bounding rectangles are created. All boxes whose height is less than or equal to `height_cutoff` or  whose width is less than or equal to `width_cutoff` are filtered out.


### Tracking

#### border_distance_cutoff
Ecoli can only appear or disappear from the frame if they are close to the border. This option determines the maximum distance in pixels

#### IoA_threshold
If there are two boxes at time $t+1$ whose Intersection over Area score for a box at time $t$ is above this threshold, duplication is assumed 

#### IoU_threshold
Sets the minimum Intersection over Union score for a box at time $t$ to be mapped to a box at time $t+1$

#### sliding_window_search_height_increase & sliding_window_search_width_increase
If we "lose" a cell or merging happened we try to find the cell again by performing sliding window search. The search region will be it's old bounding box increased by a certain amount in height and width specified by these options.

### Forest
In `line 492` in `_utils.py` you might want to adjust the second  value according to the duration of video. Depending on the duration maybe also other offsets need to be changed. It might be necessary to modify `line 515` and `line 435` to adjust the offsets for the labels  

### Disclaimer
All those parameters may potentially be required to be finetuned when working with a new video.

### Labeling
When `labelImg` opens you first need to press `Open Dir` and select the folder with your images (usually `./images`) and then press `Change Save Dir` to select the directory where the label files are being stored. This directory should also contain a file called `classes.txt` which contains all the different classes. For now it would just look like this
```
Ecoli
```  

### How to start
If you have everthing set up you can start tracking with
```
python track.py
```
and if you just want to use the code for generating labeled training data you can run
```
python automatic_labeling.py
```

### Export
Once your tracking has finished you can export it with
```
python export.py
```


### Questions
If you have
any questions regarding the source code just contact me on Discord: elson1608.