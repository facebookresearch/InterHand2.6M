# InterHand2.6M dataset

<p align="middle">
<img src="https://drive.google.com/uc?export=view&id=1z9N0FDVyHMmaFpn2-NxXxVBkOtmnND2n" width="260" height="160"><img src="https://drive.google.com/uc?export=view&id=13jImH8aWcY408JLTLSSoFqffRVrJ2nn2" width="260" height="160"><img src="https://drive.google.com/uc?export=view&id=1mW8oPeyUp0woghHOA9EBgo845onrDAU3" width="260" height="160">
</p>
<p align="middle">
<img src="https://drive.google.com/uc?export=view&id=1Z_001absV25Jm8l3kn6EOiaiC5O4QmeC" width="390" height="240"><img src="https://drive.google.com/uc?export=view&id=1Ms1ARdGPNaJa-_mC6dQkIWNtvtxAshUc" width="390" height="240">
</p>

## Download link
* [Images]()
* [Annotations](https://drive.google.com/drive/folders/1qglqTVYovTBv9XR1Mu1AO5j6OPlvBzqx?usp=sharing)

## Directory
The `${ROOT}` is described as below.
```
${ROOT}
|-- images
|   |-- train
|   |   |-- Capture0 ~ Capture26
|   |-- val
|   |   |-- Capture0
|   |-- test
|   |   |-- Capture0 ~ Capture7
|-- annotations
|   |-- skeleton.txt
|   |-- subject.txt
|   |-- all
|   |-- human_annot
|   |-- machine_annot
```

## Annotaion files
* Using Pycocotools for the data load is recommended. Run `pip install pycocotools`.
* `skeleton.txt` contains information about hand hierarchy (keypoint name, keypoint index, keypoint parent index).
* `subject.txt` contains information about subject (subject_id, subject directory, subject gender).

There are three `.json` files.

```
InterHand2.6M_$DB_SPLIT_data.json: dict
|-- 'images': [image]
|-- 'annotations': [annotation]

image: dict
|-- 'id': int (image id)
|-- 'file_name': str (image file name)
|-- 'width': int (image width)
|-- 'height': int (image height)
|-- 'capture': int (capture id)
|-- 'subject': int (subject id)
|-- 'seq_name': str (sequence name)
|-- 'camera': str (camera name)
|-- 'frame_idx': int (frame index)

annotation: dict
|-- 'id': int (annotation id)
|-- 'image_id': int (corresponding image id)
|-- 'bbox': list (bounding box coordinates. [xmin, ymin, width, height])
|-- 'joint_valid': list (can this annotaion be use for hand pose estimation training and evaluation? 1 if a joint is annotated and inside of image. 0 otherwise)
|-- 'hand_type': str (one of 'right', 'left', and 'interacting')
|-- 'hand_type_valid': int (can this annotation be used for handedness estimation training and evaluation? 1 if hand_type in ('right', 'left') or hand_type == 'interacting' and np.sum(joint_valid) > 30, 0 otherwise.)
```

```
InterHand2.6M_$DB_SPLIT_camera.json
|-- str (capture id)
|   |-- 'campos'
|   |   |-- str (camera name): [x,y,z] (camera position)
|   |-- 'camrot'
|   |   |-- str (camera name): 3x3 list (camera rotation matrix)
|   |-- 'focal'
|   |   |-- str (camera name): [focal_x, focal_y] (focal length of x and y axis
|   |-- 'princpt'
|   |   |-- str (camera name): [princpt_x, princpt_y] (principal point of x and y axis)
```

```
InterHand2.6M_$DB_SPLIT_joint_3d.json
|-- str (capture id)
|   |-- str (frame idx): Jx3 list (3D joint coordinates in the world coordinate system.)
```


## InterHand2.6M in 30 fps
* Above InterHand2.6M is downsampled to 5 fps to remove redundancy.
* We additionally release InterHand2.6M in 30 fps for the video-related research.
* It has exactly same directory structure with that of the InterHand2.6M in 5 fps
* Download link: [Images]() and [Annotations](https://drive.google.com/drive/folders/1va3FctE_7n9YUMEUzRFD2xUfpsocX_Bh?usp=sharing)

# Reference  
```  
@InProceedings{Moon_2020_ECCV_InterHand2.6M,  
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},  
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},  
booktitle = {European Conference on Computer Vision (ECCV)},  
year = {2020}  
}  
```
