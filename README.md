# People-Counter-app-openvino
This  project was developed on the [Intel Edge for Iot Developers Nanodegree](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131).

The pretrained models were downloaded from the tensorflow zoo [repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), which provide with a wide variety of models and its implementations.

## Explaining Custom Layers
The process behind converting custom layers involves 

The main potential reasons for handling custom layers are:
- Layers that are not recognised by openvino are going to throw an error in some point of the process.
- They have some differences depending on the original framework, we could add the custom layers as extensions of the Model Optimizer or register the layer as **Custom**, then we use the original framework to calculate the output shape of the layer.

To see the full list of **Supported Layers** go to following [link](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)

## Comparing Model Performance
My method(s) to compare models before and after conversion to Intermediate Representations was by passing de Pedestrian video as input between both implementations:


### Model size
The size of the model pre- and post-conversion was...
| |SSD MobileNet V2 COCO|SSD Coco MobileNet V1|
|-|-|-|
|Before Conversion|68 MB|28 MB|
|After Conversion|65 MB|26 MB|

### Inference Time
The inference time of the model pre- and post-conversion was...
| |SSD MobileNet V2|SSD Coco MobileNet V1|
|-|-|-|
|Before Conversion|50 ms|55 ms|
|After Conversion|60 ms|60 ms|




## Assess Model Use Cases
Some of the potential use cases of the people counter app are preventing people to get too close to each other:
1. This year we have been hit by a pandemic scenario, so using a camera to warn people to hold distance from each other in massive centers is critic, it will also help in:

- Fullfill protocols of health security for any stablishment.
- Protect people from getting covid19 on their systems.

2. Check maximum capacity of a stablishment.
- It will prevent stablishment of exceed the maximun number of people they can hold on their area.

3. It could serve as a system to measure the time a client spend in a queue in order to optimize this delays.
- Banks
- Markets

4. It might be possible to adapt the model to warn officers if burglars are trying to get inside of a stablishment, because of the time they spend now that at nights is difficult to cover wide areas.

## Assess Effects on End User Needs


Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

`Lack of light in the video`: It will directly affect the accuracy of the model , but it could be possible to change the device with better features at performing in poorly lit areas. Another option is to preprocess the image to improve the quality of it.

`Model accuracy`: 
- Model training: the model is validated before convert it to an intermediate representation with a set of images the model have never seen before.

-  Model validation after conversion: In production, It could be possible to store a sample of the frames an the boxes or the labels, to validate if the model is still performing well or it has decrease its performance. we could adjust the transformation with the open vino framework.

`Camera focal length/image`: 
after conversion: In production, It could be possible to store a sample of the frames an the boxes or the labels, to validate if the model is still performing well or it has decrease its performance. we could adjust the transformation with the open vino framework.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

### Model 1: Faster-RCNN & Inception-v2:
<img src = "https://www.researchgate.net/profile/Akif_Durdu/publication/334987612/figure/fig3/AS:788766109224961@1565067903984/High-level-diagram-of-Faster-R-CNN-16-for-generic-object-detection-2-Inception-v2-The.ppm"></img>

Note: you can find out more information in this [link](https://www.researchgate.net/figure/High-level-diagram-of-Faster-R-CNN-16-for-generic-object-detection-2-Inception-v2-The_fig3_334987612)

#### Model Metadata

**Model Name**                                                                                                                                                                                    | Speed (ms) | COCO mAP[^1] | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                                       | 58         | 28           | Boxes

I followed this steps to test the model:
  
#### Download the model into the workspace.
  ```
  wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  ```
  
#### Unzip 
 ```
 tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
 ```
 
 #### Convert to Intermeditate Representation
 ```
 python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
 ```
  
#### Model Optimizer arguments:

```console
Common parameters:
        - Path to the Input Model:      /home/workspace/models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
        - Path for generated IR:        /home/workspace/models/faster_rcnn_inception_v2_coco_2018_01_28/.
        - IR output name:       frozen_inference_graph
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       True
 
 [ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate Representation file is generated with the input image size of a fixed size.
Specify the "--input_shape" command line parameter to override the default shape which is equal to (600, 600).

The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/workspace/models/faster_rcnn_inception_v2_coco_2018_01_28/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/workspace/models/faster_rcnn_inception_v2_coco_2018_01_28/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 145.63 seconds.

``` 
 
 
- The model was insufficient for the app because it was too heavy to be loaded into the workspace. It throw an error related to lack of memory. I also search for solutions but it seems this model wouldn't fit devices like raspberry whose resources are limited.

 
### Model 2: SSD MobileNet v2
<img src = "https://1.bp.blogspot.com/-M8UvZJWNW4E/WsKk-tbzp8I/AAAAAAAAChw/OqxBVPbDygMIQWGug4ZnHNDvuyK5FBMcQCLcBGAs/s640/image5.png"></img>


#### Model Metadata

**Model Name** | Speed (ms) | COCO mAP[^1] | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)                                                                       | 31         | 22           | Boxes

#### Download the model into the workspace.  
``` 
  wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
```  

#### Unzip 
 ```
 tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
 ```  
  
#### Model Optimizer arguments:

```console
root@666985e2a18d:/home/workspace/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03# python $MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json 
```

Actually, This model perform well as the first try, It was well suited for the workspace and I just have to adjust the threshold probability of the boxes.



