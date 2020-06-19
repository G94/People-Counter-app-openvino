# People-Counter-app-openvino
This  project was developed on the [Intel Edge for Iot Developers Nanodegree](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131).
The pretrained models were downloaded from thie tensorflow zoo [repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

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

The difference between model accuracy pre - and post-conversion was


## Assess Model Use Cases
Some of the potential use cases of the people counter app are preventing people to get too close to each other:
1. This year we have been hit by a pandmic scenario, so using a camera to warn people to hold distance from each other in massive centers is critic, it will also help in:

- Fullfill protocols of health security for any stablishment.
- Protect people from getting covid19 on their systems.


2. Check maximum capacity of a stablishment.
- It will prevent stablishment of exceed the maximun number of people they can hold on their area.



Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [faster_rcnn_inception_v2_coco]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  
  wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  
 tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
 
 Convert to Intermeditate Representation
 python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
 
 Model Optimizer arguments:
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
 
 
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [ssd_inception_v2_coco_2018_01_28]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  
 wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
 
 
 
 
Common parameters:
        - Path to the Input Model:      /home/workspace/models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
        - Path for generated IR:        /home/workspace/models/ssd_inception_v2_coco_2018_01_28/.
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
  
  
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/workspace/models/ssd_inception_v2_coco_2018_01_28/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/workspace/models/ssd_inception_v2_coco_2018_01_28/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 52.11 seconds. 
  
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer


 python $MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json 
  
  
  
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03]
  - [Model Source]
  
  wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
  
  
root@666985e2a18d:/home/workspace/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03# python $MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  
  
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...


python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3005/fac.ffm


