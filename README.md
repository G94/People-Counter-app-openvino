# People-Counter-app-openvino
This  project was developed on the program nanodegree Intel AI on the Edge.



# Project Write-Up


## Explaining Custom Layers
The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

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


