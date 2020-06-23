"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt



from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    
    Parse command line arguments.
    :return: command line arguments
    
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,         
                        default=None,                       
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def get_boxes(image, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    
    current_count = 0
    
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_width)
            ymin = int(obj[4] * initial_height)
            xmax = int(obj[5] * initial_width)
            ymax = int(obj[6] * initial_height)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
            
    return image, current_count


def verify_imput(input_source):
    """
    Verify the input pased through the app
    :param input_source: argument input -i
    :type input_source: string
    :return: input or exit the application
    """
    
    result = ""
    ## If the input contains 0, it means we have to connect to the camera. 
    if input_source == "0":
        result = 0
        
    else:
        
        assert os.path.isfile(input_source), "input source does not exist"
        
        if input_source.endswith(".png") or input_source.endswith(".jpg") or input_source.endswith(".bmp"):
            result = input_source
            
        elif input_source.endswith(".mp4") or input_source.endswith(".mkv") or input_source.endswith(".avi"):
            result = input_source
    
    return result
        

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    ## PARAMS
#     global external_count, external_count_zeros
         
    external_count = []
    external_count_zeros = []
    status_count = 0
    REQUESTED_ID = 0
    total_count = 0
    last_count = 0
    list_count = []
    
    # Initialise the class
    infer_network = Network()
 
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()

    dir_source =  args.input

    result_source = verify_imput(dir_source)
 
    cap = cv2.VideoCapture(result_source)
    cap.open(result_source)
    
    if not cap.isOpened():
        log.error("Couldn't open the file {}".format(result_source))
    
    global initial_width, initial_height
    initial_width  = cap.get(3) 
    initial_height = cap.get(4)    
    
  
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
        global prob_threshold
     
        prob_threshold = args.prob_threshold
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###   Need revision   
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        
        p_frame = p_frame.reshape(1, *p_frame.shape)
        inference_start = time.time()
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(REQUESTED_ID, p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait(REQUESTED_ID) == 0:
            time_inference = time.time() - inference_start
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(REQUESTED_ID)

            ### TODO: Extract any desired stats from the results ### Need Revision
            perf_count = infer_network.get_performance(REQUESTED_ID)

            ### TODO: Calculate and send relevant information on ###
            out_frame, count_in_frame = get_boxes(frame, result)

            inf_time_measure = "Inference time: {:.3f}ms".format(time_inference * 1000)

            cv2.putText(out_frame, inf_time_measure, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)   
     
                                 
            ### Topic "person": keys of "count" and "total" ###
            #### si el contador actual es mayor se actualiza
            
            
            #### contar si el 1 aparecio constantemente en los ultimos 4 segundos
            if count_in_frame==1 and status_count == 0:
                external_count.append(1)
#                 print("Persona presnte",np.sum(external_count))

                if np.sum(external_count)>100:
#                     print("----")
#                     print("total porncetaje", np.sum(external_count)/ len(external_count))
#                     if np.sum(external_count)/ len(external_count) >= 0.6:

                    total_count = total_count + 1
#                     print("total", total_count)
                    client.publish("person", json.dumps({"total": total_count}))
                    external_count = []
                    status_count = 1
            
            else:
                external_count_zeros.append(1)
                external_count.append(0)
#                 print("No presente",np.sum(external_count_zeros))
                
                if np.sum(external_count_zeros)>100:
                    external_count_zeros = []
                    external_count = []
                    status_count = 0
                    total_count = 0
                
            if count_in_frame > last_count:
                start_time = time.time()
                
    
            ### Topic "person/duration": key of "duration" ###
            if count_in_frame < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))  
                
                
            last_count = count_in_frame  
            client.publish("person", json.dumps({"count": count_in_frame}))
           
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush() 
        
        
        ### TODO: Write an output image if `single_image_mode` ###
  
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
