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
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return image, current_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    ## PARAMS
    REQUESTED_ID = 0
    total_count = 0
    last_count = 0
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    #     prob_threshold = args.prob_threshold
 
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
#     print("infer network")

    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()
#     print("net_input_shape", net_input_shape)
    
    ### TODO: Loop until stream is over ###
#     print(args.input)
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        global initial_w, initial_h, prob_threshold
 
        initial_w = cap.get(3)
        initial_h = cap.get(4)
    
        prob_threshold = args.prob_threshold
        key_pressed = cv2.waitKey(60)
#         print(net_input_shape)
        
        ### TODO: Pre-process the image as needed ###   Need revision   
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        
#         p_frame = p_frame.reshape((net_input_shape[0], net_input_shape[1],net_input_shape[2], net_input_shape[3]))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        
        inference_start = time.time()
#         print(inference_start)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(REQUESTED_ID, p_frame)
#         print("   after infert execnet")
        
        ### TODO: Wait for the result ###
        if infer_network.wait(REQUESTED_ID) == 0:
            time_inference = time.time() - inference_start
            ### TODO: Get the results of the inference request ###
#             print("antes de result")
            result = infer_network.get_output(REQUESTED_ID)
#             print(result)
            ### TODO: Extract any desired stats from the results ### Need Revision
            perf_count = infer_network.get_performance(REQUESTED_ID)
#             print(perf_count)
            ### TODO: Calculate and send relevant information on ###
            out_frame, count_in_frame = get_boxes(frame, result)
#             print(count_in_frame)
            inf_time_measure = "Inference time: {:.3f}ms".format(time_inference * 1000)
#             print("abcd", inf_time_measure)
#             cv2.putText(out_frame, inf_time_measure, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)   
            
            
                                 
            ### Topic "person": keys of "count" and "total" ###
            #### si el contador actual es mayor se actualiza
            if count_in_frame > last_count:
                start_time = time.time()
                total_count = total_count + count_in_frame - last_count
                client.publish("person", json.dumps({"total": total_count})) 
                
                 
            ### Topic "person/duration": key of "duration" ###
            if count_in_frame < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))  
                
                
            last_count = count_in_frame  
            client.publish("person", json.dumps({"count": count_in_frame}))
           
        ### TODO: Send the frame to the FFMPEG server ###
#         out_frame = np.ascontiguousarray(out_frame, dtype = np.float32)
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush() 
        
        
        ### TODO: Write an output image if `single_image_mode` ###
        
        
        
#     cap.release()
#     cv2.destroyAllWindows()
#     client.disconnect()
    
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
