import zmq
import argparse
from ipaddress import ip_address

parser = argparse.ArgumentParser(description='Check in on your Hal.')
parser.add_argument('--host', type=ip_address, required=True, help="IP address of Hal") 
args = parser.parse_args()

context = zmq.Context()
socket = context.socket(zmq.SUB)

socket.connect(f'tcp://{args.host}:5555')
socket.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    msg = socket.recv_string()
    print(msg)
