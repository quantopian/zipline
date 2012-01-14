import zmq
from data.sources.equity import *

def main():
    context = zmq.Context()
    controller = context.socket(zmq.SUB)
    controller.connect("tcp://127.0.0.1:10099")    
    controller.setsockopt(zmq.SUBSCRIBE, '')
    while True:
        try:
            message = controller.recv()
            print message
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                raise err
     
if __name__ == "__main__":
    main()              
     