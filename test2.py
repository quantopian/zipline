import zmq

def main():
    context = zmq.Context()
    controller = context.socket(zmq.PUB)
    controller.bind("tcp://127.0.0.1:10099")    
    while True:
        controller.send("HELLO3")
     
if __name__ == "__main__":
    main() 