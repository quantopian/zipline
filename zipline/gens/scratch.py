
# Inside Client
def __init__(self, addresses, ...):
    self.pull_socket = ...
    
    self.control_socket ...

def run(self):
    for message in gen_from_pull(self.pull_socket):
        #Do things with messages.
        heartbeat()
    
    signal_done()
    sys.exit(0)

# Inside Merge
def __init__(self, addresses, source_ids ...):
    self.poller = ... # Poller on multiple xforms, single socket.

    self.processor = ... # Generator that  
    
    self.push_socket = ... # Outbound socket
    
def run(self):
    
    incoming = gen_from_poll(self.poller)# Receives messages from all xforms.
    
    processed = self.processor(incoming, source_ids) # Maintains internal queues and merges.
    
    for message in self.processed:
        heartbeat()
        self.push_socket.send(message)

# Inside 
    
    
