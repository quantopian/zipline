import multiprocessing
import test
import test2

def main():
    proc1 = multiprocessing.Process(target=test2.main)
    proc1.start()
    proc2 = multiprocessing.Process(target=test.main)
    proc2.start()
    
if __name__ == "__main__":
    main() 