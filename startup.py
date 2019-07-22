""" Dependencies """
from multiprocessing import Process




def start_full_system():
    process = Process(target=)
    process.start()
    # Should I still join? I don't think should join since the process is an infinite loop. Which will preven the exit from being called.
    process.join()

    # Exit the current process
    exit(0)
