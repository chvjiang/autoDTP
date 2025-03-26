import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import os

import argparse

def worker(queue, count):
    while True:
        item = queue.get()
        if item is None:
            break
            
        # Perform some operation on the item
        command = (f" wget -c -P DATA {item} ")
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()
    queue.task_done()
    

def main():
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    workers = 10
    
    processes = []
    for _ in range(workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count)
        )
        process.daemon = True
        process.start()
        processes.append(process)

    object_paths_list = [f'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval{str(i+1).zfill(2)}_blobs.tgz' for i in range(workers)]
    object_paths_list.sort()

    for object_path in object_paths_list:
        queue.put(object_path)
        
    # Add sentinels to the queue to stop the worker processes
    for i in range(workers):
        queue.put(None)
        
    # Wait for all tasks to be completed
    queue.join()
    for process in processes:
        process.join()
    
if __name__ == '__main__':
    main()
