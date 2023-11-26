import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time

def task(v, x):
    """session state does not work here"""
    time.sleep(1)
    return v * v + x


if __name__ == '__main__':
    num_workers = 2
    jobs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    processed_jobs = []
    result = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for j in jobs:
            pj = executor.submit(task, v=j, x=j+1)
            processed_jobs.append(pj)

        for future in concurrent.futures.as_completed(processed_jobs):
            try:
                res = future.result()
                result.append(res)
            except concurrent.futures.process.BrokenProcessPool as ex:
                    raise Exception(ex)
    print(result)
    
