import multiprocessing
import time
 
 # 작업자 함수 정의
def work_log(task):
    name, duration = task
    print(f"Process {name} waiting {duration} seconds")
    time.sleep(duration)
    print(f"Process {name} Finished.")
 
 # Pool 설정 및 실행
if __name__ == '__main__':
    work = [('A', 5), ('B', 2), ('C', 1), ('D', 3)]
 
    print("Starting tasks...")
    with multiprocessing.Pool(processes=2) as pool:
         # pool.map()을 사용하여 작업 실행
        pool.map(work_log, work)
    print("All tasks finished.")
