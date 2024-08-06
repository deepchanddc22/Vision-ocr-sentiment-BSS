import multiprocessing
import uvicorn

def run_backend1():
    uvicorn.run("ocr.ocr:app", host="0.0.0.0", port=8000)

def run_backend2():
    uvicorn.run("sentiment.sentiment:app", host="0.0.0.0", port=8001)


def run_backend3():
    uvicorn.run("vision.vision:app", host="0.0.0.0", port=8002)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_backend1)
    p2 = multiprocessing.Process(target=run_backend2)
    p3 = multiprocessing.Process(target=run_backend3)


    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
