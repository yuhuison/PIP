import pymeocap
import asyncio
import time

async def main():
    m = pymeocap.Meocap(6)
    await m.connect()
    m.start()
    while True:
        time.sleep(0.016)
        rpt = m.poll()
        print(rpt)


pymeocap.enable_log("info")
asyncio.run(main())
