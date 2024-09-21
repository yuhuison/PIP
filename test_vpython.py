import asyncio
import time

from vpython import *
import pymeocap




async def main():
    m = pymeocap.Meocap(1)
    await m.connect()
    m.start()
    while True:
        time.sleep(0.016)
        rpt = m.poll()
        print(rpt)


pymeocap.enable_log("info")
asyncio.run(main())

pass


scene = canvas(title='Rotating Cube with VPython')

cube_raw = box(pos=vector(0, 2, 0), size=vector(1, 1, 1), color=color.red)
cube_filtered = box(pos=vector(0, -2, 0), size=vector(1, 1, 1), color=color.green)
# 动画循环
while True:
    rate(60)
    cube_raw.rotate()






