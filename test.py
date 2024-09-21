import pymeocap
import asyncio
import time

import torch
from pygame.time import Clock
from scipy.spatial.transform import Rotation


async def main():
    m = pymeocap.Meocap(10)
    await m.connect()
    m.start()
    clock = Clock()
    while True:
        clock.tick(10)
        rpt = m.poll()
        if rpt is not None:
            print(rpt)
            euler = Rotation.from_matrix(torch.transpose(Rotation.from_quat(rpt[5].rot), 0, 1)).as_euler("xyz", degrees=True)



pymeocap.enable_log("info")
asyncio.run(main())
