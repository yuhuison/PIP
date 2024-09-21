import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import threading
import pymeocap
import asyncio
import time
from test_rotation_filter import RotationFilter3D
from scipy.spatial.transform import Rotation as R
sensor_rot = [1,0,0,0]
async def sensor_main():
    m = pymeocap.Meocap(1)
    await m.connect()
    m.start()
    global sensor_rot
    while True:
        time.sleep(0.016)
        rpt = m.poll()
        if rpt is not None:
            sensor_rot = rpt[0].rot

def reading_thread():
    pymeocap.enable_log("info")
    asyncio.run(sensor_main())


threading.Thread(target=reading_thread).start()

def main():
    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((640, 480), video_flags)
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(640, 480)
    init()
    frames = 0
    ticks = pygame.time.get_ticks()
    filter = RotationFilter3D(0.01,0.00)
    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break

        [w, nx, ny, nz] = read_data()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw(w, nx, ny, nz,-1.0,True)
        [w, nx, ny, nz] = filter.update(R.from_quat([w, nx, ny, nz])).as_quat()
        draw(w, nx, ny, nz,1.0)
        pygame.display.flip()
        frames += 1
    print("fps: %d" % ((frames * 1000) / (pygame.time.get_ticks() - ticks)))



def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0 * width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)





def read_data():
    return sensor_rot



def draw(w, nx, ny, nz,t_z,draw_text = False):

    glLoadIdentity()
    glTranslatef(t_z, 0,-7.0 )

    if draw_text:
        drawText((-2.6, 1.8, 2), "PyTeapot", 18)
        drawText((-2.6, 1.6, 2), "Module to visualize quaternion or Euler angles data", 16)
        drawText((-2.6, -2, 2), "Press Escape to exit.", 16)


        [yaw, pitch, roll] = quat_to_ypr([w, nx, ny, nz])
        drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" % (yaw, pitch, roll), 16)
    glRotatef(2 * math.acos(w) * 180.00 / math.pi, -1 * nx, nz, ny)


    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def quat_to_ypr(q):
    yaw = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    pitch *= 180.0 / math.pi
    yaw *= 180.0 / math.pi
    yaw -= -0.13  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min
    roll *= 180.0 / math.pi
    return [yaw, pitch, roll]


if __name__ == '__main__':
    main()