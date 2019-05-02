#! /usr/bin/env python
# -*- coding=utf8 -*-
import json
from OpenGL.GL import *
#glCallList, glClear, glClearColor, glColorMaterial, glCullFace, glDepthFunc, glDisable, glEnable,\
#                      glFlush, glGetFloatv, glLightfv, glLoadIdentity, glMatrixMode, glMultMatrixf, glPopMatrix, \
#                      glPushMatrix, glTranslated, glViewport, \
#                      GL_AMBIENT_AND_DIFFUSE, GL_BACK, GL_CULL_FACE, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, \
#                      GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FRONT_AND_BACK, GL_LESS, GL_LIGHT0, GL_LIGHTING, \
#                      GL_MODELVIEW, GL_MODELVIEW_MATRIX, GL_POSITION, GL_PROJECTION, GL_SPOT_DIRECTION
from OpenGL.constants import GLfloat_3, GLfloat_4
from OpenGL.GLU import gluPerspective, gluUnProject
from OpenGL.GLUT import *
#import glutCreateWindow, glutDisplayFunc, glutGet, glutInit, glutInitDisplayMode, glutGetWindow,\
#                        glutInitWindowSize, glutMainLoop, glutPostRedisplay, glutIdleFunc, glutDestroyWindow, \
#                        GLUT_SINGLE, GLUT_RGB, GLUT_WINDOW_HEIGHT, GLUT_WINDOW_WIDTH

import numpy
import sys
import copy
import select
from numpy.linalg import norm, inv
from threading import Timer

try:
    import queue
except ImportError:
    import Queue as queue

from trackball import rotateMatrix
from interaction import Interaction
from primitive import init_primitives, G_OBJ_PLANE
from node import Sphere, Cube, SnowFigure, HierarchicalNode, Q, textOut
from scene import Scene

twmap = {
        0: ([ 0,  1,  2,  3,  4,  5,  6,  7,  8],  [0.0, 0.0,  0.0314, 1.0],  [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), 
        1: ([ 9, 10, 11, 12, 13, 14, 15, 16, 17],  [0.0, 0.0,  0.0314, 1.0],  [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), 
        2: ([18, 19, 20, 21, 22, 23, 24, 25, 26],  [0.0, 0.0,  0.0314, 1.0],  [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), 
        3: ([ 0,  1,  2,  9, 10, 11, 18, 19, 20],  [0.0, -0.0314, 0.0, 1.0],  [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]), 
        4: ([ 3,  4,  5, 12, 13, 14, 21, 22, 23],  [0.0, -0.0314, 0.0, 1.0],  [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]), 
        5: ([ 6,  7,  8, 15, 16, 17, 24, 25, 26],  [0.0, -0.0314, 0.0, 1.0],  [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]), 
        6: ([ 0,  3,  6,  9, 12, 15, 18, 21, 24],  [ 0.0314, 0.0, 0.0, 1.0],  [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), 
        7: ([ 1,  4,  7, 10, 13, 16, 19, 22, 25],  [ 0.0314, 0.0, 0.0, 1.0],  [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), 
        8: ([ 2,  5,  8, 11, 14, 17, 20, 23, 26],  [ 0.0314, 0.0, 0.0, 1.0],  [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), 
        }

twplan = {
        0: [[(0,6,1),(2,0,3),(5,2,-1),(4,8,-3)],[(1,0,0)]],
        1: [[(0,3,1),(2,1,3),(5,5,-1),(4,7,-3)],[]],
        2: [[(0,0,1),(2,2,3),(5,8,-1),(4,6,-3)],[(3,0,1)]],
        3: [[(1,0,1),(2,0,1),(3,0,1),(4,0,1)],[(0,0,1)]],
        4: [[(1,3,1),(2,3,1),(3,3,1),(4,3,1)],[]],
        5: [[(1,6,1),(2,6,1),(3,6,1),(4,6,1)],[(5,0,0)]],
        6: [[(1,2,3),(0,2,3),(3,6,-3),(5,2,3)],[(2,0,0)]],
        7: [[(1,1,3),(0,1,3),(3,7,-3),(5,1,3)],[]],
        8: [[(1,0,3),(0,0,3),(3,8,-3),(5,0,3)],[(4,0,1)]],
        }

tips=[  [-2, -1.2, 1.5, '6'], [-2, -0.2, 1.5, '5'], [-2, 0.8, 1.5, '4'], 
        [-1.2, -2, 1.5, '9'], [-0.2, -2, 1.5, '8'], [0.8, -2, 1.5, '7'], 
        [-2, 1.5, -1.0, '3'], [-2, 1.5, 0., '2'], [-2, 1.5, 1, '1'], 
        ]

STYLE = {
        'fore':
        {   # 前景色
            'black'    : 30,   #  黑色
            'red'      : 31,   #  红色
            'green'    : 32,   #  绿色
            'yellow'   : 33,   #  黄色
            'blue'     : 34,   #  蓝色
            'purple'   : 35,   #  紫红色
            'cyan'     : 36,   #  青蓝色
            'white'    : 37,   #  白色
            },

        'back' :
        {   # 背景
            'black'     : 40,  #  黑色
            'red'       : 41,  #  红色
            'green'     : 42,  #  绿色
            'yellow'    : 43,  #  黄色
            'blue'      : 44,  #  蓝色
            'purple'    : 45,  #  紫红色
            'cyan'      : 46,  #  青蓝色
            'white'     : 47,  #  白色
            },

        'mode' :
        {   # 显示模式
            'mormal'    : 0,   #  终端默认设置
            'bold'      : 1,   #  高亮显示
            'underline' : 4,   #  使用下划线
            'blink'     : 5,   #  闪烁
            'invert'    : 7,   #  反白显示
            'hide'      : 8,   #  不可见
            },

        'default' :
        {
            'end' : 0,
            },
        }

def UseStyle(string, mode = '', fore = '', back = ''):
    mode  = '%s' % STYLE['mode'][mode] if STYLE['mode'].has_key(mode) else ''
    fore  = '%s' % STYLE['fore'][fore] if STYLE['fore'].has_key(fore) else ''
    back  = '%s' % STYLE['back'][back] if STYLE['back'].has_key(back) else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end  = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s'%(style, string, end)

def coltxt(string, ind):
    style = '\033[%sm' % ind
    end  = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s'%(style, string, end)

class Viewer(object):
    def __init__(self):
        """ Initialize the viewer. """
        self.init_interface()
        self.init_opengl()
        self.init_scene()
        self.initRubic()
        self.init_interaction()
        init_primitives()
        self.Q = queue.Queue()

    def init_interface(self):
        """ initialize the window and register the render function """
        glutInit()
        glutInitWindowSize(1000, 600)
        glutCreateWindow("魔方求解")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutDisplayFunc(self.render)

    def init_opengl(self):
        """ initialize the opengl settings to render the scene """
        self.inverseModelView = numpy.identity(4)
        self.modelView = numpy.identity(4)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 0, 1, 0))
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.4, 0.4, 0.4, 0.0)

    def initRubic(self):
        self.anim=0
        self.rotateCnt=0
        self.boxMap=[i for i in xrange(27)]
        self.plane=[]
        self.cmdbuf=[]
        for i in xrange(6):
            self.plane.append([])
            for j in xrange(3):
                self.plane[i].append([i]*3)

    def init_scene(self):
        """ initialize the scene object and initial scene """
        self.scene = Scene()
        self.outStr = ''
        #self.create_sample_scene()
        self.loadScene('rubic.json')
        

    def loadScene(self, fn):
        scencedat = json.load(file(fn)) 
        scencedat['node'].reverse()
        self.rubic=[]
        self.planmap = {}

        for ind, nodedat in enumerate(scencedat['node']):
            node = HierarchicalNode()
            node.translate(nodedat['pos'][0], nodedat['pos'][1], nodedat['pos'][2]) 
            for i in xrange(6):
                subnode=Q(i)
                subnode.ind = ind
                #subnode.showind=i+1
                if nodedat['col'][i] >= 6: 
                    subnode.no4back=1

                subnode.color_index = nodedat['col'][i]

                if nodedat['col'][i]<6:
                    x=nodedat['pos'][0]
                    y=nodedat['pos'][1]
                    z=nodedat['pos'][2]
                    #planind={0:(4,1-y,z+1),1:(2,y+1,z+1),2:(1,x+1,z+1),3:(3,1-x,z+1),4:(0,1+x,1+y),5:(5,1+x,1-y)}[i]
                    planind={0:(4,1-y,z+1),1:(2,1-y,1-z),2:(5,1-z,x+1),3:(0,z+1,x+1),4:(3,1-y,1-x),5:(1,1-y,1+x)}[i]
                    self.planmap[planind]=subnode
                    #print '....', self.planmap[palnind], i, planind, subnode.color_index
                    #self.boxmap[boxind]=[(i,x,y),(i,x,y)]

                node.append(subnode)
            self.rubic.append(node)
            self.scene.add_node(node)
        #print "planmap", len(self.planmap), self.planmap

    def create_sample_scene(self):
        cube_node = Cube()
        cube_node.translate(2, 0, 2)
        cube_node.color_index = 1
        self.scene.add_node(cube_node)

        sphere_node = Sphere()
        sphere_node.translate(-2, 0, 2)
        sphere_node.color_index = 3
        self.scene.add_node(sphere_node)

        #hierarchical_node = SnowFigure()
        #hierarchical_node.translate(-2, 0, -2)
        #self.scene.add_node(hierarchical_node)

    def init_interaction(self):
        """ init user interaction and callbacks """
        self.interaction = Interaction()
        self.interaction.register_callback('cmd', self.cmd)
        #self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)
        #self.interaction.register_callback('place', self.place)
        #self.interaction.register_callback('rotate', self.rotate)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        self.interaction.register_callback('scale', self.scale)

    def main_loop(self):
        glutMainLoop()

    def render(self):
        """ for the scene """
        self.init_view()

        self.scene.showback=False
        glEnable(GL_LIGHTING)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glLoadIdentity()
        loc = self.interaction.translation
        glTranslated(-2.0, 0.0, 0.0)
        glTranslated(loc[0], loc[1], loc[2])
        glMultMatrixf(self.interaction.trackball.matrix)

        # store the inverse of the current modelview.
        currentModelView = numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        self.modelView = numpy.transpose(currentModelView)
        self.inverseModelView = inv(numpy.transpose(currentModelView))

        # render the scene. This will call the render function for each object in the scene
        for t in tips:
            textOut(t[0], t[1], t[2], t[3])

        glCullFace(GL_BACK)
        self.scene.render()

        # draw the grid
        glDisable(GL_LIGHTING)
        glCallList(G_OBJ_PLANE)

        self.scene.showback=True
        glLoadIdentity()
        loc = self.interaction.translation
        glTranslated(3.0, 0.0, 0.0)
        glTranslated(loc[0], loc[1], loc[2])
        #glMultMatrixf([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1.2]])
        glMultMatrixf([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1.2]])
        glMultMatrixf(self.interaction.trackball.matrix)

        # store the inverse of the current modelview.
        currentModelView = numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        self.modelView = numpy.transpose(currentModelView)
        self.inverseModelView = inv(numpy.transpose(currentModelView))

        # render the scene. This will call the render function for each object in the scene
        glCullFace(GL_FRONT)
        self.scene.render()

        # draw the grid
        glDisable(GL_LIGHTING)
        glCallList(G_OBJ_PLANE)

        glPopMatrix()

        if self.outStr:
            textOut(-6.0, 3.0, 0, this.outStr)

        # flush the buffers so that the scene can be drawn
        glFlush()

    def init_view(self):
        """ initialize the projection matrix """
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        aspect_ratio = float(xSize) / float(ySize)

        # load the projection matrix. Always the same
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, xSize, ySize)
        gluPerspective(30, aspect_ratio, 0.1, 1000.0)
        glTranslated(0, 0, -15)
        glutIdleFunc(self.idleFn)

    def idleFn(self):
        if select.select([sys.stdin,],[],[],0.0)[0]:
            cmds=raw_input()
            if len(cmds)>1 and cmds[0] in 'ip':
                self.cmd(cmds, 0 ,0)
                return
            if len(cmds)>0:
                self.cmdbuf.extend(list(cmds))
                self.cmd(self.cmdbuf.pop(0), 0, 0)

    def get_ray(self, x, y):
        """ Generate a ray beginning at the near , in the direction that the x, y coordinates are facing
            Consumes: x, y coordinates of mouse on screen
            Return: start, direction of the ray """
        self.init_view()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # get two points on the line.
        start = numpy.array(gluUnProject(x, y, 0.001))
        end = numpy.array(gluUnProject(x, y, 0.999))

        # convert those points into a ray
        direction = end - start
        direction = direction / norm(direction)
        return (start, direction)

    def setPlane(self, pind, row, col, crl):
        self.plane[pind][row][col]=crl
        self.planmap[(pind,row,col)].color_index=crl

    def setbox(self, inds):
        for pind,ps in enumerate(inds.split(';')):
            for rowind,p in enumerate(ps.split(',')):
                if len(p)>0 and len(p)!=3:
                    p=p[:3]
                    p=p+"".join([p[-1]]*(3-len(p)))
                for col,v in enumerate(p):
                    self.setPlane(pind, rowind, col, int(v))

    def cmd(self, key, x, y):
        rotmap='!@#$%^&*('
        if key >= '1' and key <= '9':
            self.twistFace(ord(key)-ord('1'), self.anim, 0)
        elif key in rotmap:
            self.twistFace(rotmap.find(key), self.anim, 1)
        if key == 'q':
            exit(0)
        if key[0] == 'i':
            self.setbox(key[1:])
            self.dumpRubic()
        if key[0]=='p' and len(key)>=5:    
            self.setPlane(int(key[1]), int(key[2]), int(key[3]), int(key[4]))
        if key == 'c':
            self.rubic = []
            self.scene.node_list = []
            self.init_scene()
            self.initRubic()
        if key==' ':
            import resolve
            reload(resolve)
            dupplan = copy.copy(self.plane)
            cmds = resolve.run(dupplan, self.adjustPlan)
            for c in cmds:
                if (c >= '1' and c <= '9') or c in rotmap:
                    self.cmdbuf.append(c)
            if len(cmds)>0:
                self.cmdbuf = list(cmds)
                self.cmd(self.cmdbuf.pop(0), 0, 0)
        if key=='a' or key=='A':
            self.anim = 0 if key == 'A' else 1
        if key == 'r':
            import random
            for i in xrange(80):
                self.twistFace(int(random.random()*9), 1, 0)
        glutPostRedisplay()

    def rotateFn(self):
        for i in self.rotateTarget:
            self.rubic[i].translation_matrix = numpy.dot(self.rotateMat, self.rubic[i].translation_matrix)
        self.rotateCnt += 1

    def dumpFace(self, face):
        outstr=''
        for v in xrange(3):
            outstr+=coltxt(" %d "%face[v], 41+face[v])
        print outstr,

    def dumpRubic(self):
        print '---'*3
        for j in xrange(3):
            self.dumpFace(self.plane[0][j])
            print
        print
        for j in xrange(3):
            for v in xrange(1,5):
                self.dumpFace(self.plane[v][j])
            print
        print
            #    UseStyle("%d"%self.plane[1][j][v], back='red')
            #print self.plane[1][j], self.plane[2][j], self.plane[3][j], self.plane[4][j]
        for j in xrange(3):
            self.dumpFace(self.plane[5][j])
            print 
            #print self.plane[5][j]

        print '==='*3
        return

        for i in xrange(1,5):
            for j in xrange(3):
                print self.plane[i][j]
            print '==='*3
        print '---'*3
        return
        for i in xrange(3):
            print '--'*5
            for j in xrange(3):
                for k in xrange(3):
                    print '%2d'%self.boxMap[k+j*3+9*i],
                print 
        print '=='*6

    def tranf(self, idlst, pf, tbl):
        tmp = idlst[pf[0]][tbl[0]/3][tbl[0]%3] 
        for i in xrange(len(tbl)-1):
            idlst[pf[i]][tbl[i]/3][tbl[i]%3] = idlst[pf[i+1]][tbl[i+1]/3][tbl[i+1]%3]
        idlst[pf[-1]][tbl[-1]/3][tbl[-1]%3] = tmp

    def adjustPlan(self, plan, rotInd,  reverse):
        planscroll, planrot = twplan[rotInd]
        pf = [planscroll[0][0], planscroll[1][0], planscroll[2][0], planscroll[3][0]]
        if reverse:
            pf.reverse()
        for i in xrange(3):
            ri = [planscroll[0][1]+i*planscroll[0][2], planscroll[1][1]+i*planscroll[1][2], planscroll[2][1]+i*planscroll[2][2], planscroll[3][1]+i*planscroll[3][2]]
            if reverse:
                ri.reverse()
            self.tranf(plan, pf, ri)
        
        if len(planrot) == 0: return

        pf = [planrot[0][0]]*4
        if planrot[0][2] > 0:
            ri = [3, 7, 5, 1]
            rc = [6, 8, 2, 0]
        else:
            ri = [1, 5, 7, 3]
            rc = [0, 2, 8, 6]

        if reverse:
            ri.reverse()
            rc.reverse()
        self.tranf(plan, pf, ri)
        self.tranf(plan, pf, rc)

    def adjust(self, reverse):
        self.rotateMat = twmap[self.rotateInd][2]
        if reverse:
            self.rotateMat = numpy.linalg.inv(self.rotateMat)
        for i in self.rotateTarget:
            self.rubic[i].translation_matrix = numpy.dot(self.rotateMat, self.rubic[i].translation_matrix0)

        rot = twmap[self.rotateInd][0]
        ri = [6, 8, 2, 0]
        if reverse:
            ri.reverse()
        tmp = self.boxMap[rot[ri[0]]] 
        for i in xrange(len(ri)-1):
            self.boxMap[rot[ri[i]]] = self.boxMap[rot[ri[i+1]]]
        self.boxMap[rot[ri[-1]]] = tmp
        #print
        ri = [3, 7, 5, 1]
        if reverse:
            ri.reverse()
        tmp = self.boxMap[rot[ri[0]]] 
        for i in xrange(len(ri)-1):
            self.boxMap[rot[ri[i]]] = self.boxMap[rot[ri[i+1]]]
        self.boxMap[rot[ri[-1]]] = tmp
        
        self.adjustPlan(self.plane, self.rotateInd, reverse)
        self.dumpRubic()
        if len(self.cmdbuf)>0:
            self.cmd(self.cmdbuf.pop(0), 0, 0)

    def animStep(self, reverse):
        if self.rotateCnt > 0: 
            self.rotateFn()
            glutPostRedisplay()
            if self.rotateCnt > 25:
                self.rotateCnt=0
                self.adjust(reverse)
            else:
                glutTimerFunc(10, self.animStep, reverse)

    def twistFace(self, ind, noAnim, reverse):
        if ind not in twmap: return
        if self.rotateCnt > 0: return

        self.rotateInd = ind
        self.rotateTarget = []

        for bi in twmap[ind][0]:
            self.rotateTarget.append(self.boxMap[bi]) 
        print ' %s '%(['↩️ ..', '.↩️ .', '..↩️ ', '⬅️ ..', '.⬅️ .', '..⬅️ ', '..⬆️ ', '.⬆️ .', '⬆️  ..'][ind] 
                if reverse else ['↪️ ..', '.↪️ .', '..↪️ ', '➡️ ..', '.➡️ .', '..➡️ ', '..⬇️ ', '.⬇️ .', '⬇️  ..'][ind] )
        for i in self.rotateTarget:
            self.rubic[i].translation_matrix0 = self.rubic[i].translation_matrix
    
        if noAnim:
            self.adjust(reverse)
        else:
            self.rotateMat = rotateMatrix(twmap[ind][1])
            if reverse:
                self.rotateMat = numpy.linalg.inv(self.rotateMat)
            self.rotateCnt = 1
            self.animStep(reverse)

    def pick(self, x, y):
        """ Execute pick of an object. Selects an object in the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.pick(start, direction, self.modelView)

    def place(self, shape, x, y):
        """ Execute a placement of a new primitive into the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.place(shape, start, direction, self.inverseModelView)

    def rotate(self, x, y):
        """ Execute a move command on the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.rotate_selected(start, direction, self.inverseModelView)

    def move(self, x, y):
        """ Execute a move command on the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.move_selected(start, direction, self.inverseModelView)

    def rotate_color(self, forward):
        """ Rotate the color of the selected Node. Boolean 'forward' indicates direction of rotation. """
        self.scene.rotate_selected_color(forward)

    def scale(self, up):
        """ Scale the selected Node. Boolean up indicates scaling larger."""
        self.scene.scale_selected(up)

if __name__ == "__main__":
    viewer = Viewer()
    viewer.main_loop()
