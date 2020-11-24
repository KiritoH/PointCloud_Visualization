# -*- coding: utf-8 -*-
"""
# @File Name: draw_txt.py
# @Author : KiritoH
# @Data : 2020/11/23
# @Brief: PyCharm
"""
'''
这个文件主要实现点云的可视化(txt文件),解耦出一个方便执行的版本,要求是输入点云,输出图像
'''
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import plotly.graph_objects as go
import plotly.express as px
from path import Path

random.seed = 42


# 读off文件
'''
这里注意off的数据格式是这样的:
OFF
顶点数,面数,边数
//接下来两大部分
第一部分: 顶点坐标(x,y,z)
第二部分: 面的顶点索引(面的顶点数,索引1,索引2,索引3...(第一个数是多少,就有几个,要记住这是索引,代表的是第几个点,而非坐标))
'''
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    # 分别拿到顶点数,面数,边数,file.readline是读一行的意思
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def read_txt(file):
    # 分别拿到顶点数,面数,边数,file.readline是读一行的意思
    verts = [[float(s) for s in file.readline().strip().split(',')] for i_vert in range(10000)]
    return verts


# mesh或点云的动画旋转
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

# 点云可视化(基于上面实现)
def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs,
                         mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    # 文件导出需要用到plotly-orca
    # 安装方法: conda install -c plotly plotly-orca
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/fig1.svg")


# 用于点采样??
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points

# 标准化(将坐标变换到(-1,1)内)
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud





# 规范一点
if __name__ == "__main__":
    path = Path("/Users/kirito/学习/Project/")
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)};
    with open(path / "pointnet/bottle_0001.txt", 'r') as f:
        verts = read_txt(f)
    x, y, z = (np.array(verts).T)[:3,:]
    pcshow(x, y, z)
    #visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i, j=j, k=k)]).show()
    #pcshow(x, y, z)
