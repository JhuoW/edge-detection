from __future__ import division
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, title, savefig
import cv2 as cv
import numpy as np
import os

root = "orginal/"
plain = "plain"
twill = "twill"


# 标准霍夫线变换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 310)  # apertureSize参数默认其实就是3  # 50 310
    # cv.imshow("edges", edges)
    edge = Image.fromarray(edges)
    edge.save("edge.jpeg")
    lines = cv.HoughLines(edges, 1, np.pi / 180, 68)  # 68
    # l1 = lines[:, 0, :]
    # print(l1)
    mink = float('inf')
    maxk = -float('inf')
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        print("x1: %s, y1:%s, x2:%s, y2:%s" % (x1, y1, x2, y2))
        k = (y2 - y1) / (x2 - x1)
        if k > maxk:
            maxk = k
            xmax1 = x1
            ymax1 = y1
            xmax2 = x2
            ymax2 = y2
            lineMax = line
        if k < mink:
            mink = k
            xmin1 = x1
            ymin1 = y1
            xmin2 = x2
            ymin2 = y2
            lineMin = line
    cv.line(image, (xmax1, ymax1), (xmax2, ymax2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
    cv.line(image, (xmin1, ymin1), (xmin2, ymin2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
    crossX = int((maxk * xmax1 - ymax1 - mink * xmin1 + ymin1) / (maxk - mink))
    crossY = int((maxk * mink * (xmax1 - xmin1) + maxk * ymin1 - mink * ymax1) / (maxk - mink))
    print(crossX, 500 - crossY)
    height = 500 - crossY
    print("顶点高度：" + str(int(height)))
    x1 = (-height) / mink + crossX
    x2 = (-height) / maxk + crossX
    print("与x轴交点：%f,%f" % (x1, x2))
    # 底边长度
    xl = abs(x1 - x2)
    cv.circle(image, (crossX, crossY), 3, (0, 255, 0), -1)  # 两直线交点
    cv.circle(image, (xmax2, ymax2), 3, (0, 0, 255), -1)
    cv.circle(image, (xmin1, ymin1), 3, (0, 0, 255), -1)
    vector1 = np.array([xmax2 - crossX, ymax2 - crossY])
    vector2 = np.array([xmin1 - crossX, ymin1 - crossY])
    L1 = np.sqrt(vector1.dot(vector1))
    L2 = np.sqrt(vector2.dot(vector2))
    cos_angle = vector1.dot(vector2) / (L1 * L2)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    print(angle2)
    # cv.imshow("image-lines", image)
    im = Image.fromarray(image)
    # im.save(image_line)
    cv.waitKey(0)
    return angle2, edge, image, height, xl


def line_detection_low(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 310)  # apertureSize参数默认其实就是3  # 50 310
    # cv.imshow("edges", edges)
    edge = Image.fromarray(edges)
    edge.save("edge.jpeg")
    lines = cv.HoughLines(edges, 1, np.pi / 180, 30)  # 68
    # l1 = lines[:, 0, :]
    # print(l1)
    mink = float('inf')
    maxk = -float('inf')
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        print("x1: %s, y1:%s, x2:%s, y2:%s" % (x1, y1, x2, y2))
        k = (y2 - y1) / (x2 - x1)
        if k > maxk:
            maxk = k
            xmax1 = x1
            ymax1 = y1
            xmax2 = x2
            ymax2 = y2
            lineMax = line
        if k < mink:
            mink = k
            xmin1 = x1
            ymin1 = y1
            xmin2 = x2
            ymin2 = y2
            lineMin = line
    cv.line(image, (xmax1, ymax1), (xmax2, ymax2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
    cv.line(image, (xmin1, ymin1), (xmin2, ymin2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
    crossX = int((maxk * xmax1 - ymax1 - mink * xmin1 + ymin1) / (maxk - mink))
    crossY = int((maxk * mink * (xmax1 - xmin1) + maxk * ymin1 - mink * ymax1) / (maxk - mink))
    print(crossX, 250 - crossY)
    height = 250 - crossY
    print("顶点高度：" + str(int(height)))
    x1 = (-height) / mink + crossX
    x2 = (-height) / maxk + crossX
    print("与x轴交点：%f,%f" % (x1, x2))
    # 底边长度
    xl = abs(x1 - x2)
    cv.circle(image, (crossX, crossY), 3, (0, 255, 0), -1)  # 两直线交点
    cv.circle(image, (xmax2, ymax2), 3, (0, 0, 255), -1)
    cv.circle(image, (xmin1, ymin1), 3, (0, 0, 255), -1)
    vector1 = np.array([xmax2 - crossX, ymax2 - crossY])
    vector2 = np.array([xmin1 - crossX, ymin1 - crossY])
    L1 = np.sqrt(vector1.dot(vector1))
    L2 = np.sqrt(vector2.dot(vector2))
    cos_angle = vector1.dot(vector2) / (L1 * L2)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    print(angle2)
    # cv.imshow("image-lines", image)
    im = Image.fromarray(image)
    # im.save(image_line)
    cv.waitKey(0)
    return angle2, edge, image, height, xl


def run(kind, pic_name):
    ##### change here: ######
    # kind = "twill"
    # pic_name = "285"
    #########################
    img_name = root + kind + "/" + pic_name + ".jpg"
    path = "result/" + kind + "/" + pic_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    result_file = path + pic_name + ".jpg"

    img_cutname = path + pic_name + "_cut.jpeg"
    # img_final = root + "_final.jpeg"
    # img_line = root + "_line.jpeg"

    image = Image.open(img_name)
    img_size = image.size
    print("图片宽度和高度分别是{}".format(img_size))
    im = Image.open(img_name)
    x = 0
    y = 0
    w = 1292
    h = 500
    region = im.crop((x, y, x + w, y + h))
    region.save(img_cutname)
    img = cv.imread(img_cutname)
    angle, edge, image2, height, xl = line_detection(img)
    # 面积
    acr = height * xl / 2
    subplot(2, 2, 1)
    imshow(image)
    title("original")
    subplot(2, 2, 2)
    imshow(region)
    # subplot(2, 2, 3)
    # imshow(edge)
    subplot(2, 2, 3)
    imshow(image2)
    title("angle: %.2f° height:%d acr:%d" % (angle, height, acr))
    savefig(result_file)
    show()


def runhigh(kind, pic_name):
    img_name = root + kind + "/" + pic_name + ".jpg"
    path = "result/" + kind + "/" + pic_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    result_file = path + pic_name + ".jpg"

    img_cutname = path + pic_name + "_cut.jpeg"
    image = Image.open(img_name)
    img_size = image.size
    print("图片宽度和高度分别是{}".format(img_size))
    im = Image.open(img_name)
    if kind == "plain":
        x = 300
        y = 200
        w = 700
        h = 300
    else:
        x = 100
        y = 0
        w = 1000
        h = 480
    region = im.crop((x, y, x + w, y + h))
    region.save(img_cutname)
    img = cv.imread(img_cutname)
    angle, edge, image2, height, xl = line_detection(img)
    # 面积
    acr = height * xl / 2
    subplot(1, 2, 1)
    imshow(image)
    title(pic_name)
    # subplot(2, 2, 2)
    # imshow(region)
    # subplot(2, 2, 3)
    # imshow(edge)
    subplot(1, 2, 2)
    imshow(image2)
    title("angle: %.2f° \n height:%d \n acr:%d" % (angle, height, acr))
    savefig(result_file)
    show()


def runlow(kind, pic_name):
    img_name = root + kind + "/" + pic_name + ".jpg"
    path = "result/" + kind + "/" + pic_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    result_file = path + pic_name + ".jpg"

    img_cutname = path + pic_name + "_cut.jpeg"
    image = Image.open(img_name)
    img_size = image.size
    print("图片宽度和高度分别是{}".format(img_size))
    im = Image.open(img_name)
    if kind == "plain":
        x = 300
        y = 200
        w = 700
        h = 280
    else:
        x = 100
        y = 0
        w = 1000
        h = 480
    region = im.crop((x, y, x + w, y + h))
    region.save(img_cutname)
    img = cv.imread(img_cutname)
    angle, edge, image2, height, xl = line_detection_low(img)
    # 面积
    acr = height * xl / 2
    subplot(1, 2, 1)
    imshow(image)
    title(pic_name)
    # subplot(2, 2, 2)
    # imshow(region)
    # subplot(2, 2, 3)
    # imshow(edge)
    subplot(1, 2, 2)
    imshow(image2)
    title("angle: %.2f° \n height:%d \n acr:%d" % (angle, height, acr))
    savefig(result_file)
    show()


# 测试用例 ：p322 & 285
if __name__ == '__main__':
    # 用于较高的顶点
    # runhigh(twill, "282")
    # 用于较低的顶点
    runlow(plain, "1p343")
