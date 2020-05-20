#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2 as cv
import numpy as np
 
#像素相加
def pixel_add(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("pixel_add", dst)
    
#像素相减
def pixel_subtract(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("pixel_subtract", dst)
    
#像素相除
def pixel_divide(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("pixel_divide", dst)
    
#像素相乘法
def pixel_multiply(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("pixel_multiply", dst)