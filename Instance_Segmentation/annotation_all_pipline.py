import os,datetime
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import random
from pathlib import Path

# RT:RightTop 
# LB:LeftBottom 
# bbox: [xmin, xax, ymin, ymax]
def IOU(bbox_a, bbox_b):
    '''
    W = min(A.RT.x, B.RT.x) - max(A.LB.x, B.LB.x) 
    H = min(A.RT.y, B.RT.y) - max(A.LB.y, B.LB.y) 
    if W <= 0 or H <= 0: 
        return 0 
    SA = (A.RT.x - A.LB.x) * (A.RT.y - A.LB.y) 
    SB = (B.RT.x - B.LB.x) * (B.RT.y - B.LB.y) 
    cross = W * H return cross/(SA + SB - cross)
    '''
    W = min(bbox_a[1], bbox_b[1]) - max(bbox_a[0], bbox_b[0]) 
    H = min(bbox_a[3], bbox_b[3]) - max(bbox_a[2], bbox_b[2]) 
    if W <= 0 or H <= 0: 
        return 0
    SA = (bbox_a[1] - bbox_a[0]) * (bbox_a[3] - bbox_a[2]) 
    SB = (bbox_b[1] - bbox_b[0]) * (bbox_b[3] - bbox_b[2])  
    cross = W * H 
    return cross/(SA + SB - cross)

def get_bbox_from_xmlobj(obj_element):
    xmlbox = obj_element.find('bndbox')
    return [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
            float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]

def get_obj_from_xml(xml_name):
    in_file = open(xml_name)
    tree=ET.parse(in_file)
    root = tree.getroot()
    return [obj for obj in root.iter('object')]

def get_obj_from_image_file(file, bbox):
    img = cv2.imread(file)
    img_obj = img[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])]
   #print(img_obj.shape)
    return img_obj

def get_bboxes_from_etree(etree):
    root = etree.getroot()  
    objects = root.findall('object')
    bboxes = []
    for obj in objects:
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]
        bboxes.append(b)
    return bboxes
def past_to_background_from_image_file(file, bboxes, background_img_array, extend_spaces=0):
    
    img = cv2.imread(file)
    #img = img - 50
    if(img.shape != background_img_array.shape):
       #print('shape not match')
        return
    #print(img.shape)
    #print(img)
    img_objs = []
    for bbox in bboxes:
        img_obj = img[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])]
        img_objs.append(img_obj)
    i = 0
    for bbox in bboxes:
        background_img_array[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = img_objs[i]
        i = i+1
    cv2.imwrite(GENE_IMG_DIR+file.split('/')[-1], background_img_array)
    return 
def generate_new_xmlobj(xmlobj_old, new_position, new_size):
    element_object = ET.Element('object')
    tag_name = ET.SubElement(element_object, 'name')
    tag_name.text = xmlobj_old.find('name').text

    tag_difficult = ET.SubElement(element_object, 'difficult')
    tag_difficult.text = xmlobj_old.find('difficult').text

    element_bndbox = ET.SubElement(element_object, 'bndbox')
    tag_xmin = ET.SubElement(element_bndbox, 'xmin')
    tag_ymin = ET.SubElement(element_bndbox, 'ymin')
    tag_xmax = ET.SubElement(element_bndbox, 'xmax')
    tag_ymax = ET.SubElement(element_bndbox, 'ymax')
    tag_xmin.text = str(new_position[0])
    tag_ymin.text = str(new_position[1])
    tag_xmax.text = str(new_position[0] + new_size[1])
    tag_ymax.text = str(new_position[1] + new_size[0])
    return element_object

def generate_new_xmlobj_simple(xmlobj_old, new_bbox):
    element_object = ET.Element('object')
    tag_name = ET.SubElement(element_object, 'name')
    tag_name.text = xmlobj_old.find('name').text
    tag_difficult = ET.SubElement(element_object, 'difficult')
    tag_difficult.text = xmlobj_old.find('difficult').text
    element_bndbox = ET.SubElement(element_object, 'bndbox')
    tag_xmin = ET.SubElement(element_bndbox, 'xmin')
    tag_ymin = ET.SubElement(element_bndbox, 'ymin')
    tag_xmax = ET.SubElement(element_bndbox, 'xmax')
    tag_ymax = ET.SubElement(element_bndbox, 'ymax')
    tag_xmin.text = str(int(new_bbox[0]))
    tag_ymin.text = str(int(new_bbox[2]))
    tag_xmax.text = str(int(new_bbox[1]))
    tag_ymax.text = str(int(new_bbox[3]))
    return element_object
    
def past_and_insert(img_obj, img_array, new_position, obj_element, etree):
    new_xmlobj = generate_new_xmlobj(obj_element, new_position, img_obj.shape[:2])
    new_xml_etree = insert_to_xml(new_xmlobj, etree)
    new_pil_img = past_obj_to_background(img_obj, img_array, new_position)
    return new_pil_img, new_xml_etree
    

def insert_to_xml(xml_obj, xml_etree):
    root = xml_etree.getroot()
    root.append(xml_obj)
    return xml_etree
    
def past_obj_to_background(img_obj, img_array, position = (200, 200)):
    img = Image.fromarray(img_array)
    img_obj = Image.fromarray(img_obj)
    img.paste(img_obj, position)
    return img
    
def get_cls_from_xmlobj(obj_element):    
    return obj_element.find('name').text

def generate_new_position(img_size, img_obj_size):
    array_x = np.arange(int(img_size[1] - img_obj_size[1]))
    array_y = np.arange(int(img_size[0] - img_obj_size[0]))
    random_x = random.sample(list(array_x), 1)[0]
    random_y = random.sample(list(array_y), 1)[0]
    new_position = (random_x, random_y)
    #print(new_position)
    return new_position

def generate_new_bbox(img_size, img_obj_size):
    array_x = np.arange(int(img_size[1] - img_obj_size[1]))
    array_y = np.arange(int(img_size[0] - img_obj_size[0]))
    random_x = random.sample(list(array_x), 1)[0]
    random_y = random.sample(list(array_y), 1)[0]
    new_position = (random_x, random_y)
    new_bbox = [random_x, random_x + img_obj_size[1], random_y, random_y + img_obj_size[0]]
    #print(new_bbox)
    return new_bbox
def generate_union_bbox(matrix1, matrix2): #xin,xmax,ymin,ymax
    img_obj_shape = (int(max(matrix1[3], matrix2[3])-min(matrix1[2], matrix2[2])), 
                int(max(matrix1[1], matrix2[1])-min(matrix1[0], matrix2[0])))
    return img_obj_shape

def generate_union_bbox_shape(matrix_list): #xin,xmax,ymin,ymax
    matrix_array = np.array(matrix_list)
    img_obj_shape = (int(max(matrix_array[:,3]) - min(matrix_array[:,2])),
                     int(max(matrix_array[:,1]) - min(matrix_array[:,0])))
    return img_obj_shape

def get_union_bbox_left_top(matrix_list): #xin,xmax,ymin,ymax
    matrix_array = np.array(matrix_list)
    left_top = (int(min(matrix_array[:,0])),
                     int(min(matrix_array[:,2])))
    return left_top
def generate_union_bbox_(matrix_list):
    img_obj_shape = generate_union_bbox_shape(matrix_list)
    left_top = get_union_bbox_left_top(matrix_list)
    union_bbox = [int(left_top[0]), 
                  int(left_top[0]+int(img_obj_shape[1])),
                  int(left_top[1]),
                  int(left_top[1]+int(img_obj_shape[0]))]
    return union_bbox

def generate_new_bbox_within_distance(img_size, img_obj_size, old_bbox, scale=2.5):
    array_x = np.arange(int(img_size[1] - img_obj_size[1]))
    #array_y = np.arange(int(img_size[0] - img_obj_size[0])/3, int(img_size[0] - img_obj_size[0])/3*2)
    array_y = np.arange(max(0, int((old_bbox[3]+old_bbox[2])/2 - scale*img_obj_size[0])),
                        min(int((old_bbox[3]+old_bbox[2])/2 + scale*img_obj_size[0]), int(img_size[0] - img_obj_size[0])))
    random_x = random.sample(list(array_x), 1)[0]
    random_y = random.sample(list(array_y), 1)[0]
    new_position = (random_x, random_y)
    new_bbox = [random_x, random_x + img_obj_size[1], random_y, random_y + img_obj_size[0]]
    #print(new_bbox)
    return new_bbox

def check_before_insert(img_size, img_obj_size, etree, union_bbox, scale_x=2, scale_y=1):
    new_bbox = generate_new_bbox_within_distance(img_size, img_obj_size, union_bbox, scale_x, scale_y)
    bboxes = get_bboxes_from_etree(etree)
    retry_times = 0
    while(not check_bbox(new_bbox, bboxes)):
       #print('new_bbox not suitable, retry...')
        retry_times = retry_times + 1
        if(retry_times > 100):
            return (False, False)
        new_bbox = generate_new_bbox_within_distance(img_size, img_obj_size, union_bbox)
    return (new_bbox[0], new_bbox[2])
    
def inset_obj_to_an_image_and_xml(img_obj, img, obj_element, etree):
    new_bbox = generate_new_bbox(img.shape[:2], img_obj.shape[:2])
    bboxes = get_bboxes_from_etree(etree)
    retry_times = 0
    while(not check_bbox(new_bbox, bboxes)):
       #print('new_bbox not suitable, retry...')
        retry_times = retry_times + 1
        if(retry_times > 50):
            return False, False
        new_bbox = generate_new_bbox(img.shape[:2], img_obj.shape[:2])
    #print('new_bbox succussful')
    new_pil_img, new_xml_etree = past_and_insert(img_obj, img, (new_bbox[0], new_bbox[2]), obj_element, etree)
    
    return new_pil_img, new_xml_etree

def do_inset_obj_to_an_image_and_xml(img_obj, img, obj_element, etree, insert_point):
    new_pil_img, new_xml_etree = past_and_insert(img_obj, img, insert_point, obj_element, etree)    
    return new_pil_img, new_xml_etree

def inset_dobule_obj_to_an_image_and_xml(img_obj1, img_obj2, img, obj_element1, obj_element2, etree):
    new_bbox = generate_new_bbox(img.shape[:2], img_obj.shape[:2])
    bboxes = get_bboxes_from_etree(etree)
    retry_times = 0
    while(not check_bbox(new_bbox, bboxes)):
       #print('new_bbox not suitable, retry...')
        retry_times = retry_times + 1
        if(retry_times > 50):
            return False, False
        new_bbox = generate_new_bbox(img.shape[:2], img_obj.shape[:2])
    #print('new_bbox succussful')
    new_pil_img, new_xml_etree = past_and_insert(img_obj, img, (new_bbox[0], new_bbox[2]), obj_element, etree)
    
    return new_pil_img, new_xml_etree

def check_bbox(new_bbox, bboxes):
    for bbox in bboxes:
        if(IOU(new_bbox, bbox) > 0):
            return False
    return True
def caculate_move_vector(bbox, new_point):
    original_xmin, original_ymin = bbox[0], bbox[2]
    diffx, diffy = new_point[0] - original_xmin, new_point[1] - original_ymin
    return (diffx, diffy)

def rotate_img(img, thealta):
    (h_, w_) = img.shape[:2]
    point_list = [(0, 0), (0, h_), (w_, h_), (w_, 0)]
    center = (w_ // 2, h_ // 2)
    roted_point_list = [((point[0]-center[0])*np.cos(np.pi*thealta/180) 
                         - (point[1]-center[1])*np.sin(np.pi*thealta/180) 
                         + center[0], (point[0]-center[0])*np.sin(np.pi*thealta/180) 
                         + (point[1]-center[1])*np.cos(np.pi*thealta/180) + center[1]) for point in point_list]
    #print(roted_point_list)

    temp = np.zeros((2, 4))
    temp[0] = [roted_point[0] for roted_point in roted_point_list]
    temp[1] = [roted_point[1] for roted_point in roted_point_list]
    (xmin, xmax, ymin, ymax) = (np.min(temp[0]), np.max(temp[0]), np.min(temp[1]), np.max(temp[1]))
   #print('xmin: {0}, xmax: {1}, ymin: {2}, ymax: {3}'.format(xmin, xmax, ymin, ymax))

    roted_h, roted_w = ymax - ymin, xmax - xmin
   #print('roted_h: {0}, roted_w: {1}'.format(roted_w, roted_h))
    
    top_bottom, left_right = int((roted_h - h_)/2), int((roted_w - w_)/2)
   #print('top_bottom:{0}, left_right_:{1}'.format(top_bottom, left_right))
    padding = lambda arg : max(arg, 0)
    #dst = cv2.copyMakeBorder(img, padding(top_bottom), padding(top_bottom), padding(left_right), 
    #padding(left_right), cv2.BORDER_CONSTANT)
    dst = cv2.copyMakeBorder(img, padding(top_bottom), padding(top_bottom), padding(left_right), 
                             padding(left_right), cv2.BORDER_CONSTANT)
    
    (h, w) = dst.shape[:2]   
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, thealta, 1.0)
    rotated_img = cv2.warpAffine(dst, M, (w, h))
    if(top_bottom < 0):
        top_bottom = abs(top_bottom)
        rotated_img = rotated_img[top_bottom:-top_bottom, :, :]
    if(left_right < 0):
        left_right = abs(left_right)
        rotated_img = rotated_img[:, left_right:-left_right, :]
    return rotated_img

def get_fliped_bbox(bbox, img_w):
    return (img_w-bbox[1], img_w-bbox[0], bbox[2], bbox[3])

def generate_etree(etree_old):
    root_old = etree_old.getroot()    
    root = ET.Element('annotation')
    root.append(root_old.find('folder'))
    root.append(root_old.find('filename'))
    root.append(root_old.find('source'))
    root.append(root_old.find('size'))
    root.append(root_old.find('segmented'))    
    tree = ET.ElementTree(root)    
    return tree
def get_intercouse_box(bbox_a, bbox_b):
    xmin_ = max(bbox_a[0], bbox_b[0])
    xmax_ = min(bbox_a[1], bbox_b[1])
    ymin_ = max(bbox_a[2], bbox_b[2])
    ymax_ = min(bbox_a[3], bbox_b[3])
    if((xmax_ - xmin_ < 10) or (ymax_ - ymin_ < 10)):
        return False
    return [xmin_, xmax_, ymin_, ymax_]

def generate_new_bbox_within_distance(img_size, img_obj_size, old_bbox, scale_x=2.5, scale_y=2.5):
    #array_x = np.arange(int(img_size[1] - img_obj_size[1]))
    array_x = np.arange(max(0, int((old_bbox[0]+old_bbox[1])/2 - scale_x*img_obj_size[1])),\
    min(int((old_bbox[0]+old_bbox[1])/2 + scale_x*img_obj_size[1]), int(img_size[1] - img_obj_size[1])))
    #array_y = np.arange(int(img_size[0] - img_obj_size[0])/3, int(img_size[0] - img_obj_size[0])/3*2)
    array_y = np.arange(max(0, int((old_bbox[3]+old_bbox[2])/2 - scale_y*img_obj_size[0])),\
    min(int((old_bbox[3]+old_bbox[2])/2 + scale_y*img_obj_size[0]*0), int(img_size[0] - img_obj_size[0])))
    #print('array_x:{0}'.format(list(array_x)))
    #print('array_y:{0}'.format(list(array_y)))
    if(len(list(array_x)) == 0):
        array_x = [0]
    if(len(list(array_y)) == 0):
        array_y = [0]
    
    random_x = random.sample(list(array_x), 1)[0]
    random_y = random.sample(list(array_y), 1)[0]
    new_position = (random_x, random_y)
    new_bbox = [random_x, random_x + img_obj_size[1], random_y, random_y + img_obj_size[0]]
    #print(new_bbox)
    return new_bbox

def xml_reverse_xmin_xmax(etree):
    root = etree.getroot()
    objects = root.findall('object')
    for obj in objects:
        xmlbox = obj.find('bndbox')
        xmlbox.find('xmin').text, xmlbox.find('xmax').text = xmlbox.find('xmax').text, xmlbox.find('xmin').text
    return etree