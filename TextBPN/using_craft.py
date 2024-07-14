import os
import natsort
import pickle
import numpy as np
import glob

def get_file_list(file_path):
    
    file_list = os.listdir(file_path)
    file_list_txt = [file for file in file_list if file.endswith(".txt")]
    filepath_txt = natsort.natsorted(file_list_txt)

    txt_list = []
    # print(filepath_txt,'\n\n\n')
    for filename in filepath_txt:
        
        txt_comp = []
        full_filename = os.path.join(file_path,filename)
        f = open(full_filename,"r")
        data = f.read().splitlines() 
        # ['119,200,222...','234,553,...']
            
        for comp in range(len(data)):
            
            txt_comp.append(data[comp].split(",")) 
            # data.split =

            
            
        txt_list.append(txt_comp)
    
    for i in range(len(txt_list)):
        for j in range(len(txt_list[i])):
            txt_list[i][j] = list(map(int,txt_list[i][j]))

    return txt_list


def linear_eq(x1,y1,x2,y2):
    m = (y2-y1)/(x2-x1)
    n = y1 - (m*x1)
    return m,n

def horizantal_point(x1,y1,x2,y2,poly_list):
    if x1 != x2:
        m,n = linear_eq(x1,y1,x2,y2)
        pt_x1 = (int(np.random.uniform(x1,x2)))
        pt_y1 = (m*pt_x1) + n
        pt_x2 = (int(np.random.uniform(pt_x1,x2)))
        pt_y2 = (m*pt_x2) + n 
        pt_x3 = (int(np.random.uniform(pt_x2,x2)))
        pt_y3 = (m*pt_x3) + n 
    else:
        pt_x1 = x1
        pt_y1 = int(np.random.uniform(y1,y2))
        pt_x2 = x1
        pt_y2 = int(np.random.uniform(pt_y1,y2))
        pt_x3 = x1
        pt_y3 = int(np.random.uniform(pt_y2,y2))
    
    
    return poly_list.append([x1,y1]),poly_list.append([pt_x1,pt_y1]),poly_list.append([pt_x2,pt_y2]),poly_list.append([pt_x3,pt_y3]),poly_list.append([x2,y2])
    


def vetex_point(x1,y1,x2,y2,poly_list):
    if x1 != x2:
        m,n = linear_eq(x1,y1,x2,y2)
        pt_x1 = (int(np.random.uniform(x1,x2)))
        pt_y1 = (m*pt_x1) + n
        pt_x2 = (int(np.random.uniform(pt_x1,x2)))
        pt_y2 = (m*pt_x2) + n
    else:
        pt_x1 = x1
        pt_y1 = int(np.random.uniform(y1,y2))
        pt_x2 = x1
        pt_y2 = int(np.random.uniform(pt_y1,y2))
    
    
    return poly_list.append([pt_x1,pt_y1]),poly_list.append([pt_x2,pt_y2])
    

def get14p(filepath):
    txt_list = get_file_list(filepath)
    poly1 = []
    for i in range(len(txt_list)):
        poly2 = []
        for j in range(len(txt_list[i])):
            poly3 = []
            # array1 = np.zeros((14,2))

            if len(txt_list[i][j]) != 28:
                tl_x = txt_list[i][j][0] # top_left
                tl_y = txt_list[i][j][1]
                tr_x = txt_list[i][j][2] # top_right
                tr_y = txt_list[i][j][3]
                br_x = txt_list[i][j][4] # bottom right
                br_y = txt_list[i][j][5]
                bl_x = txt_list[i][j][6] # bottom left
                bl_y = txt_list[i][j][7]
                origin = [tl_x,tl_y]

                
                horizantal_point(tl_x,tl_y,tr_x,tr_y,poly3) # top 5 point
                vetex_point(tr_x,tr_y,br_x,br_y,poly3) # right 2 point
                horizantal_point(br_x,br_y,bl_x,bl_y,poly3) # bottom 5 point
                vetex_point(bl_x,bl_y,tl_x,tl_y,poly3)
                

                
                # poly4 = np.array(poly3).tolist()
                # poly4 = sorted(poly4,key=clockwiseangle_and_distance)
                #poly3[:5] = sorted[poly3[:5],axis=0]
                poly2.append(np.array(poly3).tolist())
                
                
            
            else:
                poly2.append((np.array(txt_list[i][j]).reshape(14,2)).tolist())
                

        poly1.append(poly2)
    #print("len poly2:",len(poly2))
    #print("len poly1:",len(poly1))
    #print(poly1[:3])

    with open("inference.pkl","wb") as f:
        pickle.dump(poly1,f)
    return poly1

def train_get14p(filepath):
    txt_list = get_file_list(filepath)
    poly1 = []
    for i in range(len(txt_list)):
        poly2 = []
        for j in range(len(txt_list[i])):
            poly3 = []
            # array1 = np.zeros((14,2))

            if len(txt_list[i][j]) != 28:
                tl_x = txt_list[i][j][0] # top_left
                tl_y = txt_list[i][j][1]
                tr_x = txt_list[i][j][2] # top_right
                tr_y = txt_list[i][j][3]
                br_x = txt_list[i][j][4] # bottom right
                br_y = txt_list[i][j][5]
                bl_x = txt_list[i][j][6] # bottom left
                bl_y = txt_list[i][j][7]
                origin = [tl_x,tl_y]

                
                horizantal_point(tl_x,tl_y,tr_x,tr_y,poly3) # top 5 point
                vetex_point(tr_x,tr_y,br_x,br_y,poly3) # right 2 point
                horizantal_point(br_x,br_y,bl_x,bl_y,poly3) # bottom 5 point
                vetex_point(bl_x,bl_y,tl_x,tl_y,poly3)
                

                
                # poly4 = np.array(poly3).tolist()
                # poly4 = sorted(poly4,key=clockwiseangle_and_distance)
                #poly3[:5] = sorted[poly3[:5],axis=0]
                poly2.append(np.array(poly3).tolist())
                
                
            
            else:
                poly2.append((np.array(txt_list[i][j]).reshape(14,2)).tolist())
                

        poly1.append(poly2)
    #print("len poly2:",len(poly2))
    #print("len poly1:",len(poly1))
    #print(poly1[:3])

    with open("./craft_ctw_train.pkl","wb") as f:
        pickle.dump(poly1,f)
    return poly1
