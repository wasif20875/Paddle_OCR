from pdf2image import convert_from_path
import torchvision
import torch
import zipfile
import os
import pandas as pd
import numpy as np
import streamlit as st
from wand.image import Image
from wand.display import display
import cv2
import time
from paddleocr import PaddleOCR,draw_ocr,PPStructure,save_structure_res

def nms(l):
    image_path=l
    image_cv=cv2.imread(image_path)
    image_height=image_cv.shape[0]
    image_width=image_cv.shape[1]
    # output=ocr.ocr(image_path)
    output=table_engine(image_cv, return_ocr_result_in_table=True)
    print(output)#output[0]['res']['cell_bbox'][0] is first box, boxes[0](one box) has cv2 type rectangel two coordinates rec_rex is a list of sets txt and probability
    boxes=[line for line in output[0]['res']['boxes']]
    texts=[line[0] for line in output[0]['res']['rec_res']]
    probabilities=[line[1] for line in output[0]['res']['rec_res']]
    image_boxes=image_cv.copy()
    for box in boxes:
        print('@@##@@##',box[0])
        cv2.rectangle(image_boxes,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),1)
    cv2.imwrite('detections.jpg',image_boxes)
    horiz_boxes=[]
    vert_boxes=[]
    im=image_cv.copy()
    for box in boxes:
        x_h,x_v=0,int(box[0])
        y_h,y_v=int(box[1]),0

        w_h,w_v=image_width,int(box[2]-box[0])
        h_h,h_v=int(box[3]-box[1]),image_height

        horiz_boxes.append([x_h,y_h,x_h+w_h,y_h+h_h])
        vert_boxes.append([x_v,y_v,x_v+w_v,y_v+h_v])
        print(x_h,y_h,x_h+w_h,y_h+h_h)
    #     cv2.rectangle(im,(x_h,y_h),(x_h+w_h,y_h+h_h),(255,0,0),1)
    #     cv2.rectangle(im,(x_v,y_v),(x_v+w_v,y_v+h_v),(0,255,0),1)
    # cv2.imwrite('detections1.jpg',im)

    horiz_out=torchvision.ops.nms(torch.Tensor(horiz_boxes),torch.Tensor(probabilities),0.1) 
    horiz_lines=np.sort(np.array(horiz_out))
    print(horiz_lines)

    im_nms=image_cv.copy()
    for val in horiz_lines:
        cv2.rectangle(im_nms,(horiz_boxes[val][0],horiz_boxes[val][1]),(horiz_boxes[val][2],horiz_boxes[val][3]),(255,0,0),1)
    

    vert_out=torchvision.ops.nms(torch.Tensor(vert_boxes),torch.Tensor(probabilities),0.1) 
    vert_lines=np.sort(np.array(vert_out))
    print(vert_lines)
    for val in vert_lines:
        cv2.rectangle(im_nms,(vert_boxes[val][0],vert_boxes[val][1]),(vert_boxes[val][2],vert_boxes[val][3]),(255,0,0),1)
    cv2.imwrite('detections2.jpg',im_nms)
    linelst=[]
    filenamelst=[]
    heightofbox=[]
    # for h in result:#iterating over lines #every word (if we have words in each line)
    #     print(h)
    #     print(asdf)
    #     heightofbox.append(abs(h[0][0][1]-h[0][2][1]))
    #     heightofbox.append(abs(h[0][1][1]-h[0][3][1]))
    # heightofbox=np.array(heightofbox)
    # himgdic1[imgcount]=np.average(heightofbox)

    out_array=[["" for i in range(len(horiz_lines))] for j in range(len(horiz_lines))]

    unordered_boxes=[]
    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])

    ordered_boxes=np.argsort(unordered_boxes)
    
    def intersection(box1,box2):
        return [box2[0],box1[1],box2[2],box1[3]]

    def iou(box1,box2):
        x_1=max(box1[0],box2[0])
        y_1=max(box1[1],box2[1])
        x_2=min(box1[2],box2[2])
        y_2=min(box1[3],box2[3])
        inter=abs(max((x_2-x_1,0))*max((y_2-y_1),0))
        if inter==0:
            return 0
        box_1_area=abs((box1[2]-box1[0])*(box1[3]-box1[1]))
        box_2_area=abs((box2[2]-box2[0])*(box2[3]-box2[1]))
        return inter/float(box_1_area+box_2_area-inter)
    clone_tracker=[]
    for z in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant=intersection(horiz_boxes[horiz_lines[z]],vert_boxes[vert_lines[ordered_boxes[j]]])
            kg=0
            for b in range(len(boxes)):
                the_box=[boxes[b][0],boxes[b][1],boxes[b][2],boxes[b][3]]
                if (iou(resultant,the_box)>kg):
                    kg=iou(resultant,the_box)
                    kg1=b
            if kg1 in clone_tracker:
                continue
            out_array[z][j]=texts[kg1]
            clone_tracker.append(kg1)
    return(out_array)

if __name__=="__main__":
    st.set_page_config(page_title="image extraction",layout="wide")
    col=st.columns([0.2,0.6,0.2])
    sst=st.session_state

    if 'buttonmemory1' not in st.session_state:
        sst.buttonmemory1=False
        sst.lang=''
        sst.zippath=''

    with col[1].form("sub_form"):
        sst.zippath=st.text_input("Input the address of zip file")
        sst.lang=st.selectbox('Language present in the files',['English','German'])
        sst.button1=st.form_submit_button("Submit")

    if sst.button1:
        sst.buttonmemory1=True
    if not sst.buttonmemory1:
        st.markdown("Please submit the form")
        st.stop()
    
    path=r"C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\tempfiles"
    langdic={'English':'en','German':'gr'}
    # reader=easyocr.Reader([langdic[sst.lang]])
    dic={}
    dic1={}
    himgdic={}
    himgdic1={}

    with zipfile.ZipFile(sst.zippath,'r') as z:
        z.extractall(path)
    dicexl={}
    imgcount=0
    # creating images and extracting from images provided
    for fn in os.listdir(path):
        if fn.endswith('.pdf'):
            images=convert_from_path(os.path.join(path,fn))
            os.makedirs(path+f'\{str(fn)[:-4]}',exist_ok=True)
            for j in range(len(images)):
                images[j].save(path + f"\{str(fn)[:-4]}\{j}.jpg",'JPEG')
        elif fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.PNG') or fn.endswith('.JPG') or fn.endswith('.JPEG') or fn.endswith('.jpeg'): #iterating over images
            imgcount+=1
            l=os.path.join(path,fn)
            #deskewing to be done here
            dt=1
            while(dt<=1):
                with Image(filename=l) as img:
                    img.deskew(0.4*img.quantum_range)
                    img.save(filename=os.path.join(path,fn))
                    dt+=1
            table_engine=PPStructure(layout=False,show_log=True)
            ocr=PaddleOCR(use_angle_cls=True,lang='en')
            out_array=nms(l)
            #os.system('notepad')
            #print(f"cd /d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text\\padgit\\PaddleOCR& python .//tools//infer_kie_token_ser_re.py -c configs//kie//vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=.//pretrained_model//re_vi_layoutxlm_xfund_pretrained//best_accuracy Global.infer_img={'../../tempfiles/'+fn} -c_ser configs//kie//vi_layoutxlm//ser_vi_layoutxlm_xfund_zh.yml -o_ser Architecture.Backbone.checkpoints=.//pretrained_model//ser_vi_layoutxlm_xfund_pretrained//best_accuracy& cd //d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text")
            #os.system(f"cd /d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text\\padgit\\PaddleOCR& python .//tools//infer_kie_token_ser_re.py -c configs//kie//vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=.//pretrained_model//re_vi_layoutxlm_xfund_pretrained//best_accuracy Global.infer_img={'../../tempfiles/'+fn} -c_ser configs//kie//vi_layoutxlm//ser_vi_layoutxlm_xfund_zh.yml -o_ser Architecture.Backbone.checkpoints=.//pretrained_model//ser_vi_layoutxlm_xfund_pretrained//best_accuracy& cd //d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text")
            os.system(f"cd /d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text\\padgit\\PaddleOCR& python .//tools//infer_kie_token_ser.py -c configs//kie//vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=.//pretrained_model//ser_vi_layoutxlm_xfund_pretrained//best_accuracy Global.infer_img={'../../tempfiles/'+fn}& cd //d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text")
            # os.system(r'cd /d C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\padgit\PaddleOCR')
            # os.system(f'python ./tools/infer_kie_token_ser_re.py -c configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./pretrained_model/re_vi_layoutxlm_xfund_pretrained/best_accuracy Global.infer_img={fn} -c_ser configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o_ser Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy')
            # os.system(r'cd /d C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text>streamlit run zipext1.py')
            while True:
                try:
                    print('@@',imgcount)
                    rehand=open(r'C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\padgit\PaddleOCR\output\re\xfund_zh\res.txt\infer_results.txt','r')
                    break
                except IOError:
                    time.sleep(1)
            refile=rehand.read()
            # print('@@\n',refile)
            dicexl[fn]=pd.DataFrame(out_array)
            
    with pd.ExcelWriter('nmpconglomerate.xlsx') as writer:
        print(writer.engine)
        for k in dicexl.keys():
            dicexl[k].to_excel(writer,sheet_name=k)


    #extracting text from images in files
    # for i in os.listdir(path):
    #     if not os.path.isfile(os.path.join(path,i)): # if element in the list is a directory i.e., not a file
    #         j=os.path.join(path,i)
    #         result=[]
    #         linelst=[]
    #         filenamelst=[]
    #         filenamedic={}
    #         himgdic={}
    #         count=0
    #         for xxx,k in enumerate(os.listdir(j)):#iterating over images
    #             count+=1
    #             if count<=1:
    #                 l=os.path.join(j,k)
    #                 print(i)
    #                 print(j)
    #                 print(k)
    #                 print(l)
    #                 ddl=reader.readtext(l)
    #                 result.append(ddl)
    #                 filenamedic[xxx]=os.path.join(i,k)
    #         #result has all images in a pdf
    #         for x,m in enumerate(result):#iterating over image
    #             heightofbox=[]
    #             for n in m:#iterating over lines #every word (if we have words in each line)
    #                 heightofbox.append(abs(n[0][0][1]-n[0][2][1]))
    #                 heightofbox.append(abs(n[0][1][1]-n[0][3][1]))
    #             himgdic[x]=np.average(heightofbox)

    #             count1=0
    #             for xx,n in enumerate(m):#iterating over words to add to line #one image
    #                 if xx==0:
    #                     string=m[xx][1]
    #                     continue
    #                 print('@@@@@@@@@',m[xx])
    #                 print(himgdic[x])
    #                 if abs(m[xx-1][0][0][1]-m[xx][0][0][1])<himgdic[x]:
    #                     string=string+" "+str(n[1])
    #                 else:
    #                     if xx==len(result)-1:
    #                         linelst.append(string)
    #                         linelst.append(m[xx][1])
    #                         count1+=2
    #                         break
    #                     linelst.append(string)
    #                     string=m[xx][1]
    #                     count1+=1
    #                 listofimgname=[filenamedic[x]]
    #                 filenamelst=listofimgname*count1
    #             #     '''if xx==0:
    #             #         continue
    #             #     if abs(m[xx-1][0][1]-m[xx][0][1])<himgdic[x]+10:
    #             #         linelst.append(n[1])
    #             #         count1+=1
    #             # filenamelst.append(list(filenamedic[x])*count1)
    #             # filenamelst=[i for j in filenamelst for i in j]'''
    #         print('@@@@@@@@@',linelst)
    #         print('@@@@',filenamelst)
    #         dic1['lines']=linelst
    #         dic1['filename']=filenamelst
    #         dic[i]=pd.DataFrame(dic1)
    #         print(result)

    # with pd.ExcelWriter('imgextxl.xlsx') as writer:
    #     print(writer.engine)
    #     for k in dic.keys():
    #         dic[k].to_excel(writer,sheet_name=k)
                

