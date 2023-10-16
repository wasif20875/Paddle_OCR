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
import PIL
from paddleocr import PaddleOCR,draw_ocr,PPStructure,save_structure_res
from deep_translator import GoogleTranslator

#this function is used to find question and answer relationsips
def qanda(smasterstring,refile):
    qstringlist=[]
    keylist=[]
    splitlist=refile.split('\t')
    ewdiclist=eval(splitlist[1])['ocr_info']

    #this function calculates the iou-(inverse over union) values of two boxes given, used to add semantic data to ocr results
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
    for i in range(len(smasterstring)):
        for j in smasterstring[i]:
            kg=0
            for x,k in enumerate(ewdiclist):
                if (iou(j[1],k['bbox'])>kg):
                    kg=iou(j[1],k['bbox'])
                    kg1=x
            #we are appending the semantic info of each text box to smasterstring.
            j.append(ewdiclist[kg1]['pred']) 
    dicans={}
    #creating question  by concatenating all the words before an answer
    for i in range(len(smasterstring)): 
        for j in range(len(smasterstring[i])):
            if smasterstring[i][j][2]=='ANSWER':
                #escaping the word after which there exists an immediate answer
                if j!=(len(smasterstring[i])-1) and smasterstring[i][j+1][2]=='ANSWER': 
                    continue
                k=j
                qstring=''
                switch2=0
                serialanswerkey=0
                serialanswerstring=0
                key=smasterstring[i][j][0]
                while True:
                    k=k-1
                    if k<0:
                        break
                    #if there are continuously answers one after another
                    if smasterstring[i][k][2]=='ANSWER': 
                        switch=0
                        try:
                            int(smasterstring[i][k][0])
                            print('@@##@@##@##',smasterstring[i][k][0])
                            if qstring=='' or serialanswerkey==len(key.split(' ')):
                                key=smasterstring[i][k][0]+' '+key
                                print(key)
                                serialanswerkey=len(key.split(' '))
                                switch=1
                        except Exception as err:
                            if qstring=='' or serialanswerstring==len(qstring.split(' ')):
                                qstring=smasterstring[i][k][0]+' '+qstring
                                serialanswerstring=len(qstring.split(' '))
                                switch=1
                        if switch==1:
                            continue
                        break
                    #if there exists O tag just before the answer and it is continued
                    if smasterstring[i][k][2]=='O': 
                        if smasterstring[i][k+1][2]=='ANSWER':
                            qstring=smasterstring[i][k][0]+' '+qstring
                            serialanswerkey=0
                            memory=smasterstring[i][k][0]
                            switch2=1
                            continue
                        if switch2==1 and smasterstring[i][k+1][0]==memory:
                            qstring=smasterstring[i][k][0]+' '+qstring
                            serialanswerkey=0
                            continue
                        qstring=smasterstring[i][k][0]+' '+qstring
                    #to add text with question tag
                    qstring=smasterstring[i][k][0]+' '+qstring 
                    if k==0:
                        break
                if sst.lang=='English':
                    qstringlist.append(GoogleTranslator(source='auto', target='english').translate(qstring))
                    keylist.append(key)
                elif sst.lang=='German':
                    qstringlist.append(GoogleTranslator(source='auto', target='german').translate(qstring))
                    keylist.append(key)   
                else:
                    qstringlist.append(qstring)
                    keylist.append(key)  
    return (qstringlist,keylist)

#threshold based yx sorting algorthm
def tbyx(l): 
    image_path=l
    image_cv = PIL.Image.open(l).convert('RGB')
    image_cv=PIL.ImageOps.exif_transpose(image_cv)
    table_engine=PPStructure(layout=False,show_log=True)
    #getting ocr results from paddleocr
    output=table_engine(np.array(image_cv), return_ocr_result_in_table=True) 
    print(output)
    boxes=[line for line in output[0]['res']['boxes']]
    texts=[line[0] for line in output[0]['res']['rec_res']]
    probabilities=[line[1] for line in output[0]['res']['rec_res']]
    print(boxes)
    print(texts)
    print(probabilities)
    avglist=[]
    #creating list of heights of all the text bounding boxes
    if len(boxes)>=10:
        for i in range(int(0.25*len(boxes)),int(0.75*len(boxes))):
            avglist.append(abs(int(boxes[i][1])-int(boxes[i][3])))
    else:
        for i in range(len(boxes)):
            avglist.append(abs(int(boxes[i][1])-int(boxes[i][3])))
    yaverage=np.average(avglist)
    ymax=np.max(avglist)
    ymin=np.min(avglist)
    print(ymin)
    print(ymax)
    #taking half of actual average as average as we are comparing upper y coordinates of two boxes which dont require avg height of bounding box to distinguish 
    yaverage=0.5*yaverage 
    print(yaverage)
    masterstring=[]
    string=[]
    #tbyx algorithm logic
    for i in range(len(boxes)):
        if i==0:
            string.append([texts[i],boxes[i]])
            continue
        if abs(boxes[i-1][1]-boxes[i][1])<yaverage:
            string.append([texts[i],boxes[i]])
        elif i==len(boxes)-1 and abs(boxes[i-1][1]-boxes[i][1])<yaverage:
            masterstring.append(string)
            break
        else:
            if i==len(boxes)-1:
                masterstring.append(string)
                masterstring.append([[texts[i],boxes[i]]])
                break
            masterstring.append(string)
            string=[[texts[i],boxes[i]]]
    smasterstring=[]
    #sorting based on x values
    for i in masterstring:
        smasterstring.append(sorted(i,key=lambda x:int(x[1][0]))) 
    #creating an empty matrix to add sorted values from smasterstring
    out_array=[["" for i in range(len(smasterstring[j]))] for j in range(len(smasterstring))]
    print(len(out_array[0]))
    print(len(out_array))
    print('@#$#@',l)
    for i in range(len(smasterstring)):
        for j in range(len(smasterstring[i])):
            if sst.lang=="English": #translating to english
                out_array[i][j]=GoogleTranslator(source='auto', target='english').translate(smasterstring[i][j][0])
            elif sst.lang=="German": #translating to german
                out_array[i][j]=GoogleTranslator(source='auto', target='german').translate(smasterstring[i][j][0])
            else:
                out_array[i][j]=smasterstring[i][j][0]
    return out_array,smasterstring

#function to deskew images and run ocr extraction on images and semantic entity recognition on images
def heart(path,switch1,i):
    dicexl={}
    dicexl1={}
    imgcount=0
    pdfswitch=0
    dirlist=os.listdir(path)#creating a list of all the files in the path
    for fn in dirlist:
        if fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.PNG') or fn.endswith('.JPG') or fn.endswith('.JPEG') or fn.endswith('.jpeg'): #iterating over images
            imgcount+=1
            l=os.path.join(path,fn)
            #deskewing to be done here
            dt=1
            while(dt<=1): #deskewing the images
                with Image(filename=l) as img:
                    img.deskew(0.4*img.quantum_range)
                    img.save(filename=l)
                    dt+=1
            
            ocr=PaddleOCR(use_angle_cls=True,lang='en')
            out_array,smasterstring=tbyx(l) #calling function to create a sorted smasterstring array

            if switch1==0:#executing commands in command prompt to run SER
                os.system(f"cd /d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text\\padgit\\PaddleOCR& python .//tools//infer_kie_token_ser.py -c configs//kie//vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=.//pretrained_model//ser_vi_layoutxlm_xfund_pretrained//best_accuracy Global.infer_img={'../../tempfiles/'+fn}& cd //d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text")
            elif switch1==1:#executing commands if a pdf is given
                os.system(f"cd /d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text\\padgit\\PaddleOCR& python .//tools//infer_kie_token_ser.py -c configs//kie//vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=.//pretrained_model//ser_vi_layoutxlm_xfund_pretrained//best_accuracy Global.infer_img={'../../tempfiles/'+i+'/'+fn}& cd //d C:\\Users\\INMOWAS1\\OneDrive - ABB\\Wasif_ABB_Internship\\Projects\\Image_to_text")
            while True:#reading the ser output file created by paddle ocr
                try:
                    print('@@', fn, imgcount)
                    rehand=open(r'C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\padgit\PaddleOCR\output\ser\xfund_zh\res\infer_results.txt','rb')
                    break
                except IOError:
                    time.sleep(1) #waiting until file is created
            refile=rehand.read()
            refile=refile.decode('utf-8')
            rehand.close()
            os.remove(r'C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\padgit\PaddleOCR\output\ser\xfund_zh\res\infer_results.txt')
            qstringlist=[]
            keylist=[]
            qstringlist,keylist=qanda(smasterstring,refile) #calling function to get question and answer list
            dicqn={}
            dicqn['parameters']=qstringlist
            dicqn['values']=keylist
            dicexl[fn]=pd.DataFrame(out_array)
            dicexl1[fn]=pd.DataFrame(dicqn)
            pdfswitch=1
    #exiting if the images are not parsed
    if pdfswitch==0: 
        return None

    if switch1==0:
        excel1='nmpconglomerate.xlsx'
        excel2='nmpqaconglomerate.xlsx'
    elif switch1==1:
        excel1='nmppdfconglomerate.xlsx'
        excel2='nmppdfqaconglomerate.xlsx'
    #creating excels out of dictionaries created
    with pd.ExcelWriter(excel1) as writer:
        print(writer.engine)
        for k in dicexl.keys():
            dicexl[k].to_excel(writer,sheet_name=k)
    with pd.ExcelWriter(excel2) as writer:
        print(writer.engine)
        for k in dicexl1.keys():
            dicexl1[k].to_excel(writer,sheet_name=k) 
    return None

#this where the program starts
if __name__=="__main__":
    #creating user ui using streamlit webpage for user to input data
    st.set_page_config(page_title="image extraction",layout="wide")
    col=st.columns([0.2,0.6,0.2])
    sst=st.session_state

    if 'buttonmemory1' not in st.session_state:
        sst.buttonmemory1=False
        sst.lang=''
        sst.zippath=''

    with col[1].form("sub_form"):
        sst.zippath=st.text_input("Input the address of zip file")
        sst.lang=st.selectbox('Language to be converted into',['English','German','None'])
        sst.button1=st.form_submit_button("Submit")

    if sst.button1:
        sst.buttonmemory1=True
    if not sst.buttonmemory1:
        st.markdown("Please submit the form")
        st.stop()
    
    path=r"C:\Users\INMOWAS1\OneDrive - ABB\Wasif_ABB_Internship\Projects\Image_to_text\tempfiles"
    langdic={'English':'en','German':'gr'}

    #extracting zip file to a path
    with zipfile.ZipFile(sst.zippath,'r') as z: 
        z.extractall(path)
    #converting each page of pdf to image and storing them in a folder
    for i in os.listdir(path):
        if i.endswith('.pdf'):
            images=convert_from_path(os.path.join(path,i))
            os.makedirs(path+f'\{str(i)[:-4]}',exist_ok=True)
            for j in range(len(images)):
                images[j].save(path + f"\{str(i)[:-4]}\{j}.jpg",'JPEG')

    # creating images and extracting from images provided
    switch1=0
    heart(path,switch1,None) #calling function to analyse and create excel

    #extracting text from images in files
    for i in os.listdir(path):
        if not os.path.isfile(os.path.join(path,i)): # if element in the list is a directory i.e., not a file
            path1=os.path.join(path,i)
            switch1=1
            heart(path1,switch1,i)
            
    
                

