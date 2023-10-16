# Paddle_OCR
This is the program in which when given images of specification boards ex: planks mentioning rated voltage, rated current, resistance etc., would give an excel containing values of each parameter mapped to its correct exact label.  
<br>
After creating bounding boxes around each word in the figure. We have used three different methods to achieve mapping of values to correct labels.  
<br>
a) Sorting based on co-ordinates of bounding boxes i.e., Threshold based yx sorting algorithm. (Text_identification_and_semantic_recognition-2).  
<br>
b) Creating the mapping using Null Max Suppression (NMS) algorithm. (Text_identification_and_semantic_recognition-1).  
<br>
c) Using Semantic recognition using transformers in Paddle OCR mixed with rule based sorting. (Text_identification_and_semantic_recognition-2).  
