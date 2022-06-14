# REAL TIME HUMAN DETECTION

- In this project the model with the input either as a live feed from any camera or a video file ,can detect the number of people at that instant as shown in the below video
- This is an end to end project , python's Streamlit framework is used.
- The model works with YOLOV4 api.

https://user-images.githubusercontent.com/82393353/173562045-5b2c7d60-2f85-455c-b982-b21e5e945864.mp4

## Imporovemnts 
- > To make the project run faster, to improve frame processing rate I wanna use python queue data structure.The process of Reading of frames from the camera is done through sepearate thread

- > As an Improvement, the end users can give input as pictures of a person and can detect this person in the live feed or video(finding lost people like children and thief detection). The model will be trained on these images to detect the person.
## Deployment 
```diff
- Deployed this project on HuggingFace Platform and can be accessed by any one with the link given below , 
- but unfortunately this project needs considerable computation power, this is  running on the free plan of 
- huggingFace platform, the camera won't even open and the detection for video files is very very slow,
- but still if you wanna take a look, use the link below
```
- link - https://huggingface.co/spaces/SSahas/Pedestrian_detection
