# myPHOSCWork
visualization of results from phoscnet

yoloToPhoscVIsualization.ipynb
    THIS SCRIPT PERFORM PHOSCPREDICTION FROM IMAGE CROPS

    1. THE IMAGE CROPS CAN BE FROM A YOLOV3 PREDICTION SCRIPT 
    2. THE IMAGE CROPS CAN BE FROM A ORIGINAL BB FROM LOCALIZATION DATA
    3. THIS SCRIPT PROJECT BACK THE RESULT ON ORIGINAL IMAGE WHERE WE
    CAN SEE CORRECT AND WRONG PREDICTION.
    4. IMAGE CROPS HAS SPECIFIC NAME FORMAT


yoloToPhoscPipeLine2.ipynb
THIS SCRIPT USES YOLO3 TO LOCALISE THE WORDS AND EXTRACTS CROPS FROM THE ORIGINAL IMAGE.

createDataYoloOfficialSplit.ipynb:
 machine: laptop this script creates yolo training data with IAM handwritten official split Aim is to train yolov3 with this split and use the same split for training PHOSCNET

So the data is created in 2 ways from IAM official split.

1. Line level annotation is taken and only lines which belongs to training data are
considered. Here training data file contains train as well as test file.

Foloowing are the files created.
"mnist_train_PhoscIamOfficialSplitForYoloLineLevel1.txt"
"mnist_test_PhoscIamOfficialSplitForYoloLineLevel1.txt"


2.All Lines which are present on training data page are taken, So in this case few lines
can not be present in as IAM official split but will be considered for YOLOV# localization.

