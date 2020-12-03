Detecting Text Areas on mixed Digital Documents.
Using Python's OpenCV, Haar features,Morhological Features,Image Integral,SOM Neural Network.
First, an image is uploaded and preprocessed. Later Haar like features and Morpological features are applied on the preprocessed image using Image Integral.This will by far reduce the time of execution of the programm. The values of the features applied are saved and passed on to the pretrained SOM Neural Network. The SOM Model is consisting of a 4x4 topology, where some neurons correspond to text areas.
When training the SOM Network, documents, published on scientific magazines (IEEE) which contain both text and graphic such as images,diagramms,tables etc. are taken into consideration. 
