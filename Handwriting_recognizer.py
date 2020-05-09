#Importing packages
from keras.models import load_model
from collections import deque
import numpy as np
import cv2


#########################################################################################################################################################

# Load the MLP model built
mlp_model = load_model('emnist_mlp_model.h5')

# Letters lookup
letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# Define the upper and lower boundaries for a color to be considered "Green"
greenLower = np.array([36,25,25])
greenUpper = np.array([70,255,255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Define Black Board
blackboard = np.zeros((480,640,3), dtype=np.uint8)

# Setup deques to store alphabet drawn on screen
    #Not more than 512 points/pixels you will store at a time
points = deque(maxlen=512)

# Define prediction variables with some random values
prediction1 = 26
prediction2 = 26


#########################################################################################################################################################

#CAPTURING VIDEO
    # Create videocapture object
    #0-> capture video from webcam
camera = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #20.0 is frame rate
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# Keep looping to grab frames from video
while(camera.isOpened()):
    # Grab the frame
    ret, frame = camera.read()
    #Setting frame size
        # 3 -> Width ; 4 -> Height
    camera.set(3,640)
    camera.set(4,480)
    if ret == True:
        #Flips image horizontally
        frame = cv2.flip(frame, 1)
        #converting frame image from BGR into HSV representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ##Check to see if we have reached the end of the video 
        ##(useful when input is a video file not a live video stream)
        #if not grabbed:
        #   break

        # Determine which pixels fall within the green boundaries
        greenMask = cv2.inRange(hsv, greenLower, greenUpper)

        #Smoothing the image identified
            #Erode-> decreases thickness of image (removes white noises)
            #dilate-> Increases thickness of image
            #morphologyEx(MORPH_OPEN)-> Keeps only the outline of the object

            #While performing noise removal, erosion is followed by dilation to make it larger in size so that easy to process
        greenMask = cv2.erode(greenMask, kernel, iterations=2)
        greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel)
        greenMask = cv2.dilate(greenMask, kernel, iterations=1)

        # FINDING CONTOURS(GREEN BOTTLE CAP) 
            #RETR_EXTRENAL-> considers only exterior/eldest contours and ignores nested contours
            #CHAIN_APPROX_SIMPLE-> representing contours by only end points of each line in contour (reduces memory needed)
        (_, cnts, _) = cv2.findContours(greenMask.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            # Sorting contours descendingly based on area
                #Based on the fact we are using bottle cap, it is assumed to have larger area compared to other objects
                #Taking the largest contour and assigning it to cnt variable
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            # Get the radius and center of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Draw the circle in the frame around the found contour , set the colour and thickness
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv2.moments(cnt)
            #Formula to calculate center based on moments
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            #Appending the found center point into deque at the start
            points.appendleft(center)


#########################################################################################################################################################

        # DISPLAYING THE WRITING IN FRAME
        # Connect the points with a line
            for i in range(1, len(points)):
                    #If value is NONE(zero), means no need to draw the line
                    if points[i - 1] is None or points[i] is None:
                            continue
                    # Drawing the line in frame by tracking the center of the object
                    cv2.line(frame, points[i - 1], points[i], (0, 204, 204), 2)
                    # Blackboard is for what model should see, so the color, line-thickness are different
                    cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

        #SCRAPING THE WRITING AND PASSING IT TO MODEL
        # len(cnts) is 0 when we stop writing but points will not be empty
        elif len(cnts) == 0:
          if len(points) != 0:
            #Converting the blackboard image into grayscale
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)

            #Blurring is done to smoothen the edges and reduce edges
                #Gaussian and median blur - reduce image noise
                #median blur-> places median value in center
            blur1 = cv2.medianBlur(blackboard_gray, 15)
                #size of kernel and sigma(SD of X and Y) are specified
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)

            #converts grayscale images into binary imagess
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Finding contours on the blackboard
                #RETR_TREE-> It retrieves all the contours and creates a full family hierarchy list
                #CHAIN_APPROX_NONE-> representing contours using all points of the contour
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            #If atleast one contour is found, arrange them in descending order and take the largest contour
            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]
                if cv2.contourArea(cnt) > 1000:
                    #Drawing a rectangle around the contour and getting endpoints
                    x, y, w, h = cv2.boundingRect(cnt)
                    #Cropping the contour with some extra space
                    alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                    #Changing it to 28x28 to fit MLP model requirements
                    newImage = cv2.resize(alphabet, (28, 28))
                    #Changing into a numpy array
                    newImage = np.array(newImage)
                    #Converting into binary to fit in same format of MLP model
                    newImage = newImage.astype('float32')/255

                    #Changing the shape of numpy to fit into MLP model
                    prediction1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                    #Getting the predictions-> returns the indices of largest element(class)
                    prediction1 = np.argmax(prediction1)

                # Return the deque and blackboard to original status to accept next input
                points = deque(maxlen=512)
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

            
        #Dispaly preciction as string in the frame
        #Specifying coordinates, font style, font scale, color and thickness
        cv2.putText(frame, "Multilayer Perceptron : " +
                    str(letters[int(prediction1)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 215, 0), 2)

        #Write the frame into output file
        out.write(frame)

        #Show the frame
        cv2.imshow("alphabets Recognition Real Time", frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
