import cv2
from os import listdir
from sklearn.metrics import classification_report 
import os
M_COEFF = 0.75

#PART 1
#create 2 dictionaries one for base and one for suspects
dict_base={}
dict_suspects={}

#loop over images, read them, convert to grayscale
for file in listdir('/home/anastasia/Έγγραφα/biometrics/lab2/DB1_B'):
    filename = os.fsdecode(file)
        
    image = cv2.imread('/home/anastasia/Έγγραφα/biometrics/lab2/DB1_B/' + filename)
    print(filename)

    #grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    
    # compute the descriptors with ORB
    kp, orb_des = orb.compute(image, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags = 0)
    #plt.imshow(img2), plt.show()

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints with ORB and compute the descriptors with ORB
    sift_des = sift.detectAndCompute(image, None)

    feat_dict = {'sift': sift_des, 'orb': orb_des}

    img_id = filename.split('.')[0]
    if img_id.split('_')[-1] == '1':
        dict_base[img_id.split('_')[0]] = feat_dict
    else:
        dict_suspects[img_id] = feat_dict
        

#PART 2
#create two matchers (BFMatcher) - one for SIFT (with cv2.NORM_L2) and
#one for ORB (with cv2.NORM_HAMMING);
matcher_sift = cv2.BFMatcher(cv2.NORM_L2)
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING)

j = 1

#for each descriptor, create two lists: y_test (for reference values) and y_pred (for algorithm’s predictions);
test_orb = []
pred_orb = []

test_sift = []
pred_sift = []

#iterate over features from suspects dictionary and try to match them with each
#of base features using knnMatch with k=2 (to get two best matches for each feature);
for des in dict_suspects:
    orb_length = {}
    sift_length = {}

    for val in dict_base:
        orb_good_match = []
        sift_good_match = []
        
        orb_match = matcher_orb.knnMatch(dict_suspects[des]['orb'][1], dict_base[val]['orb'][1], k = 2)
        sift_match = matcher_sift.knnMatch(dict_suspects[des]['sift'][1], dict_base[val]['sift'][1], k = 2)
        
        #print(orb_match)
        
        #use match distance to perform so-called “ratio test” and select only “good matches”
        for m,n in orb_match:   
            if m.distance < M_COEFF * n.distance:
                orb_good_match.append(m)
        
        for m,n in sift_match:   
            if m.distance < M_COEFF * n.distance:
                sift_good_match.append(m)
        
        #save the length of good_matches list for each base fingerprint (we will use it as a “measure of matching”);
        orb_length[val] = len(orb_good_match)
        sift_length[val] = len(sift_good_match)
    
    #find the best (i.e. the one with largest length of good_matches) match
    #throughout matches with fingerprint base
    pred_orb.append(max(orb_length, key = orb_length.get))
    test_orb.append(des.split('_')[0])

    pred_sift.append(max(sift_length, key = sift_length.get))
    test_sift.append(des.split('_')[0])

    j = j + 1

#print the summary using classification_report 
print("--------------------ORB--------------------")
print(classification_report(test_orb, pred_orb))
print("--------------------SIFT--------------------")
print(classification_report(test_sift, pred_sift))