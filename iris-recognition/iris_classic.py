import os
import cv2
import numpy as np
from math import cos, sin, pi

class Circle:
    def __init__(self, a, b, radius):
        self.a = a
        self.b = b
        self.radius = radius

#def circle_iris(a, b, img, value_to_decrease_radius, value_to_decrease_step_seed):
#To further improve the pupil centre we should run the function a second time -
#with the initial location from the previous set, decreased radius (by about 20)
#and smaller step between seed points (fine grain search).
    # initial_radius = 50 - value_to_decrease_radius
    # upper_limit_radius = 200 - value_to_decrease_radius
    
    # initial_angle = 0
    # upper_limit_angle = 360
    
    # step = 5
    # seed_point_steps = 3 - value_to_decrease_step_seed
    
    # best_Circle = Circle(-1, -1, -1)
    # best_bright = -1
    # bright = 0
    
    # radius_range = (initial_radius, upper_limit_radius, step)
    # angle_range = (initial_angle, upper_limit_angle, step)
    
    # for a_dis in range(-5,6):
    #     for b_dis in range(-5,6):
            
    #         a1 = a + seed_point_steps * a_dis
    #         b1 = b + seed_point_steps * b_dis
            
    #         for radius in radius_range:        
    #             for angle in angle_range:
                    
    #                 a2 = radius * cos((angle*pi)/180) + a1
    #                 b2 = radius * sin((angle*pi)/180) + b1                    

            #         bright += img[b2,a2]
                    
            # avg_bright = bright / step
            # bright = 0
            
            # if(best_bright > avg_bright):
            #     best_bright = avg_bright
            #     best_Circle = (a1, b1, radius)
                 
    # return best_Circle
    
def exploding_circle_iris(image, a, b, value_to_decrease_radius, value_to_decrease_step_seed):
#To further improve the pupil centre we should run the function a second time -
#with the initial location from the previous set, decreased radius (by about 20)
#and smaller step between seed points (fine grain search).    
    initial_radius = 50 - value_to_decrease_radius
    upper_limit_radius = 200 - value_to_decrease_radius
    
    initial_angle = 0
    upper_limit_angle = 360
    
    step = 5
    seed_point_steps = 3 - value_to_decrease_step_seed
    
    best_circle = Circle(0,0,0)
    best_brightness = 0
    bright_sum = 0

    for dx in range(-5,6):
        for dy in range(-5,6):
            a1 = a + seed_point_steps * dx
            b1 = b + seed_point_steps * dy

            for radius in range(initial_radius, upper_limit_radius, step):
                for angle in range(initial_angle, upper_limit_angle, step):
                    
                    a2 = int(radius * cos((angle*pi)/180) + a1)
                    b2 = int(radius * sin((angle)*pi/180) + b1)

                    bright_sum += image[b2, a2]
                
                average_brightness = bright_sum / float(step)

                if (best_brightness > average_brightness):
                    best_brightness = average_brightness
                    best_circle = Circle(a1, b1, radius)
                    
                bright_sum = 0    

    return best_circle

def exploiding_half_circles(image, a, b, left):
    initial_radius = 50
    upper_limit_radius = 200
    
    initial_angle = 0
    upper_limit_angle = 360
    
    step = 5
    seed_point_steps = 3
    
    previous_brighness = None
    
    best_circle = Circle(0, 0, 0)
    
    for radius in range(initial_radius, upper_limit_radius, step):
        brightness_sum = 0

    image_copie=image.copy()

    for angle in range(-45+180*left, 45+180*left, step):
        
        x2 = int(radius * cos((angle*pi)/180) + a)
        y2 = int(radius * sin((angle)*pi/180) + b)

        #check that the coordinates are inside the image
        if x2<0 or x2>=image.shape[1] or y2<0 or y2>=image.shape[0]:
            break

        brightness_sum += image[y2,x2]
    
    average_brightness = float(brightness_sum) / ((120)/ float(step))

    if previous_brighness != None:
        if max_diff<(average_brightness-previous_brighness):
            max_diff =(average_brightness-previous_brighness)
            best_circle = Circle(a, b, radius-step)
    
    previous_brighness = average_brightness    
    
    return best_circle


def remove_glare(image):
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(H[150:])
    # plt.show()
    idx = np.argmax(H[150:]) + 151
    binary = cv2.threshold(image, idx, 255, cv2.THRESH_BINARY)[1]

    st3 = np.ones((3, 3), dtype="uint8")
    st7 = np.ones((7, 7), dtype="uint8")

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, st3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, st3, iterations=2)

    im_floodfill = binary.copy()

    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, st7, iterations=1)
    _, _, stats, cents = cv2.connectedComponentsWithStats(im_out)
    cx, cy = 0, 0
    for st, cent in zip(stats, cents):
        if 1500 < st[4] < 3000:
            if 0.9 < st[2] / st[3] < 1.1:
                cx, cy = cent.astype(int)
                r = st[2] // 2
                cv2.circle(image, (cx, cy), r, (125, 125, 125), thickness=2)

    image = np.where(im_out, 80, image)
    image = cv2.medianBlur(image, 5)

    return image, cx, cy


def main(data_path):
    data_path = '/home/anastasia/Έγγραφα/biometrics/lab5/iris_database_train/'
    # Get files from data path
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]

    for filename in filename_list:
        # Read image
        img = cv2.imread(os.path.join(data_path, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        pupil_circle = exploding_circle_iris(img_no_glare, x, y, 0, 0)
        pupil_circle = exploding_circle_iris(img_no_glare, x, y, 20, 2)      
        exploiding_half_circles(img_no_glare, x, y, 1)
        exploiding_half_circles(img_no_glare, x, y, 0)

        #circle_pupil = circle_iris(x, y, img_no_glare, 0, 0)
        #print(circle_pupil)
        #circle_pupil = circle_iris(x, y, img_no_glare, 20, 2)

        # Gabor filters
        # TODO

        cv2.imshow("Original image", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("No glare", img_no_glare)
        key = cv2.waitKey()
        if key == ord("x"):
            break


if __name__ == "__main__":
    data_path = "./iris_database_train"
    main(data_path)
