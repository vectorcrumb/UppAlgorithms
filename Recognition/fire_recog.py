import numpy as np
import time
import cv2


# Helper function to quickly show images
def show_img(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)

def fire_filter(img):
    r2_r_threshold = 190
    r2_g_threshold = 100
    r2_b_threshold = 140
    r6_threshold = 70
    r7_cb_threshold = 120
    r7_cr_threshold = 150
    # Start timer and split image
    t0 = time.time()
    b_img, g_img, r_img = cv2.split(img)
    ### Rule 1
    # Compare channels RG and GB and form mask by ANDing
    r1_mask_rg = cv2.compare(r_img, g_img, cv2.CMP_GT)
    r1_mask_gb = cv2.compare(g_img, b_img, cv2.CMP_GT)
    r1_mask = cv2.bitwise_and(r1_mask_rg, r1_mask_gb)

    ### Rule 2
    # Threshold channels by scalars and AND all 3 submasks
    r2_mask_r = cv2.compare(r_img, r2_r_threshold, cv2.CMP_GT)
    r2_mask_g = cv2.compare(g_img, r2_g_threshold, cv2.CMP_GT)
    r2_mask_b = cv2.compare(b_img, r2_b_threshold, cv2.CMP_LT)
    r2_mask_temp = cv2.bitwise_and(r2_mask_r, r2_mask_g)
    r2_mask = cv2.bitwise_and(r2_mask_temp, r2_mask_b)

    ### Rule 3 and 4
    # Obtain YCbCr image
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_img, cr_img, cb_img = cv2.split(ycrcb_img)
    y_mean, cr_mean, cb_mean, _ = cv2.mean(ycrcb_img)
    r3_mask = cv2.compare(y_img, cb_img, cv2.CMP_GE)
    r4_mask = cv2.compare(cr_img, cb_img, cv2.CMP_GE)

    ### Rule 5
    r5_mask_y = cv2.compare(y_img, y_mean, cv2.CMP_GE)
    r5_mask_cr = cv2.compare(cr_img, cr_mean, cv2.CMP_GE)
    r5_mask_cb = cv2.compare(cb_img, cb_mean, cv2.CMP_LE)
    r5_mask_temp = cv2.bitwise_and(r5_mask_y, r5_mask_cr)
    r5_mask = cv2.bitwise_and(r5_mask_temp, r5_mask_cb)

    ### Rule 6
    r6_cbcr_dif = cv2.absdiff(cb_img, cr_img)
    r6_mask = cv2.compare(r6_cbcr_dif, r6_threshold, cv2.CMP_GE)

    ### Rule 7
    r7_mask_cb = cv2.compare(cb_img, r7_cb_threshold, cv2.CMP_LE)
    r7_mask_cr = cv2.compare(cr_img, r7_cr_threshold, cv2.CMP_GE)
    r7_mask = cv2.bitwise_and(r7_mask_cb, r7_mask_cr)

    # Construct 7 step mask
    r12_mask = cv2.bitwise_and(r1_mask, r2_mask)
    r13_mask = cv2.bitwise_and(r12_mask, r3_mask)
    r14_mask = cv2.bitwise_and(r13_mask, r4_mask)
    r15_mask = cv2.bitwise_and(r14_mask, r5_mask)
    r16_mask = cv2.bitwise_and(r15_mask, r6_mask)
    mask = cv2.bitwise_and(r16_mask, r7_mask)
    # Filter image
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    t_exe = time.time() - t0
    # Return mask
    return filtered_img, mask, t_exe

# Constants
img_file = 'fire3.jpg'

# Load image and split it into its BGR channels
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
filtered_img, mask, t_exe = fire_filter(img)

# Show the original image and the filtered image
cv2.namedWindow('fire_img', cv2.WINDOW_NORMAL)
cv2.imshow('fire_img', img)
cv2.namedWindow('filtered_fire', cv2.WINDOW_NORMAL)
cv2.imshow('filtered_fire', filtered_img)
# Show information concerning run time
print("Time of execution:", t_exe*1000, "ms")
print("Image name: {}\nImage dimension: {} x {}\nImage size: ~{}Mpx".format(img_file, img.shape[1], img.shape[0], ((img.shape[0]*img.shape[1])//100000)/10))

# Wait for keypress to exit
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite(img_file.replace('.', '_filtered.'), filtered_img)
    cv2.destroyAllWindows()
