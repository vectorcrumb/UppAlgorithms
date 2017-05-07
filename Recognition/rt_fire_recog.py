import cv2


# Helper function to quickly show images
def show_img(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)

def onChange(arg):
    pass

# Given an image, returns fire filtered image
def fire_filter(img):
    r2_r_threshold = 190
    r2_g_threshold = 100
    r2_b_threshold = 140
    r6_threshold = 70
    r7_cb_threshold = 120
    r7_cr_threshold = 150
    # Start timer and split image
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
    # Return mask
    return filtered_img, mask

def smoke_filter(img, prev_img):
    r1_rgb_threshold = 20
    r2a_s_threshold = 0.1 * 255
    # r2b_s_threshold = 10
    r2b_s_threshold = cv2.getTrackbarPos("smoke", trackbar_win)
    # Rule 1 - RGB absdiffs less than threshold
    b_img, g_img, r_img = cv2.split(img)
    r1_rg_diff = cv2.absdiff(r_img, g_img)
    r1_gb_diff = cv2.absdiff(g_img, b_img)
    r1_rb_diff = cv2.absdiff(r_img, b_img)
    r1_rg_mask = cv2.compare(r1_rg_diff, r1_rgb_threshold, cv2.CMP_LE)
    r1_gb_mask = cv2.compare(r1_gb_diff, r1_rgb_threshold, cv2.CMP_LE)
    r1_rb_mask = cv2.compare(r1_rb_diff, r1_rgb_threshold, cv2.CMP_LE)
    r1_temp_mask = cv2.bitwise_and(r1_rg_mask, r1_gb_mask)
    r1_mask = cv2.bitwise_and(r1_temp_mask, r1_rb_mask)
    # Rule 2 - S channel under threshold
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv_img)
    h_prev_img, s_prev_img, v_prev_img = cv2.split(hsv_prev_img)
    # Rule 2a simple measures the S channel under a threshold
    r2a_mask = cv2.compare(s_img, r2a_s_threshold, cv2.CMP_LE)
    # Rule 2b filters gray sky out
    r2b_s_diff = cv2.absdiff(s_img, s_prev_img)
    r2b_mask = cv2.compare(r2b_s_diff, r2b_s_threshold, cv2.CMP_GE)
    r2_mask = cv2.bitwise_and(r2a_mask, r2b_mask)
    mask = cv2.bitwise_and(r1_mask, r2_mask)
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    return filtered_img, mask

cam_index = 0
trackbar_win = "parameters"

cam = cv2.VideoCapture(cam_index)
cv2.namedWindow('original_name', cv2.WINDOW_NORMAL)
cv2.namedWindow('fire_filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('smoke_filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('parameters')

cv2.createTrackbar("smoke", trackbar_win, 0, 255, onChange)

ret, image = cam.read()
prev_image = image

while ret:

    fire_img, fire_mask = fire_filter(image)
    smoke_img, smoke_mask = smoke_filter(image, prev_image)

    cv2.imshow('original_name', image)
    cv2.imshow('fire_filtered', fire_img)
    cv2.imshow('smoke_filtered', smoke_img)

    prev_image = image
    ret, image = cam.read()

    k = cv2.waitKey(20)
    if k == 27:
        break

cv2.destroyAllWindows()
