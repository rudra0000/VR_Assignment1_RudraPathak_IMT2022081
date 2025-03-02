import cv2
import numpy as np
import matplotlib.pyplot as plt


showImages = False

def get_image(path1):
    image = cv2.imread(path1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    image = image_rgb
    return image, image_rgb, image_gray


def SIFT(img):
    # siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
    siftDetector= cv2.SIFT_create()  # depends on OpenCV version

    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

left, left_rgb, left_gray = get_image('./images_part2/first.jpeg')

right, right_rgb, right_gray = get_image('./images_part2/second.jpeg')


kp_left, des_left = SIFT(left_gray)
left_draw = left.copy()
cv2.drawKeypoints(left_gray, kp_left, left_draw, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# print(len(kp_left))

plt.imshow(left_draw)
plt.title('first keypoints')
plt.axis('off')
plt.savefig('./images_part2/first_keypoints.jpeg')
if showImages:
    plt.show()
    plt.close()





kp_right, des_right = SIFT(right_gray)
right_draw = right.copy()
cv2.drawKeypoints(right_gray, kp_right, right_draw, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(right_draw)
plt.title('second keypoints')
plt.axis('off')
plt.savefig('./images_part2/second_keypoints.jpeg')
if showImages:
    plt.show()
    plt.close()



# using a brute force matcher
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des_left,des_right)


# sort the matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

print('number of matches', len(matches))
print('the first match', matches[0].queryIdx, matches[0].trainIdx)

# Draw first 50 matches.
out = cv2.drawMatches(left_rgb, kp_left, right_rgb, kp_right, matches[:50], None, flags=2)

plt.imshow(out)
plt.axis('off')
plt.title('first second matches')
plt.savefig('./images_part2/matches.jpeg')
if showImages:
    plt.show()
    plt.close()


src_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # these are coordinates of points in the original plane
dst_pts = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) # these are coordinates of points in the target plane

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


height, width, channels = right.shape
img2_reg = cv2.warpPerspective(left, H, (width, height)) # transforms left into the coordinate frame of right



print("Homography Matrix (H):")
print(H)


# we'll do this, for all the boundary points of the image, we'll compute the homography 
# and then we'll know how much space we need in order to fit both the images
left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# left image
height_l, width_l, channel_l = left.shape
corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
corners_new = [np.dot(H, corner) for corner in corners]
corners_new = np.array(corners_new).T # take the transpose
x_news = corners_new[0] / corners_new[2]
y_news = corners_new[1] / corners_new[2]
y_min = min(y_news)
x_min = min(x_news)


translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
H = np.dot(translation_mat, H) # so that some parts of the image don't go out of bounds
# some parts might still be out of bounds, but we'll try to minimize it...


height_new = int(round(abs(y_min) + height_l))
width_new = int(round(abs(x_min) + width_l))
size = (width_new, height_new)


warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
black = np.zeros(3)  # Black pixel.

# Stitching procedure, store results in warped_l.
# Loop over each pixel in the right warped image
for i in range(warped_r.shape[0]):
    for j in range(warped_r.shape[1]):
        pixel_l = warped_l[i, j, :]
        pixel_r = warped_r[i, j, :]
        
        if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
            warped_l[i, j, :] = pixel_l
        elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
            warped_l[i, j, :] = pixel_r
        elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
            warped_l[i, j, :] = (pixel_l + pixel_r) / 2
        else:
            # warped_l[i, j, :] = [0, 0, 255] 
            # warped_r[i, j, :] = [0, 0, 255]
            pass


# Stitch the result by slicing warped_l to match the size of warped_r
stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]


plt.imshow(stitch_image)
plt.title('final output')
plt.axis('off')
plt.savefig('./images_part2/output.jpeg')
if showImages:
    plt.show()
    plt.close()

# plt.imshow(warped_l)
# plt.title('Warped l')
# plt.show()
# plt.close()



