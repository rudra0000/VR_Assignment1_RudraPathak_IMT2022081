import cv2 
import matplotlib.pyplot as plt
import numpy as np



# def saveImage(title, img, cmap=None, flag=None):
#     plt.axis('off')
#     plt.title(title)
#     if cmap is None:
#         plt.imshow(img, cmap='gray') 
#     else:
#         plt.imshow(img, cmap=cmap)
#     plt.savefig(f'./images_part1/{title}')
#     if flag:
#         plt.show()
#         plt.close()

showImages = False


# path 
path1 = './images_part1/one.jpeg'

# Reading an image in default mode 
image = cv2.imread(path1)

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# the order is supposed to be:
# grayscale
# histogram equalization -- skip as it introduces artifacts in the image!!
# gaussian blur

ksize = (10, 10) 
# gray_image = cv2.equalizeHist(gray_image)
# blur_gray_image = cv2.blur(gray_image, ksize)
blur_gray_image = cv2.GaussianBlur(gray_image , (11,11) , 9)
# blur_gray_image = cv2.equalizeHist(blur_gray_image) 
# blur_gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)



plt.imshow(blur_gray_image, cmap='gray') # we need to exlicitly tell matplotib about the cmap
plt.axis('off')
plt.title('blurred grayscale coins')
if showImages:
    plt.show()
    plt.close()

ret, bin_img = cv2.threshold(blur_gray_image,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

edges = cv2.Canny(bin_img, 50, 250)


plt.axis('off')
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray') 
plt.savefig('./images_part1/a_canny.jpeg')
if showImages:
    plt.show()
    plt.close()



contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# plt.axis('off')
# plt.title('contours')
# plt.imshow(contours)
# plt.show()
# plt.close()

# filter contours
min_area = 250
valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
number_of_coins = len(valid_contours)
print('acc to canny contours, number of coins is', number_of_coins)


# draw the contours on the image
output_image = image.copy()
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 3)  # the last two parameters are color and thickness

plt.axis('off')
plt.title('Contours around Coins')
plt.imshow(output_image, cmap='gray') 
plt.savefig('./images_part1/a_canny_outline.jpeg')
if showImages:
    plt.show()
    plt.close()



# region based segmentation

# now we'll try clustering
img_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels and 3 color values (RGB)
# print('the image shape is', img_convert.shape)
vectorized = img_convert.reshape((-1,3)) # this is 2 * 2
# print('the new shape is', vectorized.shape)
vectorized = np.float32(vectorized)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
# the criteria is to use a change of less than epsilon and a max number of iterations



# we choose k = 2 to separate the foreground from the background
k= 2

img_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # opencv uses bgr but we want rgb
# actually it doesn't matter as long as the features are the same

# Reshape the image into a 2D array of pixels and 3 color values (RGB)
vectorized = image.reshape((-1,3)) # this is 2 * 2
vectorized = np.float32(vectorized)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply KMeans
ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
# print('k means centers are ', center)
result_image = res.reshape((img_convert.shape))


plt.axis('off')
plt.title('K means k=2')
plt.imshow(result_image, cmap='gray') 
plt.savefig('./images_part1/b_kmeans.jpeg')
if showImages:
    plt.show()
    plt.close()



# all the above stuff we do is fine, but we need to get each coin separately for which we use the watershed method
# turns out it is especially useful when the coins are touching as in the image

# Read the image
img = cv2.imread(path1)

# Convert from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)
# gray = cv2.normalize()

ret, bin_img = cv2.threshold(gray,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # all 1s in 3*3
# bin_img = cv2.morphologyEx(bin_img, 
#                            cv2.MORPH_OPEN,
#                            kernel,
#                            iterations=10)

bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_CLOSE,
                           kernel,
                           iterations=3)


sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

plt.axis('off')
plt.title('Sure Background')
plt.imshow(sure_bg, cmap='gray') 
plt.savefig('./images_part1/sure_background.jpeg')
if showImages:
    plt.show()
    plt.close()

dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

plt.axis('off')
plt.title('Distance Transform')
plt.imshow(dist, cmap='gray') 
plt.savefig('./images_part1/dist_transform.jpeg')
if showImages:
    plt.show()
    plt.close()

ret, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 250, cv2.THRESH_BINARY) # 0.45 works for this particular image
sure_fg = sure_fg.astype(np.uint8)  

plt.axis('off')
plt.title('Sure Foreground')
plt.imshow(sure_fg, cmap='gray') 
plt.savefig('./images_part1/sure_foreground.jpeg')
if showImages:
    plt.show()
    plt.close()

unknown = cv2.subtract(sure_bg, sure_fg)

plt.axis('off')
plt.title('Unknown Area')
plt.imshow(unknown, cmap='gray') 
plt.savefig('./images_part1/unknown_area.jpeg')
if showImages:
    plt.show()
    plt.close()


ret, markers = cv2.connectedComponents(sure_fg) # find connected components in foreground
# print('ret is', ret)
# print('type of markers is', type(markers))
# print('shape of markers is', markers.shape)
 
# Add one to all labels so that background is not 0, but 1
markers += 1
# mark the region of unknown with zero
markers[unknown == 255] = 0
 
plt.axis('off')
plt.title('Markers')
plt.imshow(markers, cmap='tab20') 
plt.savefig('./images_part1/markers.jpeg')
if showImages:
    plt.show()
    plt.close()



# Watershed Algorithm
markers = cv2.watershed(img, markers)
# print('markers looks like', markers) 

plt.axis('off')
plt.title('Markers after watershed')
plt.imshow(markers, cmap='tab20') 
plt.savefig('./images_part1/segmented_output.jpeg')
if showImages:
    plt.show()
    plt.close()

 
labels = np.unique(markers)
 
coins = []
f  = open('./images_part1/b_segmented_outputs.txt', 'w')
for label in labels[2:]:  
 
# Create a binary image in which only the area of the label is in the foreground 
#and the rest of the image is in the background   
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    str1 = f"Coin label {label}"
    for i in range(len(target)):
        if i%20 == 0:
            str1 += str(target[i])
    str1 += "-------------------------------------------------------\n"
    
  # Perform contour extraction on the created binary image
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coins.append(contours[0])
f.write(str1)
f.close()


# Draw the outline
img = cv2.drawContours(img, coins, -1, color=(0, 23, 255), thickness=3)

plt.axis('off')
plt.title('Contours after watershed')
plt.imshow(img, cmap='tab20') 
plt.savefig('./images_part1/watershed_contours.jpeg')
if showImages:
    plt.show()
    plt.close()
# the class or region of each pixel can be found be seeing the markers array


# counting number of coins in the image
# simply because we were told to make a function, I'm putting it in here. 
# I'm just using watershed itself
def count_coins(path1):
    img = cv2.imread(path1)

    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, bin_img = cv2.threshold(gray,
                                0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # all 1s in 3*3
    bin_img = cv2.morphologyEx(bin_img, 
                            cv2.MORPH_CLOSE,
                            kernel,
                            iterations=5)

    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)


    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 250, cv2.THRESH_BINARY) # 0.45 works for this particular image
    sure_fg = sure_fg.astype(np.uint8)  
    unknown = cv2.subtract(sure_bg, sure_fg)



    ret, markers = cv2.connectedComponents(sure_fg) # find connected components in foreground
    
    # Add one to all labels so that background is not 0, but 1
    markers += 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    labels = np.unique(markers)
    
    coins = []
    for label in labels[2:]:  
        # print('label is ', label)
        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # coins.append(contours[0])
        min_area=0
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
        if len(valid_contours) > 0:
            coins.append(valid_contours[0])
    # return len(valid_contours)
    return len(labels)-2 # background subtracted


print('number of coins according to watershed is', count_coins(path1))



def count_as_per_threshold(path1):
    img = cv2.imread(path1)
    img = cv2.resize(img , (640 , 800))
    image_copy = img.copy()
    img = cv2.GaussianBlur(img , (7 , 7) , 3)

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    ret , thresh = cv2.threshold(gray , 170 , 255 , cv2.THRESH_BINARY)
    # ret , thresh = cv2.threshold(gray , 90 , 255 , cv2.THRESH_BINARY)

    contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
    area = {}
    for i in range(len(contours)):
        cnt = contours[i]
        ar = cv2.contourArea(cnt)
        area[i] = ar
    srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
    results = np.array(srt).astype("int")
    num = np.argwhere(results[: , 1] > 500).shape[0]

    for i in range(1 , num):
        image_copy = cv2.drawContours(image_copy , contours , results[i , 0] ,
                                    (0 , 255 , 0) , 3)
    return num-1
    # cv2.imshow("final" , image_copy)

print('number of coins according to thresholding is', count_as_per_threshold(path1))