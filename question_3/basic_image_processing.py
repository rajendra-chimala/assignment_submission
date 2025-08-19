import cv2 as cv

image = cv.imread("sample_images\image.png")

gray_img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

cv.imshow("Gray Image",gray_img)
cv.imshow("Original Image",image)

cv.waitKey(0)
cv.destroyAllWindows()


cv.imwrite("sample_images\copy_img.png",gray_img)


print(f"Image Height : ",image.shape[0])
print(f"Image Weight : ",image.shape[1])
print(f"Image Channel : ",image.shape[2])
