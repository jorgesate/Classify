from pypylon import pylon
import cv2
 

EXPOSURE_TIME = 199990

tl_factory = pylon.TlFactory.GetInstance()
camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice())
camera.Open()

camera.ExposureTimeRaw = EXPOSURE_TIME
#percent by which the image is resized
SCALE_PERCENT = 50
#calculate the 50 percent of original dimensions

camera.StartGrabbing(1)

for i in range(50):
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        img = img[:, :, 0]
        print(i)

        width = int(img.shape[1] * SCALE_PERCENT / 100)
        height = int(img.shape[0] * SCALE_PERCENT / 100)
        dsize = (width, height)
        resized = cv2.resize(img, dsize)

        cv2.imshow("grayscale image", resized)
        cv2.waitKey(0)
        cv2.imwrite('./imgs/chiken_test' + str(i) + '.jpg',img)

camera.Close()
