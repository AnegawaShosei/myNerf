import read_write_model as colmap_reader
import numpy as np
import imageio.v3 as iio
import cv2


def read_data(camFile, imageFile, imagedir):
    cameraData = colmap_reader.read_cameras_binary(camFile)
    imgData = colmap_reader.read_images_binary(imageFile)
    
    print("Loading Poses...")
    imgNames = []
    w2c = []
    
    for key in imgData:
        imgNames.append(imgData[key].name)
        qvec = imgData[key].qvec2rotmat() #Getting rotation and translation matrices
        tvec = imgData[key].tvec
        mat = np.concatenate([qvec, tvec.reshape(-1, 1)], axis = 1) #Building world to camera matrix
        lastrow = np.array([0, 0, 0, 1.]).reshape(1, 4)
        mat = np.concatenate([mat, lastrow], axis = 0)
        w2c.append(mat)
    
    w2c = np.stack(w2c, 0)
    c2w = np.linalg.inv(w2c)

    print("Done")
    
    cam = cameraData[1]
    H = cam.height
    W = cam.width
    F = cam.params[0] #Focal length

    print("Reading Images...")
    images = []
    for fname in imgNames:
        im = cv2.imread(imagedir + '/' + fname)
        #images.append(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)))
        images.append(im)
    
    images = np.array(images, dtype = np.float32) / 255.
    print("Done")
    
    return images, c2w, W, H, F