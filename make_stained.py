from config import *
import os
import staintools
import cv2


normalizer = staintools.StainNormalizer(method='vahadane')
img = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMAGES_DIR,"10078.tiff")), cv2.COLOR_BGR2RGB)

normalizer.fit(img)

for ind,i in enumerate(os.listdir(TRAIN_IMAGES_DIR)):
    image = cv2.imread(os.path.join(TRAIN_IMAGES_DIR,i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = normalizer.transform(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(STAINED_IMAGES_DIR,i) ,image)
    if ind%10==0:
        print(ind)