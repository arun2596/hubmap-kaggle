from config import *
import os
import staintools
import cv2


# normalizer = staintools.StainNormalizer(method='vahadane')
# img = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMAGES_DIR,"10078.tiff")), cv2.COLOR_BGR2RGB)

# normalizer.fit(img)

# for ind,i in enumerate(os.listdir(TRAIN_IMAGES_DIR)):
#     image = cv2.imread(os.path.join(TRAIN_IMAGES_DIR,i))
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     image = normalizer.transform(image)
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(STAINED_IMAGES_DIR,i) ,image)
#     if ind%10==0:
#         print(ind)


from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
from staintools.miscellaneous.get_concentrations import get_concentrations
import numpy as np


extractor = MacenkoStainExtractor

image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_IMAGES_DIR,'351.tiff')), cv2.COLOR_BGR2RGB)
h,w = image.shape[-3:-1]

target_image = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMAGES_DIR,"10078.tiff")), cv2.COLOR_BGR2RGB)
stain_matrix_target = extractor.get_stain_matrix(target_image)
target_concentration = get_concentrations(target_image, stain_matrix_target)
maxC_target = np.percentile(target_concentration, 99, axis=0).reshape((1, 2))

stain_matrix =  extractor.get_stain_matrix(image)
source_concentration = get_concentrations(image, stain_matrix)

s0 = source_concentration[:,0].reshape(h,w)
s1 = source_concentration[:,1].reshape(h,w)

# Top Hat Transform
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
top_hat   = cv2.morphologyEx(s0, cv2.MORPH_TOPHAT, kernel)# Black Hat Transform
black_hat = cv2.morphologyEx(s0, cv2.MORPH_BLACKHAT, kernel)
s0_aug =  s0 +  top_hat 

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
# top_hat   = cv2.morphologyEx(s1, cv2.MORPH_TOPHAT, kernel)# Black Hat Transform
# black_hat = cv2.morphologyEx(s1, cv2.MORPH_BLACKHAT, kernel)
# s1_aug = s1 -  top_hat 


s0s1 = np.dstack([s0_aug,s1])
s0s1_flat = s0s1.reshape(-1,2)

maxC_source = np.percentile(source_concentration, 99, axis=0).reshape((1, 2))
s0s1_flat *= 0.8 * (maxC_target / maxC_source)

augment_flat = 255 * np.exp(-1 * np.dot(s0s1_flat, stain_matrix_target))
augment = augment_flat.reshape((h,w,3))


cv2.imshow('augment', cv2.cvtColor(augment.astype('uint8'), cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()