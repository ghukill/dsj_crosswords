"""
https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html
https://dell-research-harvard.github.io/HJDataset/
https://stackoverflow.com/questions/44649449/brew-installation-of-python-3-6-1-ssl-certificate-verify-failed-certificate/44649450#44649450
"""

import layoutparser as lp
import cv2

image = cv2.imread("test_data/dsj_t1.jpg")
image = image[..., ::-1]

model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})