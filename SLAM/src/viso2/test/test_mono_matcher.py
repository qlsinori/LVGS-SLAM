#!/usr/bin/env python2

import cv2
#from matplotlib import pyplot as plt
import os
import unittest

import viso2.matcher as matcher

# def plot(features, img):
#    
#    for f in features:
#        # cv2.circle(img,(int(f.u1p),int(f.v1p)), 1, [100],2,-1)
#        cv2.circle(img,(int(f.u1c),int(f.v1c)), 2, [100],1,-1)
#        cv2.line(img,(int(f.u1p),int(f.v1p)),(int(f.u1c),int(f.v1c)), [100], 1, 1)
#    
#    plt.imshow(img,cmap='gray')
#    plt.show()

class TestVisoMonoMatcher(unittest.TestCase):
    def test_number_matches(self):
        params=matcher.MatcherParams()
        m=matcher.Matcher(params)
        
        self.assertEqual(len(m.getMatches()), 0)
        
        test_dir = os.path.dirname(os.path.realpath(__file__)) 
        img0=cv2.imread(os.path.join(test_dir, "000106.png"),0)
        img1=cv2.imread(os.path.join(test_dir, "000107.png"),0)
        
        matcher.pushBack(m,img0,False)
        matcher.pushBack(m,img1,False)
        
        m.matchFeatures(0)
        
        features=m.getMatches()

        print("number features="+str(len(features)))
        self.assertEqual(len(features), 3138)

    #    plot(features,I1)

if __name__ == '__main__':
    unittest.main()

