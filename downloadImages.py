# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:12:05 2015

@author: mdmiah
"""

import json
import urllib
from modelInputs import count

with open('Data/colour_duplicates.json') as fh:
    i = 0
    for line in fh:
        line = json.loads(line)
        url = line["image"]
        urllib.urlretrieve(url, "Data/images/"+str(i)+".jpg")
        i += 1
        if i>count:
        	break
    print "Done"
