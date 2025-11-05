# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:50:02 2025

@author: kanch
"""

import qrcode

data = "Kanchana"
img = qrcode.make(data)
img.save("KQR.png")
