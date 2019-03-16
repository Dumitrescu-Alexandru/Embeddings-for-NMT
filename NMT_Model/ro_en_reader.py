
#%%

import numpy as np
import pandas as pd
import os

with open("data/Lit/train.lit.ro",mode='r') as f:

    content = f.readlines()
content_ro = [x.strip() for x in content] 

with open("data/Lit/train.lit.en",mode='r') as f:
    content = f.readlines()
content_en = [x.strip() for x in content] 

content_ro[0] = content_ro[0].replace('\ufeff',"")
content_en[0] = content_en[0].replace('\ufeff',"")
content = [[content_en[i],content_ro[i]] for i in range(len(content_ro))]

content[0]
data = pd.DataFrame(data=content)