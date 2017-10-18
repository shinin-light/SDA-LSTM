import numpy as np
import json

data = json.load(open("data/harmonized.json", 'r'))
records_num = len(data["records"])
attributes_num = len(data["records"][0]["waves"][0]["values"])
classes_num = len(data["records"][0]["waves"][0]["class"]["one-hot"])

values = np.array([],dtype=float)
values = np.reshape(values, (-1,attributes_num))

classes = np.array([],dtype=float)
classes = np.reshape(values, (-1,classes_num))

print("Reading dataset...")
for r in data["records"]:
    #print(r["id"])
    for w in r["waves"]:
        tmp = np.reshape(w["values"], (-1,attributes_num))
        values = np.append(values, tmp, axis=0)
        tmp = np.reshape(w["class"]["one-hot"], (-1,classes_num))
        classes = np.append(classes, tmp, axis=0)

print("Saving...")
np.save("./data/records.npy",values)
np.save("./data/classes.npy",classes)
print("Ending.")
