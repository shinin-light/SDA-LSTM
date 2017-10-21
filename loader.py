import numpy as np
import json

data = json.load(open("data/harmonized.json", 'r'))
records_num = len(data["records"])
waves_num = [6,2]
attributes_num = len(data["records"][0]["waves"][0]["values"])
classes_num = len(data["records"][0]["waves"][0]["class"]["one-hot"])

values = [[],[]]
classes = [[],[]]

print("Reading dataset...")
for i in range(len(data["records"])):
    if(i % 1000 == 0):
        print("Converting record n.{0}".format(i))

    r = data["records"][i] #Person
    db = r["id"].startswith("t") #0 = e_, 1 = t_
    if(len(r["waves"]) == waves_num[db]):
        for w in r["waves"]:
            values[db].append(w["values"])
            classes[db].append(w["class"]["one-hot"])
    
for i in range(2):
    values[i] = np.array(values[i], dtype=np.float32)
    values[i] = np.reshape(values[i], (-1, waves_num[i], attributes_num))
    classes[i] = np.array(classes[i], dtype=np.float32)
    classes[i] = np.reshape(classes[i], (-1, waves_num[i], classes_num))

print("Saving...")
np.save("./data/e_records.npy",values[0])
np.save("./data/e_classes.npy",classes[0])
np.save("./data/t_records.npy",values[1])
np.save("./data/t_classes.npy",classes[1])
print("Ending.")

'''
print("Saving...")
np.save("./data/e_records.npy",values[0])
np.save("./data/e_classes.npy",classes[0])
np.save("./data/t_records.npy",values[1])
np.save("./data/t_classes.npy",classes[1])
print("Ending.")
'''