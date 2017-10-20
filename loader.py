import numpy as np
import json

data = json.load(open("data/harmonized.json", 'r'))
records_num = len(data["records"])
attributes_num = len(data["records"][0]["waves"][0]["values"])
classes_num = len(data["records"][0]["waves"][0]["class"]["one-hot"])

values = []
classes = []

values.append(np.array([],dtype=float))
values[0] = np.reshape(values[0], (-1,attributes_num + 1))
values.append(np.array([],dtype=float))
values[1] = np.reshape(values[1], (-1,attributes_num + 1))

classes.append(np.array([],dtype=float))
classes[0] = np.reshape(classes[0], (-1,classes_num))
classes.append(np.array([],dtype=float))
classes[1] = np.reshape(classes[1], (-1,classes_num))

print("Reading dataset...")
for i in range(len(data["records"])):
    if(i % 1000 == 0):
        print('Converting record {0}'.format(i))
    r = data["records"][i]
    db = r["id"].startswith("t") #0 = e_, 1 = t_
    idn = float(r["id"].split('_')[1])
    for w in r["waves"]:
        tmp = w["values"]
        tmp.insert(0, idn)
        tmp = np.reshape(tmp, (-1,attributes_num + 1))
        values[db] = np.append(values[db], tmp, axis=0)
        tmp = np.reshape(w["class"]["one-hot"], (-1,classes_num))
        classes[db] = np.append(classes[db], tmp, axis=0)

print("Saving...")
np.save("./data/e_records.npy",values[0])
np.save("./data/e_classes.npy",classes[0])
np.save("./data/t_records.npy",values[1])
np.save("./data/t_classes.npy",classes[1])
print("Ending.")
