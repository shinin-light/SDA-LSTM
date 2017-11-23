import numpy as np
import json

training_frac = 0.7
validation_frac = 0.2

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
    if(len(r["waves"]) == waves_num[db]): #Ignores one-wave-only records
        for w in r["waves"]:
            values[db].append(w["values"])
            classes[db].append(w["class"]["one-hot"])
    
for i in range(2):
    values[i] = np.array(values[i], dtype=np.float32)
    values[i] = np.reshape(values[i], (-1, waves_num[i], attributes_num))
    classes[i] = np.array(classes[i], dtype=np.float32)
    classes[i] = np.reshape(classes[i], (-1, waves_num[i], classes_num))

training_values = [[],[]]
training_classes = [[],[]]
validation_values = [[],[]]
validation_classes = [[],[]]
test_values = [[],[]]
test_classes = [[],[]]

for i in range(2):
    length = len(values[i])
    training_len = int(length * training_frac)
    validation_len = int(length * validation_frac)
    
    idx = np.random.choice(length, training_len, replace=False)
    training_values[i] = values[i][idx]
    training_classes[i] = classes[i][idx]
    values[i] = values[i][~idx]
    classes[i] = classes[i][~idx]

    idx = np.random.choice(len(values[i]), validation_len, replace=False)
    validation_values[i] = values[i][idx]
    validation_classes[i] = classes[i][idx]
    test_values[i] = values[i][~idx]
    test_classes[i] = classes[i][~idx]

for i in range(2):
    length = len(values[i])
    training_len = int(length * training_frac)
    validation_len = int(length * validation_frac)
    
    idx = np.random.choice(length, training_len, replace=False)
    training_values[i] = values[i][idx]
    training_classes[i] = classes[i][idx]
    values[i] = values[i][~idx]
    classes[i] = classes[i][~idx]

    idx = np.random.choice(len(values[i]), validation_len, replace=False)
    validation_values[i] = values[i][idx]
    validation_classes[i] = classes[i][idx]
    test_values[i] = values[i][~idx]
    test_classes[i] = classes[i][~idx]

print("Saving...")
np.save("./data/e_records_training.npy",training_values[0])
np.save("./data/e_classes_training.npy",training_classes[0])
np.save("./data/t_records_training.npy",training_values[1])
np.save("./data/t_classes_training.npy",training_classes[1])

np.save("./data/e_records_validation.npy",validation_values[0])
np.save("./data/e_classes_validation.npy",validation_classes[0])
np.save("./data/t_records_validation.npy",validation_values[1])
np.save("./data/t_classes_validation.npy",validation_classes[1])

np.save("./data/e_records_test.npy",test_values[0])
np.save("./data/e_classes_test.npy",test_classes[0])
np.save("./data/t_records_test.npy",test_values[1])
np.save("./data/t_classes_test.npy",test_classes[1])
print("Ending.")

'''
print("Saving...")
np.save("./data/e_records.npy",values[0])
np.save("./data/e_classes.npy",classes[0])
np.save("./data/t_records.npy",values[1])
np.save("./data/t_classes.npy",classes[1])
print("Ending.")
'''