import os
import shutil

parent = os.getcwd()
curr = parent + "/data"

cat_train_count, dog_train_count = 0, 0
cat_test_count, dog_test_count = 0, 0
train_size, test_size = 1000, 400

for file in os.listdir(curr):
	animal = file.split(".")[0]
	if animal == "cat":
		if cat_train_count < train_size:
			shutil.copy2(curr + '/' + file, parent + '/train/cats')
			cat_train_count += 1
		elif cat_test_count < test_size:
			shutil.copy2(curr + '/' + file, parent + '/test/cats')
			cat_test_count += 1
	elif animal == "dog":
		if dog_train_count < train_size:
			shutil.copy2(curr + '/' + file, parent + '/train/dogs')
			dog_train_count += 1
		elif dog_test_count < test_size:
			shutil.copy2(curr + '/' + file, parent + '/test/dogs')
			dog_test_count += 1

	if cat_train_count == dog_train_count == train_size and cat_test_count == dog_test_count == test_size:
		break

print(cat_train_count, dog_train_count, cat_test_count, dog_test_count)
