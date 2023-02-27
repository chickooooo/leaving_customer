import scripts.utils as utils

# setup the model and other dependencies
setup = utils.Setup()

# data to test on
test_data = [
    [8, 15656148, 'Obinna', 376, 'Germany', 'Female', 29, 4, 115046.74, 4, 1, 0, 119346.88],   # will exit
    [24, 15725737, 'Mosman', 669, 'France', 'Male', 46, 3, 0.0, 2, 0, 1, 8487.75],   # will not exit
    [346, 15763859, 'Brown', 840, 'France', 'Female', 43, 7, 0.0, 2, 1, 0, 90908.95],   # will not exit
    [3454, 15737521, 'Ball', 619, 'Germany', 'Male', 40, 9, 103604.31, 2, 0, 0, 140947.05],   # will not exit
    [9999, 15682355, 'Sabbatini', 772, 'Germany', 'Male', 42, 3, 75075.31, 2, 1, 0, 92888.52]   # will exit
]

# make predictions
result = setup.predict(test_data)
# print result
print(result)
