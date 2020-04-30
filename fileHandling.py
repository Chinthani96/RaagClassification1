
my_list = [1,2,3,4]
with open('test_file.txt', 'w') as f:
    for item in my_list:
        f.write("%s\n" % item)