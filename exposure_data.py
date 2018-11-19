import json

with open('Sony_test_list.txt') as f:
	lines = f.readlines()
names = [line.split(' ') for line in lines]
exps = [(float(name[0][22:-5]),float(name[1][21:-5])) for name in names]
ratios = [exp[1]/exp[0] for exp in exps]
dictionary = dict(zip([name[0] for name in names],zip([name[1] for name in names],ratios)))
with open('test_exposures.json','w') as fw:
	json.dump(dictionary,fw)