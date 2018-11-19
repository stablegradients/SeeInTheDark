import json 

jsonlist=["train_exposures.json","test_exposures.json","val_exposures.json"]
for JsonName in jsonlist:
	with open(JsonName) as f:
		data = json.load(f)

	new_dict={}
	for key in data.keys():
		new_dict[key]={"Target":data[key][0],"Exposure":data[key][1]}
		

	with open("DoD"+JsonName, 'w') as fp:
		json.dump(new_dict, fp)	

