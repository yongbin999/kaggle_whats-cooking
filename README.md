# kaggle_whats-cooking
NLP figure out the cusine type from just the ingredients. 

***nb results: 55-73%
python nb.py inputs/train.json
or
python nb.py inputs/train.json inputs/test.json


***precepetron: 73-77%
python logistic_perc.py inputs/train.json 
or 
python logistic_perc.py inputs/train.json inputs/test.json



***Scikit-learn models: 70-77%
python scikit_models.py inputs/train.json 

## Gradient descend 0.772662164927
## one vs rest  multiclass 0.768876611418
## output-code errorcorreting multiclass  0.76744423982