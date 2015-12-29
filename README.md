## kaggle_whats-cooking
The goal of this project is to classify cuisines based on a list of ingredients from Kaggle Data Challenge dataset. My approach was to establish a baseline model using Naïve Bayes, then test performance against logistic models, and use model libraries such as Scikit-learn. The baseline NB model performed at 73% accuracy, Logistic Percepetron at 78%, and out-of-box Scikit-learn models around 77%. The top rank model in Kaggle stands at 83%.


###nb results: 55-73%
python nb.py inputs/train.json
or
python nb.py inputs/train.json inputs/test.json


###precepetron: 73-77% +78.2 with features
<br>python logistic_perc.py inputs/train.json 
<Br>or 
<br>python logistic_perc.py inputs/train.json inputs/test.json



###Scikit-learn models: 70-77%
python scikit_models.py inputs/train.json 

### Gradient descend 0.772662164927
### one vs rest  multiclass 0.768876611418
### output-code errorcorreting multiclass  0.76744423982
