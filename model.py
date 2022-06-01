import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
from xgboost import XGBClassifier

df = pd.read_csv(r"C:\Users\Shashank Nidhi\Downloads\SaYoPillow.csv")
df1=df.pop('sl')
del df['rem']
x_train,x_test,y_train,y_test = train_test_split(df,df1,test_size=0.2)
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
deploy = {'model':xgb}
with open('xgb_model.pkl','wb') as file :
    pickle.dump(deploy,file)


    