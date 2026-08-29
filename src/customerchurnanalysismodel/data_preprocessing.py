from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize

def processing(df):
    

    # dropping duplicated vals
    df.drop_duplicates()
    
    # imputing outliers
    for i in df.select_dtypes(exclude='object').columns:
        df[i] = winsorize(df[i],limits=[0.05,0.05])
  
    # dropping unwanted columns
    df = df.drop(columns=['CustomerID','ServiceArea'])
    
    # encoding categorical to numerical
    df['Churn'] = df['Churn'].map({'No':0,'Yes':1})
    
    # Split the Target column and Input Features
    X = df.drop(columns='Churn')
    y = df['Churn']
    
    # split data into categorical and numerical
    categorical = X.select_dtypes(include='object').columns
    numerical = X.select_dtypes(exclude='object').columns 
    
    # Spliting data in to Training and Testing data
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                test_size=0.3,
                                                random_state=1)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    
    transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])
    
    X_train = transformer.fit_transform(X_train) # type: ignore
    X_test = transformer.transform(X_test)

    # Balancing bias data
    sm=SMOTE()
    X_train,y_train = sm.fit_resample(X_train,y_train) # type: ignore
    
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train) # type: ignore
    X_test = pca.transform(X_test)

    return X_train,X_test,y_train,y_test