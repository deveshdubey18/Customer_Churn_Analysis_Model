from src.customerchurnanalysismodel.data_ingestion import data_ingestion
from src.customerchurnanalysismodel.data_preprocessing import processing
from src.customerchurnanalysismodel.model_building import model
from src.customerchurnanalysismodel.model_cluster import model_cluster

def main():
    
    df=data_ingestion()
    print(df.shape)
    
    X_train,X_test,y_train,y_test = processing(df)
    
    result = model(X_train,X_test,y_train,y_test)
    
    print(result)
    
    kmeans= model_cluster(X_train,X_test)
    print(kmeans)
    
   
main()