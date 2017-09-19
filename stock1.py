import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#plt.switch_backend('GTK') 
dates =[]
prices =[]

def get_data(filename):
    with open(filename,'r') as csvfile:
        CsvFileReader = csv.reader(csvfile)
        next(CsvFileReader)
        for row in CsvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    
    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3,degree = 2)
    svr_rdf = SVR(kernel= 'rdf', C=1e3,gamma = 0.1)
    
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rdf.fit(dates,prices)
    
    
    plt.scatter(dates,prices,color= 'black',label = 'Data')
    plt.plot(dates,svr_lin.predict(dates),color= 'green',label= 'Linear_model')
    plt.plot(dates,svr_poly.predict(dates),color= 'red',label= 'Polynomial_model')
    plt.plot(dates,svr_rdf.predict(dates),color= 'blue',label= 'RDF_model')
    plt.xlabel(dates)
    plt.ylabel(prices)
    plt.title('Support Vector Regression')
    plt.show()
    plt.legend()

    return svr_lin.predict(x)[0],svr_poly.predict(x)[0],svr_rdf.predict(x)[0]


get_data('CTSH.csv')
predicted_prices= predict_prices(dates,prices,29)
print (predicted_prices)

    
    
    
    
    
