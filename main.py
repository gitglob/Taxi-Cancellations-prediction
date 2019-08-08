# import PyQt5 libraries and packages
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QMessageBox, QApplication, QPushButton
from PyQt5.QtWidgets import QLabel, QComboBox, QVBoxLayout
from PyQt5.QtWidgets import QTableView
        
# import sklearn libraries and packages  
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

# import some packages for mathematical operations
from math import radians, sin, cos, acos

# some additional important libraries
import sys, csv

# numpy for array operations and pandas for csv handling
import numpy as np
import pandas as pd



#######################################################################################################################



# that's the preprocessing function (read our original data, preprocess and save it)
def process_data():
    # read csv file and pass to dataframe
    df=pd.read_csv('Kaggle_YourCabs_training.csv',encoding='utf-8',engine='python')
    
    # remove useless columns
    df=df.drop(['id','to_date'],axis=1)
    
    # antikatastash nan me meso oro twn oxi nan sto package_id
    count = 0
    j=0
    for i in df['package_id']:   
        if ( not(pd.isnull(i)) and (i != 'package_id') ):
            j += 1
            #print (count, i)
            count += float(i)
    package_mean = round(count/j,2)
    df['package_id'].fillna(package_mean, inplace = True)
    
    # Preprocessing gia to user_id
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df[['user_id']])
    df_normalized = pd.DataFrame(np_scaled)
    df['user_id'] = df_normalized
    
    # ONE-HOT ENCONDING FOR TRAVEL_TYPE_ID
    oh_travel_type_id = pd.get_dummies(df.travel_type_id).astype(int)
    oh_travel_type_id.columns = ['travel_type_1','travel_type_2','travel_type_3']
    
    df = pd.concat((df, oh_travel_type_id), axis = 1)
    df=df.drop(['travel_type_id'],axis=1)
    
    # one-hot for from_city_id after replacing Nans with city_id=15.0 !!!
    df['from_city_id'].fillna(15.0, inplace = True)
    c=0
    for i in df['to_city_id']:
        if pd.isnull(i):                #replacing Nans with city_id=15.0
            df['to_city_id'].fillna(df['from_city_id'][c], inplace = True)
        c += 1
    oh_from_city_id = pd.get_dummies(df.from_city_id).astype(int)
    oh_from_city_id.columns = ['from_city_1','from_city_15','from_city_31']
    df = pd.concat((df, oh_from_city_id), axis = 1)
    df=df.drop(['from_city_id'],axis=1)
    
    # new dataframe with diff meaning time from booking to reservation in days
    df['from_date']=pd.to_datetime(df['from_date'])
    df['booking_created']=pd.to_datetime(df['booking_created'])
    df_time= df['from_date'] - df['booking_created']
    diff_in_days=[]
    
    for i in df_time:
        curr=i.days+(i.seconds/(3600*24))
        diff_in_days.append(curr)
        
    # We observe that the biggest possibility to canceled a drive increase about 3% (10%), 
    # when the diff_in_days is less than 8 hours(diff_in_days<0.3).
    # So we insert a binary column which have that information    
    diff_in_days=pd.DataFrame({'diff_in_days': diff_in_days})
    diff_in_days.head()
    df= df.drop(['booking_created'],axis=1)
    df.insert(6, 'diff_in_days', diff_in_days)
    
    small_diff = []
    for i,j in zip(df.diff_in_days,df.Car_Cancellation):
        if i<0.3 :
            small_diff.append(1)
        else:
            small_diff.append(0)
    df.insert(6, 'small_diff_in_days', small_diff)
    
    # NORMALIZE VICHILE_MODEL_ID AND PACKAGE_ID
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df[['vehicle_model_id']])
    df_normalized = pd.DataFrame(np_scaled)
    df['vehicle_model_id'] = df_normalized
    
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df[['package_id']])
    df_normalized = pd.DataFrame(np_scaled)
    df['package_id'] = df_normalized
    
    # calculate km from coordinates and replace them
    km_list=[]
    for i,j,k,l in zip(df['from_lat'],df['from_long'],df['to_lat'],df['to_long']):
        slat = radians(float(i))
        slon = radians(float(j))
        elat = radians(float(k))
        elon = radians(float(l))
        dist = np.round(6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon)),1)
        km_list.append(dist)
    dist_in_km=pd.DataFrame({'dist_in_km': km_list})
    
    # drop columns and insert dist_in_km
    df= df.drop(['from_lat','from_long','to_lat','to_long'],axis=1)
    df.insert(8, 'dist_in_km', dist_in_km)
    
    # Calculate the mean of km to fill the NaNs
    count = 0
    j=0
    
    for i in df['dist_in_km']:
    
        if ( (not(pd.isnull(i))) and i != ('dist_in_km') ):
            j+=1
            count += float(i)
    
    km_mean=np.round(count/j,1)
    df['dist_in_km'].fillna(km_mean, inplace = True)
    
    # find holidays in dataset and set them as independent variables
    dates=[]
    date =[]
    for i in df['from_date']:
        date = [i.month,i.day]
        dates.append(date)
    
    holidays=[[1,26],[8,15],[10,2],[4,22],[1,14],[4,24],[3,29],[5,24],[9,9],[10,14],[10,13],[11,24],[11,17],[12,25],[1,1]]
    
    on_holidays=[]
    for i in dates:
        if i in holidays:
            on_holidays.append(1)
        else:
            on_holidays.append(0)
    on_holidays = pd.DataFrame({'On_holidays':on_holidays})
    df.insert(8, 'On_holidays', on_holidays)
    
    # drop 'from_date' column
    df= df.drop(['from_date'],axis=1)
    
    # drop to_area_id
    df= df.drop(['to_area_id'],axis=1)
    
    #drop the rest of NaNs
    df=df.dropna()
    
    # normalize from_area_id, to_city_id, dist_in_km
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df[['from_area_id','to_city_id','dist_in_km']])
    df_normalized = pd.DataFrame(np_scaled)
    df[['from_area_id','to_city_id','dist_in_km']] = df_normalized
    
    # drop remaining rows with NAN values
    df = df.dropna(how='any',axis=0)
    
    # save final dataframe
    df.to_csv('final_df.csv', sep=',', index=False, header=True)  
    
    return (df)



#######################################################################################################################



# this is the main window of our app which is connected to the widgets
class MainWindow(QMainWindow):
 
    # initializer / constractor
    def __init__(self, fileName, parent = None):
        super(MainWindow, self).__init__(parent)
        
        self.init_ui(fileName)
      
    # main method
    def init_ui(self, fileName):
                
        # save the filename
        self.fileName = fileName

        # create QMainWindow
        self.setObjectName("MainWindow")
        self.resize(666, 526)

        # add window title
        self.setWindowTitle("Final Project")
        
        # create central QWidget
        self.centralwidget = QWidget(self)
        
        # create models
        self.model = QtGui.QStandardItemModel(self.centralwidget)
        self.model2 = QtGui.QStandardItemModel(self.centralwidget)
        
        
        
####################################################################################################################
        
        

        # create title label   
        self.title = QLabel("Taxi Cancellations", self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setLineWidth(1)
        self.title.setScaledContents(False)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setWordWrap(False)
        
        # aem QLabel
        self.AEM = QLabel('AEM: 1833, 2013', self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setItalic(True)
        self.AEM.setFont(font)
        


#####################################################################################################################
        
        
        
        # load button
        self.load = QPushButton("Load", self.centralwidget)
        
        self.load.clicked.connect(self.on_pushButtonLoad_clicked) # connect with action 
        
        
        # create QTableView 1 (load)
        self.tableView = QTableView(self.centralwidget)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)  
        
        
        # create QTableView 2 (after preprocess)
        self.tableView2 = QTableView(self.centralwidget)
        self.tableView2.setModel(self.model2)
        self.tableView2.horizontalHeader().setStretchLastSection(True)  
               
        
        
        # preprocess button
        self.preprocess = QPushButton("Preprocess", self.centralwidget)
        
        self.preprocess.clicked.connect(self.PreprocessData) # connect with action
        
        
        
        # create combo box
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.addItem("Choose algorithm")
        self.comboBox.addItem("Logistic Regression")
        self.comboBox.addItem("Decision Tree (75-25 %)")
        self.comboBox.addItem("Decision Tree (10-folds)")
        self.comboBox.addItem("Random Forest (75-25 %)")
        self.comboBox.addItem("Random Forest (10-folds)")
        self.comboBox.addItem("KNN (75-25 %)")
        self.comboBox.addItem("KNN (10-folds)")
        self.comboBox.addItem("Gradient Boost (75-25 %)")
        
        self.comboBox.activated[str].connect(self.choose_alg) # connect to action depended on algorithm chosen   
        
        
        # apply algorithm QPushButton
        self.apply_algorithm = QPushButton('apply algorithm', self.centralwidget)
        
        self.apply_algorithm.clicked.connect(self.on_push_apply_alg) # connect with action
    
        
        

####################################################################################################################

        
        
        # set layout
        self.layoutVertical = QVBoxLayout(self.centralwidget)
        self.layoutVertical.addWidget(self.title)        
        self.layoutVertical.addWidget(self.AEM)
        self.layoutVertical.addWidget(self.tableView)
        self.layoutVertical.addWidget(self.tableView2)
        self.layoutVertical.addWidget(self.load)
        self.layoutVertical.addWidget(self.preprocess)
        self.layoutVertical.addWidget(self.comboBox)
        self.layoutVertical.addWidget(self.apply_algorithm)
            
        # finalize centralWidget
        self.setCentralWidget(self.centralwidget)
        
        
        
#####################################################################################################################
                                    
        
        
    # instantly load csv file
    def loadCsv(self, fileName):
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                if fileName != "final_df.csv":
                    self.model.appendRow(items)
                else:
                    self.model2.appendRow(items)

    # Preprocess our data
    def PreprocessData(self):
        #QMessageBox.about(self, "Preprocess Message", "This is the preprocess button")
        
        # call preprocess function
        df = process_data()
        
        # load the dataframe that our preprocess function created
        self.loadCsv("final_df.csv")
        
    # combo box action
    def choose_alg(self, text):
        if text == "Logistic Regression":
            self.n = 1
        elif text == "Decision Tree (75-25 %)":
            self.n = 2
        elif text == "Decision Tree (10-folds)":
            self.n = 3
        elif text == "Random Forest (75-25 %)":
            self.n = 4
        elif text == "Random Forest (10-folds)":
            self.n = 5
        elif text == "KNN (75-25 %)":
            self.n = 6
        elif text == "KNN (10-folds)":
            self.n = 7
        elif text == "Gradient Boost (75-25 %)":
            self.n = 8
        else:
            self.n == 0


    # apply algorithm button action
    def ApplyAlgorithm(self, n):
        # read csv file and pass to dataframe
        df = pd.read_csv('final_df.csv',encoding='utf-8',engine='python')
        
        # create X and y variables
        X = np.array([df.user_id,df.vehicle_model_id,df.package_id,df.from_area_id, 
                      df.to_city_id,df.small_diff_in_days,df.diff_in_days,df.On_holidays, 
                      df.dist_in_km,df.online_booking,df.mobile_site_booking,df.travel_type_1, 
                      df.travel_type_2,df.travel_type_3,df.from_city_1, 
                      df.from_city_15,df.from_city_31])
        X=X.T
        y=np.array(df.Car_Cancellation)
        y.reshape((y.shape[0],1))
        
        if n == 1:
            self.logistic(X,y)        
        elif n == 2:
            self.dec_tree_split(X,y)
        elif n == 3:
            self.dec_tree_tf(X,y)  
        elif n == 4:
            self.random_forest_split(X,y)
        elif n == 5:
            self.random_forest_tf(X,y) 
        elif n == 6:
            self.knn_split(X,y) 
        elif n == 7:
            self.knn_tf(X,y)  
        elif n == 8:
            self.gb_split(X,y)
        
        
#####################################################################################################################
                                    # Algorithms    

    
    # Logistic Regression
    def logistic(self, X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
        clf = linear_model.LogisticRegression(solver='sag',max_iter=400,tol=1e-6)
        model=clf.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        
        p1 = clf.score(X_test, y_test)
        r2score = r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        p2 = r2score
        p3 = y_pred
        
        QMessageBox.about(self, "Logistic Reression", 
                          " score:\n %s \n R-Squared Score is:\n %s \n y_pred:\n %s"%(p1, p2, p3))
        
        # H logistic para to megalo accuracy provlepei mono mhden
    
    # DecisionTree with split (25%)    
    def dec_tree_split(self, X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        regressor = DecisionTreeClassifier(random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test)
        
        p1 = confusion_matrix(y_test,y_pred)
        p2 = classification_report(y_test,y_pred)
        p3 = accuracy_score(y_test, y_pred)
        
        QMessageBox.about(self, "Decision Tree with split (75-25 %)", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
    # DecisionTree with 10-folds
    def dec_tree_tf(self, X,y):    
        skf = StratifiedKFold(n_splits=10)
        
        # blank lists to store predicted values and actual values
        predicted_y = []
        expected_y = []
        # partition data
        for train_index, test_index in skf.split(X, y):
        
            #print(train_index, test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = DecisionTreeClassifier(random_state=0)  #80% me 643 correct cancellation
            #clf=RandomForestClassifier(n_estimators=100,random_state=0)  #86% me 416 correct cancellation 
            clf.fit(X_train, y_train)  
            # y_pred = clf.predict(X_test) 
            
            predicted_y.extend(clf.predict(X_test))
            # store expected result for this specific fold
            expected_y.extend(y_test)
        
        p1 = confusion_matrix(expected_y,predicted_y)  
        p2 = classification_report(expected_y,predicted_y)  
        p3 = accuracy_score(expected_y,predicted_y) 
        
        QMessageBox.about(self, "Decision Tree with 10-folds", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
		# to confusion matrix exei morfh :   [true positives     false negatives
		#                                     false positives    true negatives]
		#true positives = swstes provlepseis gia 0
		#false positives = lathos provlepseis gia 0
		#false negative = lathos provlepseis gia 1
		#true negatives = swstes provlepseis gia 1


		#Precision -> 0:  true_positives/(true positives+false negatives) (column)
		#             1:  true_negatives / (false_positives+true_negatives) 
		#Recall ----> 0:  true_positives/support (row)
		#-----------> 1:  true_negatives/support  TO AVG/TOTAL RECALL EINAI KAI TO SYNOLIKO ACCURACY
		# f1-score--> 0,1:  2*(precision *recall) / (precision +recall) (the higher the better)
		# support --> total number of instances at each class
        
        
    # Random Forest with split
    def random_forest_split(self, X,y):
        # Random Forest with split
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        regressor = RandomForestClassifier(n_estimators=100,random_state=0)  
        #regressor = DecisionTreeClassifier(random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test) 
        
        p1 = confusion_matrix(y_test,y_pred)  
        p2 = classification_report(y_test,y_pred)  
        p3 = accuracy_score(y_test, y_pred)
        
        QMessageBox.about(self, "Random Forest with split (75-25 %)", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
        
    # Random forest with 10-folds
    def random_forest_tf(self, X,y):    
        skf = StratifiedKFold(n_splits=10)
        
        # blank lists to store predicted values and actual values
        predicted_y = []
        expected_y = []
        # partition data
        for train_index, test_index in skf.split(X, y):
        
            #print(train_index, test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #clf = DecisionTreeClassifier(random_state=0)  #80% me 643 correct cancellation
            clf=RandomForestClassifier(n_estimators=100,random_state=0)  #86% me 416 correct cancellation 
            clf.fit(X_train, y_train)  
            # y_pred = clf.predict(X_test) 
            # create and fit classifier
            #clf = linear_model.LogisticRegression(solver='sag',max_iter=600,tol=1e-4)
           # clf.fit(x_train,Y_train)
            # store result from classification
            predicted_y.extend(clf.predict(X_test))
            # store expected result for this specific fold
            expected_y.extend(y_test)
        
        p1 = confusion_matrix(expected_y,predicted_y) 
        p2 = classification_report(expected_y,predicted_y)
        p3 = accuracy_score(expected_y,predicted_y) 
        
        QMessageBox.about(self, "Random Forest with 10-folds", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
		# to confusion matrix exei morfh :   [true positives     false negatives
		#                                     false positives    true negatives]
		#true positives = swstes provlepseis gia 0
		#false positives = lathos provlepseis gia 0
		#false negative = lathos provlepseis gia 1
		#true negatives = swstes provlepseis gia 1


		#Precision -> 0:  true_positives/(true positives+false negatives) (column)
		#             1:  true_negatives / (false_positives+true_negatives)
		#Recall ----> 0:  true_positives/support (row)
		#-----------> 1:  true_negatives/support  TO AVG/TOTAL RECALL EINAI KAI TO SYNOLIKO ACCURACY
		# f1-score--> 0,1:  2*(precision *recall) / (precision +recall) (the higher the better)
		# support --> total number of instances at each class
        
    # K-N-N MODEL WITH SPLIT
    def knn_split(self, X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        regressor = KNeighborsClassifier()  
        #regressor = DecisionTreeClassifier(random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test) 
        
        p1 = confusion_matrix(y_test,y_pred)  
        p2 = classification_report(y_test,y_pred)  
        p3 = accuracy_score(y_test, y_pred)
        
        QMessageBox.about(self, "KNN with split (75-25 %)", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
    # KNN with tenfolds
    def knn_tf(self, X,y):
        # K-N-N MODEL WITH 10-FOLDS¶
        skf = StratifiedKFold(n_splits=10)
        
        # blank lists to store predicted values and actual values
        predicted_y = []
        expected_y = []
        # partition data
        for train_index, test_index in skf.split(X, y):
        
            #print(train_index, test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #clf = DecisionTreeClassifier(random_state=0)  #80% me 643 correct cancellation
            clf=KNeighborsClassifier()  #86% me 416 correct cancellation 
            clf.fit(X_train, y_train)  
            # y_pred = clf.predict(X_test) 
            # create and fit classifier
            #clf = linear_model.LogisticRegression(solver='sag',max_iter=600,tol=1e-4)
           # clf.fit(x_train,Y_train)
            # store result from classification
            predicted_y.extend(clf.predict(X_test))
            # store expected result for this specific fold
            expected_y.extend(y_test)
        
        # results
        p1 = confusion_matrix(expected_y,predicted_y)  
        p2 = classification_report(expected_y,predicted_y)  
        p3 = accuracy_score(expected_y,predicted_y)
        
        QMessageBox.about(self, "KNN with 10-folds", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
    
    
    # gradient boosting with split
    def gb_split(self, X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        regressor = GradientBoostingClassifier(subsample=0.8,learning_rate=0.2,n_estimators=140,max_depth=4)  
        #regressor = DecisionTreeClassifier(random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test) 
        
        p1 = confusion_matrix(y_test,y_pred)  
        p2 = classification_report(y_test,y_pred)  
        p3 = accuracy_score(y_test, y_pred)
        
        QMessageBox.about(self, "Gradient Boosting with split (75-25 %)", 
                          " confusion matrix:\n %s \n classification report:\n %s \n accuracy score:\n %s"%(p1, p2, p3))
        
        
        
#####################################################################################################################
        

                
    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()         
                        


######################################################################################################################
        

            
    @QtCore.pyqtSlot()
    def on_pushButtonLoad_clicked(self):
        self.loadCsv(self.fileName)
        
    @QtCore.pyqtSlot()
    def on_push_apply_alg(self):
        self.ApplyAlgorithm(self.n)



######################################################################################################################


        
if __name__ == '__main__':
    
    # define our app
    app = QApplication(sys.argv)
    app.setApplicationName('Taxi Cancellations')
    
    # create the main window
    main = MainWindow("Kaggle_YourCabs_training.csv")
    
    # show the main window
    main.show()
    
    # close app when user says so
    sys.exit(app.exec_())
    
