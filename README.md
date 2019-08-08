# Project for ECE-422 (Data Mining) : Predictions on taxi-cancellations (https://www.kaggle.com/c/predicting-cab-booking-cancellations) using ML algorithms and creation of Graphic User Interface

**Files included** :
* final_project.ipynb : Jupyter notebook with the entire preprocessing and ML algorithms' implementation process
* main.py : My main python file with the entire code (including the GUI), main class, functions and comments
* Kaggle_YourCabs_training.csv : Our data (from Kaggle competition)
* final_df.csv : Our data after preprocessing and right before implementing the ML algorithms
* data fields.txt : Explanation of our data.
 
What **dataset** did we have? 
* A well-behaved csv dataset with multiple columns from a Kaggle competition. 
'car cancellation' column is our target variable, which shows whether or not the driver canceled the ride.

What was our **task**?
  1. Read, understand and preprocess our data.
  2. Implement whatever techniques I deemed fit for the problem in order to predict the cancellations.
  3. Create a GUI, that would automatically perform the algorithms on our dataset and show the results.
   
What **type of problem** was it?
* It was a classification problem, since we already knew which drivers canceled the ride, and we wanted
to come up with a model that would accurately predict that.
    
What **techniques** did we use?
  1. Preprocessing: Normalization, NaN values handling, column dropping etc...
  2. ML algorithms: Logistic Regression, Random Forest, Decision Tree, KNN, Gradient Boosting  
  
  
Important to **note**:
* Libraries: I used the PyQT5 library of python to create the GUI. Sklearn for preprocessing and ML algorithms, 
Numpy for array manipulation, Pandas for csv manipulation and Math for some mathematical operations. 
* Since for the purpose of the Class I was asked to create a GUI, the task became an *Object-Oriented* programming task. I am not
very comfortable with Object-Oriented programming, but I did my best, starting basically from zero.
