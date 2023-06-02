""" 
COMP 472 Mini-Project 1
Due Date: June 5th 2023
Contributes:
Sabrina Kim 40066662
Youngjae Kim 40169282
Eyal Azimov 27527551
"""

# Import necessary libraries
from sklearn import preprocessing
from sklearn import tree
import numpy as np
import pandas as pd

"""
Returns user input array of types in transformed numerical values. 
Uses training data to get appropriate numerical values so 
decision tree can read it and make prediction.

Parameters:
- user_arr: 1D array of user inputs as strings
- training_arr: 2D array of training data as strings

Returns:
- 1D array of user inputs as numerical values
"""

def transfer_userInput(user_arr, training_arr):
  # Extract input features from training data
  X = np.array(training_arr[:, 0:10])
  # Convert user input to a NumPy array
  user_arr2 = np.array(user_arr)
  # Stack user input on top of training data
  new_X = np.vstack((X, user_arr2))
  # Apply LabelEncoder to each column of new_X
  for col_index in range(10):
     new_X[:, col_index] = le.fit_transform(new_X[:, col_index])
  # Return the last row of new_X as transformed user input
  return new_X[-1]

# Read dataset from CSV file
dataset = pd.read_csv(
    r'restaurant.csv')
# Create DataFrame from dataset with specific column names
df = pd.DataFrame(dataset, columns=['alternative', 'bar', 'Friday', 'hungry', 'patrons', 'price', 'raining', 'reservation', 'type',
                                    'waitEstimate', 'willWait'])
# Convert DataFrame to NumPy array
dfArray = pd.DataFrame.to_numpy(df)
# Remove spaces from elements in dfArray
# dfArray = [element.replace(" ", "") for element in dfArray]
dfArray_without_spaces = np.char.replace(dfArray.astype(str), " ", "")
# Convert elements in dfArray_without_spaces back to their original data type
dfArray_without_spaces = dfArray_without_spaces.astype(dfArray.dtype)

# Extract input features (X) and output labels (old_y) from dfArray_without_spaces
X = np.array(dfArray_without_spaces[:, 0:10])
old_y = np.array(dfArray_without_spaces[:,10])

# Create a LabelEncoder for numerical encoding
le = preprocessing.LabelEncoder()
# Iterate over each column of X and transform string classes into numerical encoded values
for col_index in range(10):
    X[:, col_index] = le.fit_transform(X[:, col_index])
# Transform output labels into numerical encoded values
y = le.fit_transform(old_y)

# Create Decision tree classifier
dtc = tree.DecisionTreeClassifier(criterion="entropy")
# Train dtc
dtc.fit(X, y)

# Print welcome messages and instructions
print("\nWELCOME TO THE RESTAURANT DECISION TREE CLASSIIER WITH ENTROPY-BASED SPLITTING AI")
print("---------------------------------------------------------------------------------\n")

print("This program predicts if you should wait for a table in a restaurent.")
print("Please fill in the following below, and the AI will tell you if it's worth waiting or not!\n")

# Prompt the user to enter inputs
alternative = input("Enter if there is an alternative. (\"Yes\" or \"No\") : ")
bar = input("Enter if there is an bar. (\"Yes\" or \"No\") : ")
friday = input("Enter if today is Friday. (\"Yes\" or \"No\") : ")
hungry = input("Enter if you are hungry. (\"Yes\" or \"No\") : ")
patrons = input(
    "Enter how many patrons there are. (\"Some\" or \"Full\" or \"None\") : ")
price = input("Enter how expensive it is. (\"$\" or \"$$\" or \"$$$\") : ")
raining = input("Enter if it is raining. (\"Yes\" or \"No\") : ")
reservation = input("Enter if reservation is possible. (\"Yes\" or \"No\") : ")
type_food = input("Enter the type of the food. (\"French\" or \"Thai\" or \"Burger\" or \"Italian\")  : ")
waitEstimate = input(
    "Enter the estimated time (\"0-10\" or \"10-30\" or \"30-60\" or \">60\")  : ")
print()

# Store user inputs in an array
user_array = [alternative, bar, friday, hungry, patrons,
              price, raining, reservation, type_food, waitEstimate]

# IGNORE: test array
# user_array = ['Yes', 'Yes', 'Yes', 'Yes','Full', '$', 'No', 'No', 'Burger', '30-60']

# Predict the output using the user input and the trained decision tree classifier
# print(transfer_userInput(user_array, dfArray_without_spaces))
y_pred = dtc.predict([transfer_userInput(user_array, dfArray_without_spaces)])
y = le.fit_transform(old_y)
# print("Predicted output: ", le.inverse_transform(y_pred))

# Print the prediction result indicating whether the user should wait for the table or not
print("Hmmm... The AI is thinking...")
if(le.inverse_transform(y_pred)[0]=="Yes"):
    print("Congrats! The AI says the restaurant it's definitely worth waiting for! :) \n")
else:
    print("Sorry! The AI says this resturant is not wroth the wait :( \n)")

