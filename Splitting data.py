#using sklearn splitting data into train and test
X = transactions.drop(labels='Class', axis=1) 
y = transactions.loc[:,'Class']               
del transactions                              

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
del X, y

X_train.shape
X_test.shape

# Prevent view warnings
X_train.is_copy = False
X_test.is_copy = False
