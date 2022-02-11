#Emma Barake
#2021 Ecotone Waste CLassification Project

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


df = pd.read_pickle('waste_features.pkl')
features = df['vgg16_fc1_feature']
features = np.array(list(features)) # convert all stored features to a (1800, 4096) numpy array
features.shape


df.head()


df_sort =  df.sort_values('label')
features_sort = df_sort['vgg16_fc1_feature']
features_sort = np.array(list(features_sort))
plt.figure(figsize=(20,20))
plt.imshow(features_sort, cmap='RdBu', vmin=-1, vmax=1)
plt.xlabel('number of components')
plt.ylabel('index value of training/testing images')
plt.colorbar()


pca = PCA() # create pca object
pca.fit(features) # fit to features
var = pca.explained_variance_ratio_.cumsum() # cumulative fraction of variance 


fig, ax = plt.subplots(1,2, figsize=(6,4), dpi=150, sharey=True)
a = ax[0]
a.plot(range(1,len(var)+1), var)
a.set_xlabel('number of components')
a.set_ylabel('cumulative fraction of variance explained')
a.set_title('all 4096 components')
a = ax[1]
a.plot(range(1,81), var[:80])
a.set_xlabel('number of components')
a.set_title('first 80 components')
for a in ax:
    a.grid('on')
plt.show()

print(var[70])

pca_50 = PCA(n_components=70, # number of components we want to keep
             svd_solver='randomized', # for larger data matrices we can use 'randomized' (and set random_state if desired) which will run faster
             whiten=False) # will apply the whitening transformation after computing the components

X_pca_50 = pca_50.fit_transform(features)
print('PCA shape: ', X_pca_50.shape)


fig, ax = plt.subplots(1,1, figsize=(6,4), dpi=150)
ax.plot(X_pca_50[:,0], X_pca_50[:,1], marker='+', linestyle='')
ax.set_xlabel('1st PCA component')
ax.set_ylabel('2nd PCA component')


le = LabelEncoder()
le.fit(df['label'])
label_y = le.transform(df['label'])
print(df['label'].unique())
print(le.transform(df['label'].unique()))

X_train, X_test, y_train, y_test = train_test_split(X_pca_50, 
                                                    label_y,
                                                    test_size=0.3333,
                                                    random_state=111,
                                                    shuffle=True)

X_train = np.array(list(X_train))
X_test = np.array(list(X_test))

# Check the data split

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# ## Support Vector Machine Classification
# Train the SVM on the training data using kernal trick
clf = SVC(kernel='linear', gamma='auto', random_state=4)
clf.fit(X_train, y_train)


clf.score(X_test, y_test)


cm = confusion_matrix(y_true=y_test, y_pred=clf.predict(X_test))
print(cm)

# ### save trained classifier

def save_classifier(regr, file_name):
    try:
        REGR_PATH = os.path.join(str(file_name))
    except:
        return 'it didnt work...'

    #exporting df
    with open(REGR_PATH, 'wb') as f:
        pickle.dump(regr, f)

save_classifier(clf, 'trained_SVC')

save_classifier(pca_50, 'fitted_pca_50')


