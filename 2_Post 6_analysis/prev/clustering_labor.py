#######################
# K Nearest Neighbors #
#######################

# importing libraries
import itertools
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn.utils
from sklearn import metrics
from sklearn import manifold, datasets

import random

import scipy
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance_matrix

import pylab

##################
# Importing Data #
##################

# opening csv file and reading it into a variable called df
df = pd.read_csv("combined_data.csv")

# checking dataset head
print(df.head())

# 4 dataframe buckets, s= seasonally adjusted, u= unadjusted, r=rate, l=level
df1 = df[(df['seasonal_adjustment'] == "S") & (df['rate_level'] == "R")]
df2 = df[(df['seasonal_adjustment'] == "S") & (df['rate_level'] == "L")]
df3 = df[(df['seasonal_adjustment'] == "U") & (df['rate_level'] == "R")]
df4 = df[(df['seasonal_adjustment'] == "U") & (df['rate_level'] == "L")]

###################################
# Data Visualization and Analysis #
###################################
'''
# checking frequency tabs for one of our columns
print(df2['industry_text'].value_counts())

# all categories are equally represented for each period
# industry is our depedent var for which we want to predict values for turnover numbers or buckets

# plotting a histogram of bls quit numbers
df2[df2['value_qu'] < 800].hist(column='value_qu', bins=50)
plt.show()

###############
# Feature Set #
###############

# printing off some of our columns
# to hone in on key features
print(df.columns)

# converting pd df to np array for scikitlearn usage
X = df2[['year', 'period', 'value_jo', 'value_hi', 'value_ts',
         'value_qu', 'value_dl']].values # .astype(float)
print(X[0:5])

# subsetting and selecting variable of choice
y = df2['industry'].values
print(y[0:5])

########################
# Normalizing our Data #
########################

# literally normalizes - mean 0, var 1
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

# creating a test train split suited to our problem

# test size is 20 percent
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print  ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)


##################
# Classification #
##################

# Training the algorithm, with k = 4
k = 4

# training model and predicting
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting estimated(y-values) using testset(x-values) as input
yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1, Ks):

    # Training our Model and Predicting
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# plotting model-accuracy for different numbers of neighbors

plt.plot(range(1,Ks), mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,
                            mean_acc + 1 * std_acc,
                            alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


######################
# K-Means Clustering #
######################

## Note: clustering is considered an unsupervised learning method

###################
# Generating Data #
###################

# importing library
np.random.seed(0)

# creating clusters or blobs of 5000 points at a time
# centered at the named coordinates

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1],
                [2, -3], [1, 2]], cluster_std=0.9)

# plotting the data: iteration 1: 4 clusters
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# initializing k means feature matrix
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# fitting feature matrix to blobs above
k_means.fit(X)

# labelling each point
k_means_labels = k_means.labels_
k_means_labels

# grabbing the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#######################################
# Creating a data visual: iteration 1 #
#######################################
# initializing dimensions
fig = plt.figure(figsize=(6, 4))

# colors: using map, produces array given number of labels
# k_means_labels - helps produce coloring
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# building a plot
ax = fig.add_subplot(1, 1, 1)

# for loop plotting data points and centroids
# k ranges from 0-3, and matches the cluster number

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])),
                            colors):

    # creating a list of all data points
    # checking for 'is a part of' cluster
    # labelling points in the set as true, else false
    my_members = (k_means_labels == k)

    # defining the centroid, or cluster center, using function calculator
    cluster_center = k_means_cluster_centers[k]

    # plotting datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                            markerfacecolor=col, marker='.')

    # plotting centroids distinctly
    ax.plot(cluster_center[0], cluster_center[1], 'o',
                            markerfacecolor=col, markeredgecolor='k',
                            markersize=6)

# titling
ax.set_title('KMeans')

# Removing x and y axis ticks
ax.set_xticks(())
ax.set_yticks(())

# showing the plot
plt.show()

#########################
# Practice: iteration 2 #
#########################

# picking initial centroids is somewhat arbitrary
X, y = make_blobs(n_samples=5000, centers=[[-1,3], [2, -1],
                [-2, -2]], cluster_std=0.9)

# plotting the data: iteration 2: 3 clusters
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# initializing k means feature matrix
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

# fitting feature matrix to blobs above
k_means.fit(X)

# labelling each point
k_means_labels = k_means.labels_
k_means_labels

# grabbing the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#######################################
# Creating a data visual: iteration 2 #
#######################################
# initializing dimensions
fig = plt.figure(figsize=(6, 4))

# colors: using map, produces array given number of labels
# k_means_labels - helps produce coloring
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# building a plot
ax = fig.add_subplot(1, 1, 1)

# for loop plotting data points and centroids
# k ranges from 0-3, and matches the cluster number

for k, col in zip(range(len([[-1,3], [2, -1], [-2, -2]])),
                            colors):

    # creating a list of all data points
    # checking for 'is a part of' cluster
    # labelling points in the set as true, else false
    my_members = (k_means_labels == k)

    # defining the centroid, or cluster center, using function calculator
    cluster_center = k_means_cluster_centers[k]

    # plotting datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                            markerfacecolor=col, marker='.')

    # plotting centroids distinctly
    ax.plot(cluster_center[0], cluster_center[1], 'o',
                            markerfacecolor=col, markeredgecolor='k',
                            markersize=6)

# titling
ax.set_title('KMeans')

# Removing x and y axis ticks
ax.set_xticks(())
ax.set_yticks(())

# showing the plot
plt.show()
# cluster pyramid has been established


#########################################
# Industry Segmentation: Pre-processing #
#########################################

# using df of choice
df = df2[['industry', 'year', 'period', 'value_jo', 'value_hi', 'value_ts',
         'value_qu', 'value_dl']]

# there are a few different ways to do this
# here, using the standard deviation
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

###################################
# Industry Segmentation: Modeling #
###################################

# setting a global variable
clusterNum = 3

# initializing
k_means = KMeans(init = "k-means++", n_clusters = clusterNum,
                n_init = 12)

# fitting model to real datapoints
k_means.fit(X)

# extracting assigned cluster
labels = k_means.labels_
print(labels)

###################################
# Industry Segmentation: Insights #
###################################

# new column
df["Clus_km"] = labels
df.head(5)

# checking centroid values, w/o resorting to a function
print(df.groupby('Clus_km').mean())
# displays mean value for each var per cluster centroid

# 7 rows, 0 to 6
print(X.shape)
print(type(X))
print(X.dtype.names)


#############plot 1: 2d##########################

# plotting little circles for each cluster
area = np.pi * ( X[:, 1])**2

plt.scatter(X[:, 2], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Job Openings', fontsize=18)
plt.ylabel('Hires', fontsize=16)
plt.show()

# analysis variables: of interest
# value_jo, value_hi, value_ts, value_qu, value_dl

#############plot 2: 3d##########################

# initializing dimensions
fig = plt.figure(1, figsize=(8, 6))
plt.clf()

# declaring 3d object
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

# labeling axes
ax.set_xlabel('Total Separations')
ax.set_ylabel('Quits')
ax.set_zlabel('LayoffsDischarges')

ax.scatter(X[:, 4], X[:, 5], X[:, 6], c= labels.astype(np.float))
plt.show()



###########################
# Hierarchical Clustering #
###########################


###################
# Generating Data #
###################

# specifying cluster centers
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1],
                                [1, 1], [10,4]], cluster_std=0.9)

# plotting scatterplot

plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()


############################
# Agglomerative Clustering #
############################

# aggregation: declaring an agglomerative clustering object
agglom = AgglomerativeClustering(n_clusters = 4,
                                linkage = 'average')

# fitting model to the data
agglom.fit(X1, y1)

# creating a figure frame using dimensions 6 x 4
plt.figure(figsize=(6,4))

# scaling data points down, to fit closer together

# min-max range for X1
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# averaging distance for X1
X1 = (X1 - x_min) / (x_max - x_min)

# looping to display all datapoints
for i in range(X1.shape[0]):
    # replacing pts. w/ cluster value
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
            color=plt.cm.nipy_spectral(agglom.labels_ [i] / 10.),
            fontdict={'weight':'bold', 'size': 9})

# removing x ticks, y ticks, and y axis
plt.xticks([])
plt.yticks([])

# plt.axis('off')

# displaying the plot of original data, then clustering

plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()


#########################################
# Agglom. Clustering to Dendrogram Plot #
#########################################

# printing distance matrix between features
dist_matrix = distance_matrix(X1, X1)
dist_matrix

# declaring clustering object
Z = hierarchy.linkage(dist_matrix, 'complete')

# fitting model

dendro = hierarchy.dendrogram(Z)
plt.show()


# declaring clustering object: iteration 2
Z2 = hierarchy.linkage(dist_matrix, 'average')

# fitting model iteration 2

dendro = hierarchy.dendrogram(Z2)
plt.show()


##############################
# Application: Industry Data #
##############################


###########################
# Industry Data: Cleaning #
###########################

pdf = df2

# printing specs
# print("Shape of dataset before cleaning: ", pdf.size)

# conversion type to num
pdf[['industry', 'year', 'period', 'value_jo', 'value_hi', 'value_ts',
         'value_qu', 'value_dl']] = pdf[['industry', 'year', 'period',
                            'value_jo', 'value_hi', 'value_ts',
                            'value_qu', 'value_dl']].apply(pd.to_numeric,
                                errors='coerce')

# dropping missing values
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
# print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

###################################
# Industry Data: Feature Selection #
###################################

featureset = pdf[['industry', 'year', 'period', 'value_jo', 'value_hi', 'value_ts',
         'value_qu', 'value_dl']]

################################
# Industry Data: Normalization #
################################


x = featureset.values # grabs var list into an array

min_max_scaler = MinMaxScaler()

# preparing to scale for comparability
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx [0:5])

######################################
# Industry Data: Clustering w/ Scipy #
######################################


leng = feature_mtx.shape[0]


# initializing
D = scipy.zeros([leng,leng])


# calculating and storing distances
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i],
                                                  feature_mtx[j])

Z = hierarchy.linkage(D, 'complete')


max_d = 3


# assigning clusters to vehicle data
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# setting cluster number
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)


# finally, plotting the dendrogram
fig = plt.figure(figsize=(18, 50))
def llf(id):
    return '[%s %s %s]' % (pdf['industry_text'][id],
                            pdf['year'][id],
                            int(float(pdf['period'][id])))

dendro = hierarchy.dendrogram(Z, leaf_label_func=llf,
                        leaf_rotation=0, leaf_font_size=12,
                        orientation = 'right')


#############################################
# Industry Data: Clustering w/ Scikit-learn #
#############################################

# creating distance matrix
dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

# declaring and fitting clustering object
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

# assigning labels
pdf['cluster_'] = agglom.labels_

# printing data
print(pdf.head())

n_clusters = max(agglom.labels_) + 1

# coloring and labeling clusters
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# creating figure frame
plt.figure(figsize=(16,14))


for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.value_jo[i], subset.value_hi[i],
                str(subset['industry_text'][i]),rotation=25)
    plt.scatter(subset.value_jo, subset.value_hi, s= subset.value_qu*10,
                c=color, label='cluster'+str(label), alpha=0.5)
    # plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('value_jo')
    plt.ylabel('value_hi')

plt.show()
'''


#####################
# DBSCAN Clustering #
#####################


##################
# Generating Data #
##################

# function allocates points inputted to centroids, for number of simulations run
def createDataPoints(centroidLocation, numSamples,
                                        clusterDeviation):
    # creating and storing randomly generated data in matrix shell
    # feature matrix: X, target vector: y
    X, y = make_blobs(n_samples=numSamples,
                    centers=centroidLocation,
                    cluster_std=clusterDeviation)

    # standardizing using the mean diff / var method
    X = StandardScaler().fit_transform(X)
    return X, y

# assigning clusters
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)


################################################
# Modeling our Clusters: Declaring and Fitting #
################################################

epsilon = 0.3
minimumSamples = 7

# fitting dbscan density based clustering to generated data
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)

# checking labels, assignment
labels = db.labels_
labels

#####################
# Identify outliers #
#####################
# creating an array of booleans for testing cluster assignment
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# counts number of clusters present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# subsetting to only unique values
unique_labels = set(labels)
print(unique_labels)

###############
# Data vizzes #
###############
# creating colors for our clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# plotting points using colors

for k, col in zip(unique_labels, colors):
    if k == -1:
        # black is used for random noise
        col = 'k'

    class_member_mask = (labels == k)

    # plotting datapoints assigned clusters
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o',
                                                alpha=0.5)
plt.show()


######################################
# Practice: contrasting with k-means #
######################################

# initializing k means feature matrix
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

# fitting feature matrix to blobs above
k_means.fit(X)

# labelling each point
k_means_labels = k_means.labels_
k_means_labels

# grabbing the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# plotting little circles for each cluster
area = np.pi * ( X[:, 1])**2

plt.scatter(X[:, 0], X[:, 1], s=area, c=labels.astype(np.float), alpha=0.5)
plt.show()
# very sporadic, and not fully capturing the overlay of the cluster



####################
# 2: Industry Data #
####################

pdf = df4


################################
# 3: Industry Data: Clustering #
################################


sklearn.utils.check_random_state(1000)

Clus_dataSet = pdf[['value_jo', 'value_hi']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# computing density based clusters
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# printing few select columns
print(pdf[["industry_text", "value_jo", "value_hi", "Clus_Db"]].head(5))

# for outliers, cluster label is -1
print(set(labels))

##########################################
# Visualizing Clusters Based on Location #
##########################################
'''
plt.rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, # min longitudes and latitudes
            urcrnrlon=ulon, urcrnrlat=ulat) # max longitudes and latitudes

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
# my_map.shadedrelief() # choosing not to render this fancy layer

# creating a colored map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4])) if clust_number == -1 else colors[np.int(clust_number)]
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color =c, marker='o',
                    s=20, alpha=0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

# plt.text(x,y,stn)
plt.show()

####################################################
# 6: Weather Station Data: Clustering, Iteration 2 #
####################################################

sklearn.utils.check_random_state(1000)

Clus_dataSet = pdf[['xm', 'ym', 'Tx', 'Tm', 'Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# computing density based clusters
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# printing few select columns
print(pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5))

# for outliers, cluster label is -1
print(set(labels))

##########################################################
# 7: Visualizing Clusters Based on Location, Iteration 2 #
##########################################################

plt.rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, # min longitudes and latitudes
            urcrnrlon=ulon, urcrnrlat=ulat) # max longitudes and latitudes

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
# my_map.shadedrelief() # choosing not to render this fancy layer

# creating a colored map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4])) if clust_number == -1 else colors[np.int(clust_number)]
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color =c, marker='o',
                    s=20, alpha=0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

# plt.text(x,y,stn)
plt.show()

'''
























# in order to display plot within window
# plt.show()
