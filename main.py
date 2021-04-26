import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Tahap 1 : Inisialisasi
##membuat sebuah struktur dataframe

iris = datasets.load_iris()
print(iris["DESCR"])
print(iris)
df = pd.DataFrame({
    'x':iris.data[:,0],
    'y':iris.data[:,1],
    'cluster':iris.target
})
##print(df)
centroids = {}
for i in range(3):
    result_list=[]
    result_list.append(df.loc[df['cluster'] == i]['x'].mean())
    result_list.append(df.loc[df['cluster'] == i]['y'].mean())
    centroids[i] = result_list
##print(centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'], c=iris.target)
plt.xlabel('Sepal Length', fontsize = 18)
plt.xlabel('Sepal width', fontsize = 18)


##Menambahkan Color dan Plotting untuk semua cluster

colmap = {0:'r', 1:'g', 2:'b'}
for i in range(3):
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i])
plt.show()

##complete Graph
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'], c=iris.target, alpha=0.3)
colmap = {0:'r', 1:'g', 2:'b'}
col = [0,1]
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], c=colmap[i], edgecolor='k')
plt.show()

##Stage 2 : Assignment Stage
## Assignment function : Calculating distance dan meng-update dataframe

def assignment(df, centroids):
    for i in range(3):
        #sqrt((x1-x2)^2 + (y1-y2)^2)
        df['distance_from_{}'.format(i)]=(
            np.sqrt(
                (df['x'] - centroids[i][0]) **2
                + (df['y'] - centroids[i][1]) **2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest']=df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest']=df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color']=df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df)

###Visualisasi cluster dengan warna :

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')

plt.show()

#Tahap 3: Tahap Update

### Function Update: Memperbarui Centroids

def update (k):
    for i in range(3):
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k
centroids = update(centroids)
print(centroids)

### Visualisasi graph dengan centroids update terbaru

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'],df['y'], color=df['color'], alpha=0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')
plt.show()

## Ulangi tahap assignment untuk menandai ulang poin dengan cluster

df = assignment(df, centroids)

### Memvisualisasikan grafik yang diperbarui

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.3)
for i in centroids.keys():
    plt.scatter(centroids[i][0], centroids[i][1], color=colmap[i], edgecolor='k')
plt.show()

## Melanjutkan hingga semua cluster yang ditetapkan tidak berubah lagi

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

# Final Result :

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color= df['color'])
plt.title('Hasil Clustering K-Means')
for i in centroids.keys():
    plt.scatter(centroids[i][0], centroids[i][1], color=colmap[i], edgecolor='k')
plt.show()

from scipy.stats import mode
from sklearn.metrics import classification_report

labels = np.zeros_like(closest_centroids)
for i in range(3):
    mask = (closest_centroids == i)
    labels[mask] = mode(iris.target[mask])[0]
print(labels)
print(iris.target)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(iris.target, labels, target_names=target_names))
