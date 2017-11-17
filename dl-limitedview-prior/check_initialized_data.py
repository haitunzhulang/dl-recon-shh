import matplotlib.pyplot as plt
import glob
import pickle

#folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/v6_train/'
#folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/v6_test/'
folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/v6_testFinal/'
folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/vExperiment_4_64_1_train_AD/'
files=glob.glob(folder+'*.pkl')

imgs=[]

for i in range(len(files)):
	file=files[i]
	if i<10:
		with open(file,'rb') as f:
			data = pickle.load(f)
			imgs.append(data)
			
plt.ion()
fig=plt.figure()
for i in range(len(imgs)):
	image=imgs[i]['X']
	shp=image.shape
	image=image.reshape((shp[0],shp[1]))
	ax=fig.add_subplot(1,2,1)
	ax.imshow(image['X'])
	ax=fig.add_subplot(1,2,2)
	ax.imshow(image['Y'])
	plt.pause(2)
