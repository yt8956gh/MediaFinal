# MediaFinal

**執行環境**
---------------------------------------------

Python版本: Python 2.7 in Miniconda

Python套件要求:

	
	conda install tensorflow-gpu=1.9.0
	conda install keras
	conda install csv
	conda install pillow
	conda install numpy
	
GPU: Tesla K80

**執行流程**
---------------------------------------------

### step1: Preprocessing

將「/DeepFashion_package/img」中的影像，依照「/DeepFashion_package/Eval」中的標籤分為訓練與測試資料。
再把檔案讀入Numpy陣列中，存入/data資料夾。
(因為DeepFashion_package/img的檔案太大，無法上傳至Github，這裡不提供相關檔案。)

	
	python VGG-16_for_DeepFashion/preprocessing.py
	

### step2: VGG16 Training

使用keras的VGG-16模型，再加上Dense層進行訓練。

	
	python VGG-16_for_DeepFashion/VGG16_DeepFashion.py
	