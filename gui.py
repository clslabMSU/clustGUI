import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow , QTableWidgetItem, QAction, QApplication, QMdiArea,QListWidgetItem, QListWidget, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QTableWidget,QCheckBox,QLineEdit 
from PyQt5.QtGui import QIcon
import numpy as np
import csv
from tkinter.filedialog import askopenfilenames
from tkinter import Tk

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from sklearn.metrics import silhouette_score
import internal_validation

from fileOP import writeRows
from resultOP import table_result
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score
from ExterValid import accuracy

Tk().withdraw()




class GUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.data = []
        self.label = []
        self.label_b = []
        self.gt = []
        self.internal_index = []
        self.internal_index_b = []
        self.external_index = []
        
        self.data_dict = dict()
        self.label_dict = dict()
        self.label_b_dict = dict()
        self.gt_dict = dict()
        self.internal_dict = dict()
        self.internal_dict_b = dict()
        self.external_dict = dict()
        self.external_dict_b = dict()
        
        self.initUI()
        
        
    def initUI(self):
        self.setWindowTitle("Clust")               
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        #textEdit = QTextEdit()
        #self.setCentralWidget(textEdit)
        
        
        #Actions for View
        self.addDataAction =  QAction("Add data",self)
        self.addLabelAction = QAction("Add label",self)
        self.addGTAction =  QAction("Add ground truth",self)
        self.deleteAction = QAction("Delete everything", self
                                    )
        
        self.addDataAction.setShortcut('Ctrl+D')
        self.addDataAction.setStatusTip('Add new dataset')
        
        self.addLabelAction.setShortcut('Ctrl+L')
        self.addLabelAction.setStatusTip('Add new label')
        
        self.addGTAction.setShortcut('Ctrl+G')
        self.addGTAction.setStatusTip('Add ground truth')
        
        self.addDataAction.triggered.connect(self.addData)
        self.addLabelAction.triggered.connect(self.addLabel)
        self.addGTAction.triggered.connect(self.addGt)

        self.exitAction = QAction(QIcon('exit24.png'), 'Exit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)
        
        self.deleteAction.triggered.connect(self.delete)
      
        
        #Action for Tools
        
        self.clusterAction = QAction("Cluster data", self)
        self.validateAction =  QAction("Validate data", self)
        self.exvalidationAction = QAction("External validate", self)
        self.pcaAction = QAction("PCA plot", self)
        
        self.clusterAction.triggered.connect(self.clusterData)
        self.validateAction.triggered.connect(self.internal_validate_b)
        self.exvalidationAction.triggered.connect(self.external_validate)
        
        self.statusBar()

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.toolMenu = self.menubar.addMenu('&Tools')
        self.fileMenu.addAction(self.addDataAction)
        self.fileMenu.addAction(self.addLabelAction)
        self.fileMenu.addAction(self.addGTAction)
        self.fileMenu.addAction(self.exitAction)
        self.fileMenu.addAction(self.deleteAction)
        
        #toolbar = self.addToolBar('Exit')
        #toolbar.addAction(exitAction)
        
        self.ViewMenu = self.menubar.addMenu('&View')
        self.toolMenu.addAction(self.clusterAction)
        self.toolMenu.addAction(self.validateAction)
        self.toolMenu.addAction(self.exvalidationAction)
        
        self.showDataAction =  QAction("Show data",self)
        self.showLabelAction = QAction("Show label",self)
        self.showGTAction =  QAction("Show ground truth",self)
        self.showInternalA = QAction("Show internal validation", self)
        self.showExternalA = QAction("Show external validation", self)
        
        self.showDataAction.setShortcut('Ctrl+1')
        self.addDataAction.setStatusTip('View datasets')
        
        self.showLabelAction.setShortcut('Ctrl+2')
        
        self.showGTAction.setShortcut('Ctrl+3')
        self.addGTAction.setStatusTip('View current ground truth')
        
        self.showDataAction.triggered.connect(self.showData)
        self.showLabelAction.triggered.connect(self.showLabel)
        self.showGTAction.triggered.connect(self.showGT)
        self.showInternalA.triggered.connect(self.showInternal)
        self.showExternalA.triggered.connect(self.showExternal)
        
        self.ViewMenu.addAction(self.showDataAction)
        self.ViewMenu.addAction(self.showLabelAction)
        self.ViewMenu.addAction(self.showGTAction)
        self.ViewMenu.addAction(self.showInternalA)
        self.ViewMenu.addAction(self.showExternalA)
        
        
        self.exportLabelAction = QAction("Export label", self)
        self.exportLabelAction.triggered.connect(self.exportLabel)
        
        self.exportInternalAction = QAction("Export internal", self)
        self.exportInternalAction.triggered.connect(self.exportInternal)
        
        self.exportExternalAction = QAction("Export external", self)
        self.exportExternalAction.triggered.connect(self.exportExternal)
        
        self.ExportMenu = self.menubar.addMenu('&Export')
        self.ExportMenu.addAction(self.exportLabelAction)
        self.ExportMenu.addAction(self.exportInternalAction)
        self.ExportMenu.addAction(self.exportExternalAction)
        
        self.setGeometry(100, 100, 1200, 1000)
        self.setWindowTitle('Main window')    
        self.show()
    
    def insert_data(self, data_file_name, s):
        data_name = data_file_name.split('/')[-1]   .split('.')[0]
        data_peak = np.recfromcsv(data_file_name, delimiter = ',') # peak through data to see number of rows and cols
    
        num_cols = len(data_peak[0])
        num_rows = len(data_peak)
    
        new_data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
        with open(data_file_name) as csvfile:
            row_index = 0
            reader= csv.reader(csvfile)
            for row in reader:
                for cols_index in range(num_cols):
                    new_data[row_index][cols_index]= row[cols_index]
                row_index+=1
                   
        if s == "data":
            self.data.append(new_data)
            self.data_dict[len(self.data)] = [data_name,num_rows,num_cols,  "Imported"]
        elif s == "label":
            new_data = np.transpose(new_data)
            
            self.label_b.append(new_data)
            #print(len(label_b_dict))
            #print(label_b)
            self.label_b_dict[len(self.label_b_dict) + 1] = ["AF" + data_name ]
        elif s == "gt":
            self.gt.append(np.transpose(new_data))
            self.gt_dict[len(self.data)] = [data_name,num_rows,num_cols,  "Imported"]

            
    def delete(self):
        
        self.data = []
        self.label = []
        self.label_b = []
        self.gt = []
        self.internal_index = []
        self.internal_index_b = []
        self.external_index = []
        
        self.data_dict = dict()
        self.label_dict = dict()
        self.label_b_dict = dict()
        self.gt_dict = dict()
        self.internal_dict = dict()
        self.internal_dict_b = dict()
        self.external_dict = dict()
        self.external_dict_b = dict()
        
    def addData(self):
        addr_list = askopenfilenames()
        for i in addr_list:
            self.insert_data(i, s = "data")
    
    def addLabel(self):
        addr_list = askopenfilenames()
        for i in addr_list:
            self.insert_data(i, s = "label")
    def addGt(self):
        addr_list = askopenfilenames()
        for i in addr_list:
            self.insert_data(i, s = "gt")    
        
    
    def showData(self):
        d_view = QTableWidget()
        h = ["ID", "Name","Rows", "Columns", "Note" ]
        d_view.setRowCount(len(self.data_dict))
        d_view.setColumnCount(5)
        d_view.setHorizontalHeaderLabels(h)
        d_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        d_view.setWindowTitle("Data view")
        for i in range(1, len(self.data_dict) + 1 ):
            for c in range(5):
                item = QTableWidgetItem()
                if c == 0:
                    item.setText(str(i))
                else:
                    item.setText(str(self.data_dict[i][c-1]))
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                d_view.setItem(i-1, c, item)
        self.mdi.addSubWindow(d_view)
        self.mdi.cascadeSubWindows()
        d_view.show()
                                         
    def showLabel(self):
        
        d_view = QTableWidget()
        h = ["ID", "Name","Rows", "Columns", "Note" ]
        d_view.setRowCount(len(self. label_dict))
        d_view.setColumnCount(5)
        d_view.setHorizontalHeaderLabels(h)
        d_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        d_view.setWindowTitle("Label view")
        for i in range(1, len(self. label_dict) + 1 ):
            for c in range(5):
                item = QTableWidgetItem()
                if c ==0:
                    item.setText(str(i))
                else:
                    item.setText(str(self. label_dict[i][c-1]))
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                d_view.setItem(i-1, c, item)
        self.mdi.addSubWindow(d_view)
        self.mdi.cascadeSubWindows()
        d_view.show()
        
    def showGT(self):
        d_view = QTableWidget()
        h = ["ID", "Name","Rows", "Columns", "Note" ]
        d_view.setRowCount(len(self. data))
        d_view.setColumnCount(5)
        d_view.setHorizontalHeaderLabels(h)
        d_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        d_view.setWindowTitle("Ground truth view")
        for i in range(1, len(self. gt_dict) + 1 ):
            for c in range(5):
                item = QTableWidgetItem()
                if c ==0:
                    item.setText(str(i))
                else:
                    item.setText(str(self. label_dict[i][c-1]))
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                d_view.setItem(i-1, c, item)
        self.mdi.addSubWindow(d_view)
        self.mdi.cascadeSubWindows()
        d_view.show()
    
    
    def showInternal(self):
        i_view = QTableWidget()
        h = ["Dataset", "Clustering algorithm", "Silhouette", "DB", "Xie bienie", "Dunn", "CH" ]
        i_view.setRowCount(len(self.internal_dict))
        i_view.setColumnCount(7)
        i_view.setHorizontalHeaderLabels(h)
        i_view.setWindowTitle("Internal Validation view")
        for i in range(1, len(self.internal_dict) + 1 ):
            for c in range(7):
                item = QTableWidgetItem()
                item.setText(str(self.internal_dict[i][c]))
                i_view.setItem(i-1, c, item)
        self.mdi.addSubWindow(i_view)
        self.mdi.cascadeSubWindows()
        i_view.show()
                
    def showExternal(self):
        i_view = QTableWidget()
        h = ["Dataset", "Clustering algorithm", "NMI", "AR", "Accuracy", "Jaccard" ]
        i_view.setRowCount(len(self.external_dict))
        i_view.setColumnCount(6)
        i_view.setHorizontalHeaderLabels(h)
        i_view.setWindowTitle("External Validation view")
        for i in range(1, len(self.external_dict) + 1 ):
            for c in range(6):
                item = QTableWidgetItem()
                item.setText(str(self.external_dict[i][c]))
                i_view.setItem(i-1, c, item)
        self.mdi.addSubWindow(i_view)
        self.mdi.cascadeSubWindows()
        i_view.show()
    
    def clusterData(self):
        temp = QVBoxLayout()
        temp2 = QHBoxLayout()
        self.clusterLayout = QWidget()
        self.d_view_clust = QListWidget()
        self.clust_push =  QPushButton("Cluster")
        self.clust_push.clicked.connect(self.cluster)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.d_view_clust.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.d_view_clust.setWindowTitle("Data view")
        for i in range(1, len(self.data_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.data_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.d_view_clust.addItem(item)
        self.kmeans =QCheckBox("Kmeans")
        self.average = QCheckBox("Average")
        self.complete = QCheckBox("Complete")
        self.ward =  QCheckBox("Ward")
        self.spectral = QCheckBox("Spectral")
        self.clust_text =  QLineEdit()
        self.clust_text.resize(280,40)
        self.clust_text.setPlaceholderText("Nummber of K")
        temp2.addWidget(self.kmeans)
        temp2.addWidget(self.average)
        temp2.addWidget(self.complete)
        temp2.addWidget(self.ward)
        temp2.addWidget(self.spectral)
        temp.addStretch(1)
        temp.addWidget(self.d_view_clust)
        temp.addLayout(temp2)
        temp.addWidget(self.clust_text)
        temp.addWidget(self.clust_push)
        self.clusterLayout.setLayout(temp)
        self.mdi.addSubWindow(self.clusterLayout)
        self.mdi.cascadeSubWindows()
        self.clusterLayout.show()
    
    def cluster(self):
        selected_data = []
        for i in self.d_view_clust.selectedIndexes():
            selected_data.append(i.row())
        if len(self.d_view_clust.selectedIndexes()) == 0:
            print("EMPTY")
        else:
            for data_i in selected_data:
                if self.kmeans.isChecked():
                    result  = []
                    for i in range(2, int(self.clust_text.text()) +1):
                        estimator = KMeans(init='k-means++', n_clusters = i, n_init=10, max_iter = 1000)
                        estimator.fit(self.data[data_i])
                        result.append([x  +  1  for x in estimator.labels_ ])
                    self.label.extend(result)
                    self.label_b.append(result)
                    for i in result:
                        self.label_dict[len(self.label_dict) + 1] = ["Kmeans " + str(len(np.unique(i)))+ " of dataset " + (self.data_dict[data_i+1][0]), len(self.data[data_i]),1, "ID " + str(data_i)]
                    self.label_b_dict[len(self.label_b_dict) + 1] = ["Kmeans" + self.clust_text.text() + self.data_dict[data_i+1][0] ]
                if self.average.isChecked():
                    result  = []
                    for i in range(2, int(self.clust_text.text()) +1):
                        estimator = AgglomerativeClustering(linkage='average', n_clusters = i)
                        estimator.fit(self.data[data_i])
                        result.append([x  +  1  for x in estimator.labels_ ])
                    self.label.extend(result)
                    self.label_b.append(result)

                    for i in result:
                        self.label_dict[len(self.label_dict) + 1] = ["Average " + str(len(np.unique(i)))+ " of dataset " + (self.data_dict[data_i+1][0]) ,len(self.data[data_i]),1, "ID " + str(data_i) ]
                    self.label_b_dict[len(self.label_b_dict) + 1] = ["Average" + self.clust_text.text() + self.data_dict[data_i+1][0] ]
                if self.complete.isChecked():
                    result  = []
                    for i in range(2, int(self.clust_text.text()) +1):
                        estimator = AgglomerativeClustering(linkage='complete', n_clusters = i)
                        estimator.fit(self.data[data_i])
                        result.append([x  +  1  for x in estimator.labels_ ])
                    self.label.extend(result)
                    self.label_b.append(result)
                    for i in result:
                        self.label_dict[len(self.label_dict) + 1] = ["Complete " + str(len(np.unique(i)))+ " of dataset " + (self.data_dict[data_i+1][0]) ,len(self.data[data_i]),1, "ID " + str(data_i) ]
                    self.label_b_dict[len(self.label_b_dict) + 1] = ["Complete" + self.clust_text.text() + self.data_dict[data_i+1][0] ]
                if self.ward.isChecked():
                    result  = []
                    for i in range(2, int(self.clust_text.text()) +1):
                        estimator = AgglomerativeClustering(linkage='ward', n_clusters = i)
                        estimator.fit(self.data[data_i])
                        result.append([x  +  1  for x in estimator.labels_ ])
                    self.label.extend(result)
                    self.label_b.append(result)
                    for i in result:
                        self.label_dict[len(self.label_dict) + 1] = ["Ward " + str(len(np.unique(i)))+ " of dataset " + (self.data_dict[data_i+1][0]) ,len(self.data[data_i]),1, "ID " + str(data_i) ]
                    self.label_b_dict[len(self.label_b_dict) + 1] = ["Ward" + self.clust_text.text() + self.data_dict[data_i+1][0] ]
                if self.spectral.isChecked():
                    result  = []
                    for i in range(2, int(self.clust_text.text()) +1):
                        estimator = SpectralClustering(n_clusters = i, affinity = "nearest_neighbors", n_neighbors= 15, n_init = 100 )
                        estimator.fit(self.data[data_i])
                        result.append([x  +  1  for x in estimator.labels_ ])
                    self.label.extend(result)
                    self.label_b.append(result)
                    for i in result:
                        self.label_dict[len(self.label_dict) + 1] = ["Spectral " + str(len(np.unique(i)))+ " of dataset " + (self.data_dict[data_i+1][0]) ,len(self.data[data_i]),1, "ID " + str(data_i) ]
                    self.label_b_dict[len(self.label_b_dict) + 1] = ["Spectral" + self.clust_text.text() + self.data_dict[data_i+1][0] ]
                    
    def internal_validate(self):
        temp = QVBoxLayout()
        temp2 = QHBoxLayout()
        self.internalLayout = QWidget()
        self.d_view_internal = QListWidget()
        self.internal_push =  QPushButton("Cluster")
        self.internal_push.clicked.connect(self.internal)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.d_view_internal.setWindowTitle("Data view")
        for i in range(1, len(self.data_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.data_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.d_view_internal.addItem(item)        
        self.sil =QCheckBox("Silhouette")
        self.db = QCheckBox("Db")
        self.xb = QCheckBox("Xie_biene")
        self.dunn =  QCheckBox("Dunn")
        self.ch = QCheckBox("CH")
        
        temp2.addWidget(self.sil)
        temp2.addWidget(self.db)
        temp2.addWidget(self.xb)
        temp2.addWidget(self.dunn)
        temp2.addWidget(self.ch)
        
        temp.addStretch(1)
        temp.addWidget(self.d_view_internal)
        
        self.l_view_internal = QListWidget()
        self.l_view_internal.setWindowTitle("Label view")
        self.l_view_internal.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        
        
        for i in range(1, len(self.label_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.label_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.l_view_internal.addItem(item)
            
        temp.addLayout(temp2)
        temp.addWidget(self.l_view_internal)
        temp.addWidget(self.internal_push)
        self.internalLayout.setLayout(temp)
        self.mdi.addSubWindow(self.internalLayout)
        self.mdi.cascadeSubWindows()
        self.internalLayout.show()
    
    def internal(self):
        data_validated = self.data[self.d_view_internal.selectedIndexes()[0].row()]
        label_validated_index = []
        for i in self.l_view_internal.selectedIndexes():
            label_validated_index.append(i.row())
        for i in label_validated_index:
            result = []
            label_validated = self.label[i]
            num_k = np.unique(label_validated)
            
            inter_index = internal_validation.internalIndex(len(num_k))
            
            if self.sil.isChecked():
                result.append(silhouette_score(data_validated, label_validated, metric = 'euclidean'))
            else:
                result.append("NA")
            if self.db.isChecked():
                result.append(inter_index.dbi(data_validated, label_validated))
            else:
                result.append("NA")
            if self.xb.isChecked():
                result.append(inter_index.xie_benie(data_validated, label_validated))
            else:
                result.append("NA")
            if self.dunn.isChecked():
                result.append(inter_index.dunn(data_validated, label_validated))
            else:
                result.append("NA")
            if self.ch.isChecked():
                result.append(inter_index.CH(data_validated, label_validated))
            else:
                result.append("NA")
           
            self.internal_index.append(result)
            temp = [self.data_dict[self.d_view_internal.selectedIndexes()[0].row() + 1][0], self.label_dict[i+1][0]]
            
            temp.extend(result)
            self.internal_dict[len(self.internal_dict) + 1 ] = temp
                          
    def internal_validate_b(self):
        temp = QVBoxLayout()
        temp2 = QHBoxLayout()
        self.internalLayout = QWidget()
        self.d_view_internal = QListWidget()
        self.internal_push =  QPushButton("Cluster")
        self.internal_push.clicked.connect(self.internal_b)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.d_view_internal.setWindowTitle("Data view")
        for i in range(1, len(self.data_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.data_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.d_view_internal.addItem(item)        
        self.sil =QCheckBox("Silhouette")
        self.db = QCheckBox("Db")
        self.xb = QCheckBox("Xie_biene")
        self.dunn =  QCheckBox("Dunn")
        self.ch = QCheckBox("CH")
        
        temp2.addWidget(self.sil)
        temp2.addWidget(self.db)
        temp2.addWidget(self.xb)
        temp2.addWidget(self.dunn)
        temp2.addWidget(self.ch)
        
        temp.addStretch(1)
        temp.addWidget(self.d_view_internal)
        
        self.l_view_internal = QListWidget()
        self.l_view_internal.setWindowTitle("Label view")
        self.l_view_internal.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        
        
        for i in range(1, len(self.label_b_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.label_b_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.l_view_internal.addItem(item)
            
        temp.addLayout(temp2)
        temp.addWidget(self.l_view_internal)
        temp.addWidget(self.internal_push)
        self.internalLayout.setLayout(temp)
        self.mdi.addSubWindow(self.internalLayout)
        self.mdi.cascadeSubWindows()
        self.internalLayout.show()
    
    def internal_b(self):
        data_validated = self.data[self.d_view_internal.selectedIndexes()[0].row()]
        label_validated_index = []
        for i in self.l_view_internal.selectedIndexes():
            label_validated_index.append(i.row())
        for i in label_validated_index:
            
            labels_validated = self.label_b[i]
            result_over_k = []
            for label_validated in labels_validated:
                result = []
                num_k = np.unique(label_validated)
                
                inter_index = internal_validation.internalIndex(len(num_k))
                
                if self.sil.isChecked():
                    result.append(silhouette_score(data_validated, label_validated, metric = 'euclidean'))
                else:
                    result.append("NA")
                if self.db.isChecked():
                    result.append(inter_index.dbi(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.xb.isChecked():
                    result.append(inter_index.xie_benie(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.dunn.isChecked():
                    result.append(inter_index.dunn(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.ch.isChecked():
                    result.append(inter_index.CH(data_validated, label_validated))
                else:
                    result.append("NA")
                result_over_k.append(result)
                temp = [self.data_dict[self.d_view_internal.selectedIndexes()[0].row() + 1][0], self.label_b_dict[i+1][0]+ str(len(num_k))]            
                temp.extend(result)
                self.internal_dict[len(self.internal_dict) + 1 ] = temp
            self.internal_index.append(result_over_k)
            self.internal_dict_b[len(self.internal_dict_b) + 1] = self.label_b_dict[i+1][0]
        print(self.internal_index)
        
    def external_validate(self):
        temp = QVBoxLayout()
        temp2 = QHBoxLayout()
        self.externalLayout = QWidget()
        self.d_view_external = QListWidget()
        self.external_push =  QPushButton("Cluster")
        self.external_push.clicked.connect(self.external_b)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.d_view_external.setWindowTitle("Data view")
        for i in range(1, len(self.gt_dict)  + 1):
            item = QListWidgetItem()
            item.setText(str(self.gt_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.d_view_external.addItem(item)        
        self.nmi =QCheckBox("NMI")
        self.adjr = QCheckBox("Adjusted Rand")
        self.accu = QCheckBox("Accuracy")
        self.jacc =  QCheckBox("Jaccard")
        
        temp2.addWidget(self.nmi)
        temp2.addWidget(self.adjr)
        temp2.addWidget(self.accu)
        temp2.addWidget(self.jacc)
        
        temp.addStretch(1)
        temp.addWidget(self.d_view_external)
        
        self.l_view_external = QListWidget()
        self.l_view_external.setWindowTitle("Label view")
        self.l_view_external.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        
        
        for i in range(1, len(self.label_b_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.label_b_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.l_view_external.addItem(item)
            
        temp.addLayout(temp2)
        temp.addWidget(self.l_view_external)
        temp.addWidget(self.external_push)
        self.externalLayout.setLayout(temp)
        self.mdi.addSubWindow(self.externalLayout)
        self.mdi.cascadeSubWindows()
        self.externalLayout.show()

    def external_b(self):
        data_validated = self.gt[self.d_view_external.selectedIndexes()[0].row()][0]
        label_validated_index = []
        for i in self.l_view_external.selectedIndexes():
            label_validated_index.append(i.row())
        print("DONE")
        for i in label_validated_index:
            
            labels_validated = self.label_b[i]
            result_over_k = []
            for label_validated in labels_validated:
                result = []
                num_k = np.unique(label_validated)
                print(len(data_validated))
                print(len(label_validated))
                
                if self.nmi.isChecked():
                    result.append(normalized_mutual_info_score(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.adjr.isChecked():
                    result.append(adjusted_rand_score(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.accu.isChecked():
                    result.append(accuracy(data_validated, label_validated))
                else:
                    result.append("NA")
                if self.jacc.isChecked():
                    result.append(jaccard_similarity_score(data_validated, label_validated))
                else:
                    result.append("NA")
                result_over_k.append(result)
                temp = [self.gt_dict[self.d_view_external.selectedIndexes()[0].row() + 1][0], self.label_b_dict[i+1][0]+ str(len(num_k))]            
                temp.extend(result)
                self.external_dict[len(self.external_dict) + 1 ] = temp
            self.external_index.append(result_over_k)
            self.external_dict_b[len(self.external_dict_b) + 1 ] = self.label_b_dict[i+1][0]
            print("DONE")
        
#        
    def exportInternal(self):
        temp = QVBoxLayout()
        self.exportInternalLayout = QWidget()
        self.internal_view_export = QListWidget()
        self.internal_export_push =  QPushButton("Export")
        self.internal_export_push.clicked.connect(self._exportInternal)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.internal_view_export.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.internal_view_export.setWindowTitle("Data view")
        for i in range(1, len(self.internal_dict_b) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.internal_dict_b[i]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.internal_view_export.addItem(item)
        temp.addStretch(1)
        temp.addWidget(self.internal_view_export)
        temp.addWidget(self.internal_export_push)
        self.exportInternalLayout.setLayout(temp)
        self.mdi.addSubWindow(self.exportInternalLayout)
        self.mdi.cascadeSubWindows()
        self.exportInternalLayout.show()
    
    def _exportInternal(self):
        selected_data = []
        for i in self.internal_view_export.selectedIndexes():
            selected_data.append(i.row())
        name_to_export = [self.internal_dict_b[i+1] + 'internal.csv' for i in selected_data]
        internal_to_export = [self.internal_index[i] for i in selected_data]
        if len(self.internal_view_export.selectedIndexes()) == 0:
            print("EMPTY")
        else:
            for i in range(len(selected_data)):
                to_export = np.transpose(internal_to_export[i])
                print(to_export)
                to_export = table_result(to_export,[['k' + str(i) for i in range(2, len(to_export[0]) + 2 )]] ,[['','Sil', 'Db', 'Xb', 'Dunn', 'CH']] )
                
                writeRows(name_to_export[i] , to_export)

    def exportExternal(self):
        temp = QVBoxLayout()
        self.exportExternalLayout = QWidget()
        self.external_view_export = QListWidget()
        self.external_export_push =  QPushButton("Export")
        self.external_export_push.clicked.connect(self._exportExternal)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.external_view_export.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.external_view_export.setWindowTitle("Data view")
        for i in range(1, len(self.external_dict_b) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.external_dict_b[i]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.external_view_export.addItem(item)
        temp.addStretch(1)
        temp.addWidget(self.external_view_export)
        temp.addWidget(self.external_export_push)
        self.exportExternalLayout.setLayout(temp)
        self.mdi.addSubWindow(self.exportExternalLayout)
        self.mdi.cascadeSubWindows()
        self.exportExternalLayout.show()
    
    def _exportExternal(self):
        selected_data = []
        for i in self.external_view_export.selectedIndexes():
            selected_data.append(i.row())
        print(selected_data)
        name_to_export = [self.external_dict_b[i+1] + 'external.csv' for i in selected_data]
        print(name_to_export)
        external_to_export = [self.external_index[i] for i in selected_data]
        print(external_to_export)
        if len(self.external_view_export.selectedIndexes()) == 0:
            print("EMPTY")
        else:
            for i in range(len(selected_data)):
                to_export = np.transpose(external_to_export[i])
                print(to_export)
                to_export = table_result(to_export,[['k' + str(i) for i in range(2, len(to_export[0]) + 2 )]] ,[['','NMI', 'Adjusted Rand', "Accuracy", "Jaccard"]] )
                
                writeRows(name_to_export[i] , to_export)
                
    def exportLabel(self):
        temp = QVBoxLayout()
        self.exportLabelLayout = QWidget()
        self.label_view_export = QListWidget()
        self.label_export_push =  QPushButton("Export")
        self.label_export_push.clicked.connect(self._exportLabel)
        
#        h = ["ID", "Name","Rows", "Columns", "Note" ]
#        d_view.setRowCount(len(data))
#        d_view.setColumnCount(5)
#        d_view.setHorizontalHeaderLabels(h)
        self.label_view_export.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.label_view_export.setWindowTitle("Data view")
        for i in range(1, len(self.label_b_dict) + 1 ):
            item = QListWidgetItem()
            item.setText(str(self.label_b_dict[i][0]))
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.label_view_export.addItem(item)
        temp.addStretch(1)
        temp.addWidget(self.label_view_export)
        temp.addWidget(self.label_export_push)
        self.exportLabelLayout.setLayout(temp)
        self.mdi.addSubWindow(self.exportLabelLayout)
        self.mdi.cascadeSubWindows()
        self.exportLabelLayout.show()
    
    def _exportLabel(self):
        selected_data = []
        for i in self.label_view_export.selectedIndexes():
            selected_data.append(i.row())
        name_to_export = [self.label_b_dict[i+1][0] + '.csv' for i in selected_data]
        print(name_to_export)
        label_to_export = [self.label_b[i] for i in selected_data]
        print(label_to_export)
        if len(self.label_view_export.selectedIndexes()) == 0:
            print("EMPTY")
        else:
            for i in range(len(selected_data)):
                writeRows(name_to_export[i], np.transpose(label_to_export[i]))
                
                

        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())