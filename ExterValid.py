import numpy as np 

def combinatorial(m,n):
	if n>=m-n:
		max_num=n
	else:
		max_num=m-n

	if n<=m-n:
		min_num=n
	else:
		min_num=m-n
	com=1
	i=m
	while i>max_num:
		com=com*i;
		i=i-1;
	i=2;
	while i<=min_num:
		com=com/i;
		i=i+1;
	return com 



def adj_rand( NUM_SAMPLE, NUM_CLASSES, nlabel, cluster_sample, num_of_clusters_sample ):

    
    ar=0; br=0; cr=0; dr=0;
    best = 0; best_adj = 0; 
    for i in range(NUM_SAMPLE):
        for j in range(i,NUM_SAMPLE):
            if nlabel[i]==nlabel[j] and cluster_sample[i]==cluster_sample[j]:
                ar=ar+1;
            elif nlabel[i]!=nlabel[j] and cluster_sample[i]==cluster_sample[j]:
                br=br+1;
            elif nlabel[i]==nlabel[j] and cluster_sample[i]!=cluster_sample[j]:
                cr=cr+1;
            elif nlabel[i]!=nlabel[j] and cluster_sample[i]!=cluster_sample[j]:
                dr=dr+1;

    rand = (ar+dr)/(ar+br+cr+dr);
    if rand > best:
        best = rand;
    
    if best >= 1.0:
        rand_adj = 1.0
        return rand, rand_adj

    u=np.zeros((1,NUM_CLASSES))[0];
    for i in range(NUM_SAMPLE):
        u[nlabel[i]- 1]=u[nlabel[i]- 1]+1;
  

    v=np.zeros((1,num_of_clusters_sample))[0];
    for i in range(NUM_SAMPLE):
        v[cluster_sample[i]-1]=v[cluster_sample[i]-1]+1;

    uv=np.zeros((NUM_CLASSES,num_of_clusters_sample));
    for i in range(NUM_SAMPLE):
        uv[nlabel[i]-1][cluster_sample[i]-1]=uv[nlabel[i]-1][cluster_sample[i]-1 ]+1;

    sum_of_u=0
    sum_of_v=0
    sum_of_uv=0

    for i in range(NUM_CLASSES):
        if u[i]>1:
            sum_of_u=sum_of_u+combinatorial(u[i],2);

    for i in range(num_of_clusters_sample):
        if v[i]>1:
            sum_of_v=sum_of_v+combinatorial(v[i],2);
  

    for i in range(NUM_CLASSES):
        for j in range(num_of_clusters_sample):
            if uv[i][j]>1:
                sum_of_uv=sum_of_uv+combinatorial(uv[i][j],2);
    rand_adj=(sum_of_uv-(sum_of_u*sum_of_v)/combinatorial(NUM_SAMPLE,2))/(0.5*(sum_of_u+sum_of_v)-(sum_of_u*sum_of_v)/combinatorial(NUM_SAMPLE,2));
    if best_adj<rand_adj:
        best_adj=rand_adj;
    return rand, rand_adj

def pn(label, grt):
    if label <= grt:
        p = label
        n = grt - label
    else:
        p = grt
        n = 0
    return -p/label , n/label
    
def accuracy (grt, label):
    label = [x - 1 for x in label]
    grt = [x - 1 for x in grt]
    label = np.bincount(label)
    grt = np.bincount(grt)
#    if debug:
#        print(label)
#       print(grt)
    result  = []
    for start_i in range(0, len(label)):
        total_p = 0
        grt_temp = list(grt)
        l = np.roll(label, start_i )
        for l_cluster in l:
#            if debug:
#             print(l_cluster)
            if len(grt_temp) != 0:
                p_n = [pn(l_cluster, x) for x in grt_temp]
                p_n = np.asarray(p_n)
                p_n = np.transpose(p_n)
                p  = p_n[0]
                n = p_n[1]
                remove_i = np.lexsort((p,n))[0]
#                if debug:
#                  print(remove_i)
                
                total_p = total_p + (pn(l_cluster, grt_temp[remove_i])[0] * l_cluster)
                #print(total_p)
                del grt_temp[remove_i]
        result.append(total_p / sum(label) )
    return -min(result)

def generate_report(data_file_name, label_file_name, result_file_name):
    import numpy as np
    import csv
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import jaccard_similarity_score
    from tkinter import Tk
    Tk().withdraw()
    

        
                        
                        
    ####
    data_peak = np.recfromcsv(data_file_name, delimiter = ',') # peak through data to see number of rows and cols
    
    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    
    
    with open(data_file_name) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index]= row[cols_index]
            row_index+=1
    ####
    data = np.transpose(data)
    data = data[0]
     ####       
    data_peak = np.recfromcsv(label_file_name, delimiter = ',')
    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    label = np.zeros([num_rows+1, num_cols])    
    
    with open(label_file_name) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                label[row_index][cols_index]= row[cols_index]
            row_index+=1
    ####
            
            
    result_all_k = []
    k_num = len(label[0])
    label=np.transpose(label)
    
    for k in range(k_num):
        result = []
        num_k = len(np.unique(label[k]))
        result.append(normalized_mutual_info_score(data, label[k]))
        result.append(adj_rand(len(data), len(np.unique(data)), data, label[k], len(np.unique(label[k])))[1])
        result.append(accuracy(data,label[k] ))
        result.append(jaccard_similarity_score(data, label[k]))
        
        result_all_k.append(result)
        
    result_all_k = np.transpose(result_all_k)
    header = [['k' + str(i) for i in range(2, num_k+1)]]
    att = [['','normalized mutual info score','adjusted rand', 'Accuracy multilabel', 'Jaccard']]
    result_all_k = np.concatenate((header,result_all_k), axis = 0)

    result_all_k= np.concatenate((result_all_k,np.transpose(att)), axis = 1)
    
    """
    with open(result_file_name, 'w') as text_file:
        for i in range(len(result_all_k)):
            k = i+2
            text_file.write("For k equals %s : \n"  % k)
            text_file.write("Silhouette score is %s \n" % result_all_k[i][0])
            text_file.write("db is %s \n\n " % result_all_k[i][1])
    """
         
    with open(result_file_name, 'w', newline='', encoding='utf-8') as text_file:
        csv_file= csv.writer(text_file)
    
        #result_all_k.insert(0, header)
        
        csv_file.writerows(result_all_k)
    
    """
    for i in range(len(labels)):
        temp = labels[i][0]
        print(temp)
        labels[i] = int(temp
    
    labels = labels.tolist()
    
    for i in range(len(labels)):
        temp = labels[i][0]
        labels[i] = int(temp)
    labels = np.asanyarray(labels)
    print(labels)
    """
    
    
    
    
    
