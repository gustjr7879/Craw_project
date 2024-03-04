import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
from sklearn.neighbors import NearestNeighbors



def change_data(data):
    result_list = []
    for i in range(len(data)):
        result_list.append([item.strip(" '[]") for item in data[i].split(',')])
    return result_list

def company_to_vector(data,model,vec_size):
    vectors_list = []
    for i in range(len(data)):
        vectors = [0 for i in range(vec_size)]
        cnt = 0
        for j in range(len(data[i])):
            if data[i][j] in model.wv.index_to_key:

                vectors += model.wv[data[i][j]]
                cnt += 1
            else:
                pass
        
        for k in range(len(vectors)):
            if cnt == 0:
                pass
            else:
                vectors[k]= vectors[k]/cnt
        vectors_list.append(vectors)
    return vectors_list


def draw_plotly_company_names(vectors_list, tsne,data):
    feat = tsne.fit_transform(np.array(vectors_list))
    cluster = KMeans(n_clusters=5)
    cluster.fit(vectors_list)
    y_kmeans = cluster.predict(vectors_list)
    fig = px.scatter(x = feat[:,0],y = feat[:,1],color=y_kmeans,text=data['기업명'],width=900, height=900)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=11,  # Set the font size here
            color="RebeccaPurple"
        )
    )
    return fig

def draw_plotly_skill_names(model, tsne):
    x_embed3 = tsne.fit_transform(model.wv.vectors)
    cluster = KMeans(n_clusters=3)
    cluster.fit(model.wv.vectors)
    #print(new_df['features'])
    y_kmeans = cluster.predict(model.wv.vectors)
    fig = px.scatter(x_embed3[:,0],x_embed3[:,1],color=y_kmeans,text=model.wv.index_to_key,width=800, height=800)
    return fig

def index_search(data,search_name,company_vector):
    neighbor = NearestNeighbors(n_neighbors=10,metric='cosine')
    neighbor.fit(company_vector)
    result_neig = neighbor.kneighbors(company_vector,return_distance=True)
    name_list = data['기업명']
    ind_list = []
    cnt = 0
    for i in name_list:
        if i == search_name:
            ind_list.append(cnt)
        cnt += 1
    nei_name = []
    for i in ind_list:
        for j in result_neig[1][i]:
            nei_name.append(j)
    nei_name = list(set(nei_name))
    return_name = []
    cnt = 0
    for i in nei_name:
        if cnt == 5:
            break
        elif data['기업명'][i] != search_name:
            return_name.append(data['기업명'][i])
            cnt += 1
    return return_name



def make_mean(inp, vec,model,data,vec_size):
    vectors = [0 for i in range(vec_size)]
    cnt = 0
    for i in inp:
        vector = model.wv[i]
        vectors += vector
        cnt += 1

    for k in range(len(vectors)):
        if cnt == 0:
            pass
        else:
            vectors[k]= vectors[k]/cnt  
    vec.append(vectors)
    neighbor = NearestNeighbors(n_neighbors=10,metric='cosine')
    neighbor.fit(vec)

    result_neig = neighbor.kneighbors(vec,return_distance=False)
    sim_vec = result_neig[-1]

    return_name = []
    cnt = 0
    for i in sim_vec:
        if i >= 170:
            pass
        else:
            if cnt == 5:
                break

            elif data['기업명'][i] not in return_name:
                return_name.append(data['기업명'][i])
                cnt += 1
    return return_name

def skill_search(search_name,model):
    neighbor = NearestNeighbors(n_neighbors=10,metric='cosine')
    neighbor.fit(model.wv.vectors)
    result_neig = neighbor.kneighbors(model.wv.vectors,return_distance=True)
    #print(result_neig)
    skill_name = model.wv.index_to_key
    ind_list = []
    cnt = 0
    for i in skill_name:
        if i == search_name:
            ind_list.append(cnt)
        cnt += 1
    nei_name = []
    for i in ind_list:
        for j in result_neig[1][i]:
            nei_name.append(j)
    nei_name = list(set(nei_name))
    return_name = []

    cnt = 0
    for i in nei_name:
        if cnt == 5:
            break
        elif skill_name[i] != search_name:
            return_name.append(skill_name[i])
            cnt += 1
    return return_name


df = pd.read_csv('./integrated_data.csv') #data load and preprocessing
