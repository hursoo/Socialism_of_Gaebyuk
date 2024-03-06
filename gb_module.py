# gb_module.py

# 주요 패키지 불러오기

import pandas as pd
import numpy as np
import csv

from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from adjustText import adjust_text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gb_module

# 그래프에서 한글 표시를 위한 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕으로 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 정상 표시되도록 설정

# 그래프 해상도 설정
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi' : '100', 'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})

#################################################

def combine_to_basic_df(sent, r, ho):
    '''
    # 개벽 데이터 정보 결합 -> 분석의 기본 df 생성
    '''
    sent_r = pd.merge(sent, r, left_on = 'r_no', right_on = 'r_id', how = 'inner')
    sent_rho = pd.merge(sent_r, ho, left_on = 'ho_no', right_on = 'ho_id', how = 'inner')

    gb_df = sent_rho[['sent_id',  'sent_raw',  'sent_split',  'r_no',  'title',  'writer',  'w_new',  'ho_no',  'year',  'month',  'grid_1']]
    return gb_df



def get_dtm(df, col_name, stopw, rank_n): # rank_n : 고빈도 단어 n 순위까지
    '''
    # 문서-단어 행렬(dtm) 산출 함수
    '''

    # 단어 종류 모두 벡터화. 2음절 이상
    tv = TfidfVectorizer(stop_words=stopw, norm=None)
    dtm = tv.fit_transform(df[col_name])

    # df 형태로 표시
    dtm_df = pd.DataFrame(dtm.toarray(), columns=tv.get_feature_names_out(), index=df.index)

    highword_list = dtm_df.sum().sort_values(ascending=False)[:rank_n].index.to_list()
    feature_df = dtm_df[highword_list] # 열 순서는 tfidf값이 높은 것부터 낮은 순으로 정렬
    return feature_df
    

def get_top_n_grams(corpus, n=None):
    """
    2-gram의 빈도를 기반으로 상위 n개의 2-gram을 반환합니다.
    """
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_dtm_ngram(df, col_name, stopw, rank_n):
    '''
    # ngram 단어 대상의, 문서-단어 행렬(dtm) 산출 함수
    '''
    # 2-gram의 빈도를 기반으로 상위 100개 2-gram 추출
    top_100_n_grams = get_top_n_grams(df[col_name], n=100)
    top_100_n_grams = [n_gram for n_gram, freq in top_100_n_grams]

    # 2-gram 벡터화에 상위 100개 2-gram을 사용
    tv = TfidfVectorizer(stop_words=stopw, norm=None, token_pattern=r"(?u)\b\w+\b", ngram_range=(2, 2), vocabulary=top_100_n_grams)
    dtm = tv.fit_transform(df[col_name])

    # df 형태로 표시하되 상위 50개 2-gram만 포함
    dtm_df = pd.DataFrame(dtm.toarray(), columns=tv.get_feature_names_out(), index=df.index)
    top_50_tfidf = dtm_df.sum().sort_values(ascending=False).head(50).index
    dtm_df = dtm_df[top_50_tfidf]

    return dtm_df


def transform_to_gtm(df, grid_col, dtm): # 매개변수: df(gb_df), 구간 정보 열(문자열 형태로), dtm50_df
    '''
    # 문서-단어 행렬(dtm)을 구간-단어 행렬(gtm)으로 변환하기
    '''
    # 구간정보만 df로 추출
    grid = df[[grid_col]]
    # 구간 정보 결합하고, 구간을 index로 만듦
    temp_dtm = pd.concat([dtm, grid], axis=1)
    grid_dtm = temp_dtm.set_index(grid_col)
    # 구간별 평균
    gtm = grid_dtm.groupby(grid_col).mean()
    return gtm


def get_cossim(dtm): # 매개변수: docu-term-matrix
    '''
    # 코사인유사도 산출 함수
    '''
    idx = dtm.index.tolist()
    cossim_v = cosine_similarity(dtm, dtm) # 인덱스(행) 간의 관계가 산출
    cossim = pd.DataFrame(cossim_v, columns=idx, index=idx)
    return cossim


def hierarchical_clustering(grid_sim_matrix, num_clusters=2):
    '''
    계층적 군집화 관련 함수
    '''
    # 주어진 코사인 유사도 행렬을 NumPy 배열로 변환
    data = grid_sim_matrix.values
    data_labels = grid_sim_matrix.index

    # 계층적 클러스터링을 수행
    linkage_matrix = linkage(data, method='ward')  # 'ward' 연결 방법 사용

    # 덴드로그램 그리기 전에 색깔 구분을 위한 color_threshold 계산
    threshold = linkage_matrix[-(num_clusters-1), 2] if num_clusters > 1 else 0
    
    # 사용할 색상 리스트 정의
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    plt.figure(figsize=(7, 4))
    dendrogram(linkage_matrix, labels=data_labels, orientation='right', color_threshold=threshold)
    plt.xlabel('Distance')
    plt.ylabel('QRTs')
    plt.title('Hierarchical Clustering Dendrogram')

    # 지정된 군집 수로 데이터 군집화
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # 군집 결과 출력
    print(f"{num_clusters}개의 군집 결과:")
    for qrt, cluster_id in zip(data_labels, clusters):
        print(f'{qrt}: Cluster {cluster_id}')

    # Seaborn 스타일 설정
    sns.set(style='whitegrid')

    # 산점도 그리기
    plt.figure(figsize=(7, 4))
    scatter_plot_data = [data[clusters == i] for i in range(1, num_clusters + 1)]

    for i in range(num_clusters):
        sns.scatterplot(x=scatter_plot_data[i][:, 0], y=scatter_plot_data[i][:, 1],
                        label=f'Cluster {i+1}', color=colors[i % len(colors)],
                        marker='o', edgecolor='black')

    # 그래프 레이블과 제목 설정
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Clustered Data')

    # 군집 레이블 표시
    for i, label in enumerate(data_labels):
        plt.annotate(label, (data[i, 0], data[i, 1]))

    # 범례 표시
    plt.legend()

    # 그래프 출력
    plt.show()
    
    
def draw_graph(gtm, *args): # args는 a와 b 값을 포함하는 리스트
    '''
    # 주요 단어(특성)들의 구간별 빈도 변동 그래프
    '''
    if len(args) == 2:
        a, b = args
        df = gtm.iloc[:, a:b]

        # 그래프 그리기
        plt.figure(figsize=(7, 4))  # 그래프 크기 조정

        for column in df.columns:
            plt.plot(df.index, df[column], label=column, marker='o')

        # 그래프 제목과 레이블 설정
        plt.title('개벽 특성들의 동향')
        plt.xlabel('grid')
        plt.ylabel('값')

        # 범례 추가
        plt.legend()

        # 그래프 표시
        plt.grid(True)  # 격자 표시
        plt.show()
        
    else:
            print("올바른 인자 개수가 아닙니다. a와 b를 포함하는 범위를 (gtm, a, b)처럼 입력하세요.")        


def matrix_to_edge(df):
    '''
    # 특성간 유사도 matrix 형식을 -> edge 형식으로 
     (동일 단어 간 관계는 삭제: nc.exe 입력값에 맞게)
    '''
    # df : 특성 유사도 df
    df1 = df.unstack().to_frame()
    df1.columns = ['cossim']
    hetero_index = [(m,n) for m, n in df1.index if (m != n)]
    hetero_df = df1.loc[hetero_index]
    return hetero_df


def remove_same_pairwords(df):
    '''
    # pfnet .. 파일에서 순서쌍 구성요소가 동일한 행을 제거하는 함수
    '''
    df_copy = df.copy()  # 원본 DataFrame을 변경하지 않기 위해 복사본을 만듭니다.
    
    # 'node1'과 'node2' 열을 정렬하여 동일한 조합으로 처리한 새로운 열 'sorted_words'를 생성합니다.
    df_copy['sorted_words'] = df_copy[['node1', 'node2']].apply(sorted, axis=1)
    
    # 중복된 행을 제거합니다.
    df_copy = df_copy.drop_duplicates(subset='sorted_words', keep='first')

    # 'sorted_words' 열을 제거하고 인덱스를 재설정합니다.
    df_copy = df_copy.drop(columns='sorted_words').reset_index(drop=True)

    if len(df) == len(df_copy):
        print("No duplicate rows found.")

    else:
        return df_copy


def transfer_to_grade(df, col_name, grade_no):
    '''
    # 코사인유사도를 5등분 값으로 변환
    '''
    # col_name : 'weight'
    # 등급(grade) 간 격차를 0.5로 설정
    df[col_name] = pd.qcut(df[col_name], q=grade_no, labels=list(np.arange(1, 1+grade_no/2, 0.5)))
    return df


def get_gp_ctrl(pnnc, wcent, nc):
    '''
    # 그룹 및 중심성에 필요한 지표만 선별
    '''
    pnnc = pnnc.iloc[:,[0,2]]
    pnnc.columns = ['node', 'group']
    
    wcent = wcent.iloc[:,[1,3]]
    wcent.columns = ['node', 'rtbc']
    
    nc.columns = ['node', 'nc']
    
    temp = pd.merge(pnnc, wcent, how = 'inner', on = 'node')
    gp_ctrl = pd.merge(temp, nc, how = 'inner', on = 'node')

    return gp_ctrl


def generate_node_attributes(link_df, node_df, col_name, grade_no):
    '''
    # 노드 그룹 및 중심성 입력값 산출 함수
    '''
    # 그룹별 nc 최고값 산출
    group_max_nc = node_df.sort_values(by=['group', 'nc'], ascending=[True, False]) \
                          .groupby('group') \
                          .agg(max_nc = ('nc', 'max')) \
                          ['max_nc'] \
                          .to_list()

    # 최고값에 해당하는 단어에 Sphere, 10 할당
    group_headword = node_df.query('nc in @group_max_nc') \
                            [['node']] \
                            .assign(shape = 'Sphere', size=10)

    # 전역 중심성 1~4위 단어 찾아서 해당 단어에 Square, 10 할당
    global_headword = node_df.sort_values('rtbc', ascending=False) \
                             .head(4) \
                             [['node']] \
                             .assign(shape = 'Square', size=10)

    # 두 df에 중복된 이름이 있는 경우, 각 df의 다른 특정 열을 'Solid Square'로 입력
    dupl = pd.merge(group_headword, global_headword, how='inner', on='node')['node'].tolist()
    group_headword.loc[group_headword['node'].isin(dupl), 'shape'] = 'Solid Square'
    global_headword.loc[global_headword['node'].isin(dupl), 'shape'] = 'Solid Square'

    # 두 df를 합치고 중복 단어는 하나만 남김
    ctrl_word = pd.concat([group_headword, global_headword]).drop_duplicates(['node'])

    # 입력용 자료에 재결합
    ctrl_word1 = ctrl_word.set_index('node', drop = True)

    link = link_df[['node1', 'node2']]
    sorted_word = []
    for i in range(len(link)):
        pairs = link.iloc[i].values
        for j in pairs:
            if j not in sorted_word:
                sorted_word.append(j)
    
    node_attr = ctrl_word1.reindex(sorted_word)
    
    # 중심단어 이외의 단어에는 disk, 3을 부여
    node_attr['shape'] = node_attr['shape'].fillna('disk')
    node_attr['size'] = node_attr['size'].fillna(3)

    # group_option
    color = ['red', 'blue', 'green', 'gray', 'black',\
             'aqua', 'Lime', 'Olive', 'pink', '255, 128, 128',\
             '128, 128, 255', 'Purple', '255, 128, 0']

    group_num = node_df['group'].drop_duplicates()
    input_color = []
    for i in group_num:
        input_color.append(color[i-1])
    input_color

    groups = pd.DataFrame(zip(group_num, input_color), columns=['group', 'color'])
    groups['shape'] = 'disk'
    groups
    
    # 그룹정보
    group_vertices = node_df[['group', 'node']].sort_values(by=['group', 'node'], ascending=True)

    return node_attr, groups, group_vertices


def draw_ctrlity_scatter_plot(df, prd_name):
    '''
    산점도 작성 함수
    '''
    col_name1 = df.columns[0]
    col_name2 = df.columns[2]
    col_name3 = df.columns[3]

    # 한글 폰트 및 그래프 설정
    plt.rcParams.update({'font.family': 'Malgun Gothic',
                         'figure.dpi': '120',
                         'figure.figsize': [7, 4]})

    plt.scatter(df[col_name2], df[col_name3], color='r', s=10)
    plt.title('개벽_' + f"{prd_name}기" + '_논조')
    plt.xlabel('rTBC(전역중심성)')
    plt.ylabel('NC(지역중심성)')

    # 데이터를 한 번만 읽어서 사용합니다.
    names = df[col_name1].values
    xs = df[col_name2].values
    ys = df[col_name3].values

    # 텍스트 라벨 조정
    texts = [plt.text(x, y, name, fontsize=8) for name, x, y in zip(names, xs, ys)]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # 좌상에서 우하로 대각선 추가
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.plot([x_limits[0], x_limits[1]], [y_limits[1], y_limits[0]], 'g--')  # 'g--'는 녹색 점선을 의미

    plt.show()


def integrated_function(df, col_name, srh_word_n, rank, *words):
    '''
    # 본문 검색용 사용자 함수
    주어진 조건에 따라 DataFrame 내에서 특정 단어를 필터링하고, rank에 따라 상위 항목을 추출한다.
    '''
    # Step 1: find_ntimes_word
    matching_rows = df[col_name].apply(lambda x: sum([w in x for w in words]) >= int(srh_word_n))
    df_filtered = df[matching_rows]

    df_srs1 = df_filtered.groupby(['period', 'title', 'w_new', 'ho_no', 'year', 'month'])['sent_id'].count()
    df_srs1 = df_srs1.rename('sent_freq')
    prd_gisa = pd.DataFrame(df_srs1).reset_index().sort_values(['period', 'sent_freq'], ascending=[True, False])

    # Step 2: get_top_n_gisa_per_period
    top_n_gisa = prd_gisa.groupby('period').apply(lambda group: group.nlargest(int(rank), 'sent_freq')).reset_index(drop=True)

    # Step 3: look_into_sent_gisa
    df_filtered1 = df_filtered[['sent_id', 'sent_raw', 'sent_split', 'r_no', 'title', 'w_new', 'ho_no', 'period']]
    merged_df = pd.merge(top_n_gisa, df_filtered1, on=['period', 'title', 'w_new', 'ho_no'], how='left')

    return top_n_gisa, merged_df


# 결과값 중에서 특정 기사의 문장들을 자세히 살펴보기
def pick_up_sents_of_one_gisa(top_n_gisa, index_no, merged_df): # merged_df는 integrated_function()의 둘째 결과값
    srh_title_var = top_n_gisa.loc[index_no, ['title']][-1]
    ho_no_var = top_n_gisa.loc[index_no, ['ho_no']][-1]
    result_df = merged_df[(merged_df['title'] == srh_title_var)&(merged_df['ho_no'] == ho_no_var)]
    print(result_df.shape)
    return result_df

# 시기별 sent_freq의 합산 값 계산
def get_sum_by_period(top_n_gisa):
    sum_by_period = top_n_gisa.groupby('period')['sent_freq'].sum()

    # 결과 출력
    for period, sum_value in sum_by_period.items():
        print(f"시기: {period} - 빈도 합계: {sum_value}")