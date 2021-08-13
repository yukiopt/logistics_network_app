import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import pydeck as pdk
import plotly.graph_objects as go
from geopy.distance import geodesic
from pulp import *

# 倉庫データ・店舗データを読み込む関数
@st.cache(allow_output_mutation=True)
def load_data():
    warehouse = pd.read_excel('倉庫.xlsx', dtype={'倉庫コード':str})
    warehouse['倉庫コード'] = warehouse['倉庫コード'].apply(lambda x:x.zfill(6))
    warehouse['容量'] = warehouse['容量'] * 8
    warehouse = warehouse.sort_values('倉庫コード')
    store = pd.read_excel('店舗.xlsx', dtype={'店舗コード':str})
    store['店舗コード'] = store['店舗コード'].apply(lambda x:x.zfill(6))
    return warehouse, store

# 距離データを作る関数
@st.cache(persist=True)
def get_distance(origin, destination):
    distance = {}
    for i, row_o in origin.iterrows():
        for j, row_d in destination.iterrows():
            origin_coordinates = (row_o['緯度'], row_o['経度'])
            destination_coordinates = (row_d['緯度'], row_d['経度'])
            distance[row_o['倉庫コード'], row_d['店舗コード']] = \
                geodesic(origin_coordinates, destination_coordinates).km
    return distance

# 倉庫使用率を水平棒グラフにする関数
def plot_usage(warehouse):
    fig = go.Figure(go.Bar(
            x=warehouse['物量']/warehouse['容量'] * 100,
            y=warehouse['倉庫名'],
            orientation='h'))
    fig['layout']['xaxis']['range'] = [0, 100]
    fig['layout']['yaxis']['autorange'] = 'reversed'
    st.write(fig)

# 配送エリアマップを作る関数
def plot_solution(warehouse, store):
    solution_map = folium.Map(location=[35.69, 139.69], tiles='cartodbpositron', zoom_start=5)
    feature_groups = {}
    for w in warehouse.index:
        feature_groups[w] = folium.FeatureGroup(name=warehouse.at[w, '倉庫名'])
    for i, row in store.iterrows():
        folium.Circle(
            location=[row['緯度'], row['経度']],
            popup=row['店舗名'],
            radius=2000,
            fill=True,
            color=warehouse.at[row['供給元倉庫'], 'カラー'],
            fill_color=warehouse.at[row['供給元倉庫'], 'カラー']
        ).add_to(feature_groups[row['供給元倉庫']])
    for feature_group in feature_groups:
        solution_map.add_child(feature_groups[feature_group])
    solution_map.add_child(folium.map.LayerControl())
    st.components.v1.html(folium.Figure().add_child(solution_map).render(), height=500)


# 輸送コストを最小化する関数
@st.cache(persist=True)
def minimize_cost(warehouse, store, distance):
    # 前処理
    warehouse = warehouse.set_index('倉庫コード')
    store = store.set_index('店舗コード')
    warehouse_list = warehouse.index
    store_list = store.index

    # モデリング
    model = LpProblem('輸送コスト最小化', LpMinimize)

    x = {}
    for w in warehouse_list:
        for s in store_list:
            x[w, s] = LpVariable(f'x({w}, {s})', 0, 1, LpInteger)

    q = {}
    for w in warehouse_list:
        q[w] = LpVariable(f'q({w})', 0, None, LpContinuous)

    for s in store_list:
        model += lpSum(x[w, s] for w in warehouse_list) == 1

    for w in warehouse_list:
        model += q[w] == lpSum(store.at[s, '需要量'] * x[w, s] for s in store_list)
        model += lpSum(q[w]) <= warehouse.at[w, '容量']

    model += lpSum(store.at[s, '需要量'] * distance[w, s] * x[w, s] for (w, s) in x)

    status = model.solve(PULP_CBC_CMD(timeLimit=10))

    # 追加情報
    warehouse['物量'] = 0
    for w in warehouse_list:
        warehouse.at[w, '物量'] = value(q[w])

    store['供給元倉庫'] = None
    for (w, s) in x:
        if value(x[w, s]) >= 0.5:
            store.at[s, '供給元倉庫'] = w 
        
    return status, warehouse, store, value(model.objective)


# 輸送コストを最小化する関数
@st.cache(persist=True)
def leveling(warehouse, store, distance):
    # 前処理
    warehouse = warehouse.set_index('倉庫コード')
    store = store.set_index('店舗コード')
    warehouse_list = warehouse.index
    store_list = store.index

    def problem1():

        # モデリング
        model1 = LpProblem('最大使用率最小化', LpMinimize)

        x = {}
        for w in warehouse_list:
            for s in store_list:
                x[w, s] = LpVariable(f'x({w}, {s})', 0, 1, LpInteger)

        z = LpVariable('z', 0, 1, LpContinuous) # 使用率の最大値

        for s in store_list:
            model1 += lpSum(x[w, s] for w in warehouse_list) == 1

        for w in warehouse_list:
            model1 += lpSum(store.at[s, '需要量'] * x[w, s] for s in store_list) <= warehouse.at[w, '容量'] * z

        model1 += z

        status1 = model1.solve(PULP_CBC_CMD(timeLimit=10))

        return value(model1.objective)

    def problem2(max_z):

        # モデリング
        model2 = LpProblem('使用率平準化', LpMinimize)

        x = {}
        for w in warehouse_list:
            for s in store_list:
                x[w, s] = LpVariable(f'x({w}, {s})', 0, 1, LpInteger)

        z = LpVariable('z', 0, 1, LpContinuous) # 使用率の最小値

        q = {}
        for w in warehouse_list:
            q[w] = LpVariable(f'q({w})', 0, None, LpContinuous)

        for s in store_list:
            model2 += lpSum(x[w, s] for w in warehouse_list) == 1

        for w in warehouse_list:
            model2 += q[w] == lpSum(store.at[s, '需要量'] * x[w, s] for s in store_list)
            model2 += q[w] <= warehouse.at[w, '容量'] * (max_z * 1.05)

        model2 += lpSum(store.at[s, '需要量'] * distance[w, s] * x[w, s] for (w, s) in x)

        status2 = model2.solve(PULP_CBC_CMD(timeLimit=10))

        # 追加情報
        warehouse['物量'] = 0
        for w in warehouse_list:
            warehouse.at[w, '物量'] = value(q[w])

        store['供給元倉庫'] = None
        for (w, s) in x:
            if value(x[w, s]) >= 0.5:
                store.at[s, '供給元倉庫'] = w 

        return status2, value(model2.objective)

    max_usage = problem1()
    status, obj = problem2(max_usage)

    return status, warehouse, store, obj


st.title('物流ネットワーク最適化アプリ')
df_warehouse, df_store = load_data()
st.sidebar.markdown('1.使用する機能を選んでください')
func_option = st.sidebar.selectbox('機能', ('可視化', '最適化'))

if func_option == '可視化':
    st.sidebar.markdown('2.倉庫と店舗のどちらを可視化するか選んでください')
    viz_option = st.sidebar.selectbox('選択', ('倉庫', '店舗'))
    """
    ドラッグストアなどの小売チェーンの物流網の構築を考えてみます。  

    1.**全国の政令指定都市を倉庫**と考え、その人口を容量としています。  
    2.**全国の市区町村(政令指定都市を除く)を店舗**と考え、その人口を需要量(=必要量)としています。  
    ※諸事情により、以下4つの政令指定都市を倉庫から外しています。申し訳ございません。   
    ・神奈川県相模原市    
    ・神奈川県横浜市  
    ・大阪府堺市  
    ・福岡県北九州市  

    """
    st.sidebar.markdown('「実行」にチェックしてください')
    do = st.sidebar.checkbox('実行', value=False)
    if do:
        # 倉庫の可視化
        if viz_option == '倉庫':
            # データフレーム
            st.markdown('## 倉庫一覧')
            st.dataframe(df_warehouse.loc[:, ['倉庫コード', '倉庫名', '容量', '緯度', '経度']])
            # 容量グラフ
            st.markdown('## 倉庫の容量グラフ')
            fig = go.Figure(go.Bar(
                x=df_warehouse['容量'],
                y=df_warehouse['倉庫名'],
                orientation='h'))
            fig['layout']['yaxis']['autorange'] = 'reversed'
            st.write(fig)
            # マップ
            st.markdown('## 倉庫マップ')
            warehouse_map = folium.Map(location=[35.69, 139.69], tiles='openstreetmap', zoom_start=5)
            for i, row in df_warehouse.iterrows():
                folium.Marker(
                    location=[row['緯度'], row['経度']],
                    popup=row['倉庫名'],
                    icon=folium.Icon(icon='home')
                ).add_to(warehouse_map)
            st.components.v1.html(folium.Figure().add_child(warehouse_map).render(), height=500)

        elif viz_option == '店舗':
            # データフレーム
            st.markdown('## 店舗一覧')
            st.dataframe(df_store)
            # 需要量の分布
            st.markdown('## 需要量の分布')
            fig = go.Figure(go.Histogram(x=df_store['需要量'].tolist()))
            st.write(fig)
            # マップ
            st.markdown('## 店舗マップ')
            view = pdk.ViewState(latitude=35.69, longitude=139.69, pitch=50, zoom=4)
            hexagon_layer = pdk.Layer(
                'HexagonLayer',
                data=df_store,
                get_position=['経度', '緯度'],
                elevation_scale=300,
                radius=3000,
                extruded=True,

            )
            layer_map = pdk.Deck(layers=hexagon_layer, initial_view_state=view)
            st.pydeck_chart(layer_map)

elif func_option == '最適化' :
    # 距離データ
    distance = get_distance(df_warehouse, df_store)
    st.sidebar.markdown('2.最適化の目的を選んでください')
    solve_option = st.sidebar.selectbox('目的', ('輸送コスト最小化', '倉庫の使用率平準化'))
    st.sidebar.markdown('3.「実行」にチェックしてください')
    do = st.sidebar.checkbox('実行', value=False)
    """
    倉庫の容量の制約を満たしつつ、店舗への輸送コストが最小になるような倉庫と店舗の組合せを求めます。 
    各店舗は、1つの倉庫から輸送を受けるとします。 
    ### **オプション１．輸送コスト最小化**  
    倉庫$i$から店舗$j$へ輸送するときのコストを（店舗$j$の需要量×$(i, j)$間の距離）として、総輸送コストを最小にすることを考えて最適化します。  
    ### **オプション２．倉庫の使用率平準化**  

    """
    if do:
        if solve_option == '輸送コスト最小化':
            
            st.text('輸送コストが最小になるようにします！')

            # 最適化実行
            status, warehouse, store, totalcost = minimize_cost(df_warehouse, df_store, distance)

            # 総費用・使用率可視化
            st.markdown('## 倉庫の使用率')
            plot_usage(warehouse)
            # マップ
            st.markdown('## 最適解マップ')
            plot_solution(warehouse, store)

        else:
            st.text('倉庫の使用率を平準化します！')

            # 最適化実行
            status, warehouse, store, totalcost = leveling(df_warehouse, df_store, distance)
            # 総費用・使用率可視化
            st.markdown('## 倉庫の使用率')
            plot_usage(warehouse)
            # マップ
            st.markdown('## 最適解マップ')
            plot_solution(warehouse, store)


# 出典
st.markdown('#### 出典')
st.markdown('1.全国地方公共団体コード(https://www.soumu.go.jp/denshijiti/code.html)')
st.markdown('2.e-Stat(https://www.e-stat.go.jp/)')
st.markdown('本結果は、1・2を加工して作成しました。')