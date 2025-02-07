import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def calculate_metrics(store_count, sku_count, profit_rate=0.087):
    """
    计算经营指标
    """
    # 基准数据（7店数据）
    base_stores = 7
    base_sku = 383
    base_data = {
        '30天总销量': 71024,
        '30天采购价出库额': 161393
    }
    
    # 计算系数
    store_ratio = store_count / base_stores
    sku_ratio = sku_count / base_sku
    
    # 计算销售相关指标
    采购价出库额 = base_data['30天采购价出库额'] * store_ratio * sku_ratio
    利润 = 采购价出库额 * profit_rate
    
    # 计算各项成本
    人员成本 = (sku_count / 10) * store_count * 8 / 120 / 8 / 6 * 5500
    运输成本 = store_count * 350 / 12 * 2 * 4
    周转箱成本 = store_count * 10 * 2.1
    其他费用 = store_count / 10 * 300
    仓租成本 = max(sku_count - 500, 0) * 15
    合计物流成本 = 人员成本 + 运输成本 + 周转箱成本 + 其他费用 + 仓租成本
    净利额 = 利润 - 合计物流成本
    
    return {
        '采购价出库额': 采购价出库额,
        '人员成本': 人员成本,
        '运输成本': 运输成本,
        '周转箱成本': 周转箱成本,
        '其他费用': 其他费用,
        '仓租成本': 仓租成本,
        '净利额': 净利额
    }

def create_stacked_chart(x_values, fixed_value, vary_by_store=True):
    results = []
    for x in x_values:
        store_count = x if vary_by_store else fixed_value
        sku_count = fixed_value if vary_by_store else x
        metrics = calculate_metrics(store_count, sku_count)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # 创建堆叠图
    fig = go.Figure()
    
    # 添加各成本和净利润堆叠
    categories = ['人员成本', '运输成本', '周转箱成本', '其他费用', '仓租成本', '净利额']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 计算总利润（毛利润）
    df['毛利润'] = df['净利额'] + df['人员成本'] + df['运输成本'] + df['周转箱成本'] + df['其他费用'] + df['仓租成本']
    
    bottom = np.zeros(len(x_values))
    for cat, color in zip(categories, colors):
        # 计算合计物流成本
        物流成本 = df['人员成本'] + df['运输成本'] + df['周转箱成本'] + df['其他费用'] + df['仓租成本']
        
        # 准备悬停文本
        hover_text = []
        for idx, row in df.iterrows():
            cost = (row['人员成本'] + row['运输成本'] + row['周转箱成本'] + 
                   row['其他费用'] + row['仓租成本'])
            text = "<br>".join([
                f"{c}: {int(v):,} ({(v/row['采购价出库额']*100):.1f}%)" for c, v in [
                    ('人员成本', row['人员成本']),
                    ('运输成本', row['运输成本']),
                    ('周转箱成本', row['周转箱成本']),
                    ('其他费用', row['其他费用']),
                    ('仓租成本', row['仓租成本']),
                    ('合计物流成本', cost),
                    ('净利润', row['净利额']),
                    ('毛利润', row['毛利润']),
                    ('采购价出库额', row['采购价出库额'])
                ]
            ])
            hover_text.append(text)
        
        fig.add_trace(go.Bar(
            name=cat,
            x=x_values,
            y=df[cat],
            marker_color=color,
            base=bottom,
            text=None,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ))
        bottom += df[cat]
    
    # 在柱状图顶部添加毛利润和物流成本标注
    fig.add_trace(go.Bar(
        name='空白',
        x=x_values,
        y=[0] * len(x_values),
        base=bottom,
        text=[f'毛利润:{int(m)}<br>物流成本:{int(c)}' for m, c in zip(
            df['毛利润'], 
            df['人员成本'] + df['运输成本'] + df['周转箱成本'] + df['其他费用'] + df['仓租成本']
        )],
        textposition='outside',
        marker_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='none'
    ))
    
    # 计算最大可能值
    max_store = 50
    max_sku = 1000
    max_metrics = calculate_metrics(max_store, max_sku)
    max_total = (max_metrics['人员成本'] + max_metrics['运输成本'] + 
                 max_metrics['周转箱成本'] + max_metrics['其他费用'] + 
                 max_metrics['仓租成本'] + max_metrics['净利额'])
    
    # 更新布局
    title = f'{"门店数" if vary_by_store else "SKU数"}变化对成本构成的影响 ({"SKU" if vary_by_store else "门店数"}={fixed_value})'
    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis_title="门店数" if vary_by_store else "SKU数",
        yaxis_title="金额",
        showlegend=True,
        margin=dict(t=100, b=150, l=50, r=20),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        yaxis=dict(
            range=[0, max_total * 1.1],
            fixedrange=True
        ),
        xaxis=dict(
            fixedrange=True
        ),
        modebar_remove=[
            'zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d', 
            'zoomOut2d', 'autoScale2d', 'resetScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines'
        ],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        dragmode=False
    )
    
    return fig

# Streamlit 应用
st.set_page_config(
    page_title="成本分析",
    layout="wide",  # 使用宽屏布局
    initial_sidebar_state="collapsed"  # 默认收起侧边栏
)

st.title('成本分析')

# 添加缓存装饰器，避免重复计算
@st.cache_data
def get_chart(x_values, fixed_value, vary_by_store=True):
    return create_stacked_chart(x_values, fixed_value, vary_by_store)

# 使用tabs而不是columns，在手机上显示效果更好
tab1, tab2 = st.tabs(["按门店数变化", "按SKU数变化"])

with tab1:
    st.subheader('按门店数变化图')
    sku_value = st.slider('选择SKU数:', 100, 1000, 400, 100, key='sku_slider')
    store_range = np.arange(10, 51, 5)
    chart1 = get_chart(store_range, sku_value, vary_by_store=True)
    # 设置图表大小
    chart1.update_layout(
        height=500,  # 减小图表高度
        width=None,  # 自适应宽度
    )
    st.plotly_chart(chart1, use_container_width=True)

with tab2:
    st.subheader('按SKU数变化图')
    store_value = st.slider('选择门店数:', 10, 50, 20, 5, key='store_slider')
    sku_range = np.arange(100, 1001, 100)
    chart2 = get_chart(sku_range, store_value, vary_by_store=False)
    # 设置图表大小
    chart2.update_layout(
        height=500,  # 减小图表高度
        width=None,  # 自适应宽度
    )
    st.plotly_chart(chart2, use_container_width=True) 