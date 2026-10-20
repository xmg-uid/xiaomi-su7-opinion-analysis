from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# 示例：微博文本数据（预处理后）
weibo_texts = [
   
]

# 自定义微博话题的种子词
seed_words = ["冬奥会", "春晚", "疫情", "防护", "口罩"]

# 嵌入模型配置
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# 使用种子词增强特定话题表示
ctfidf_model = ClassTfidfTransformer(
    seed_words=seed_words,
    seed_multiplier=2
)

# 构建不同聚类模型
cluster_models = {
    "HDBSCAN": {
        "params": {
            "min_cluster_size": 5,
            "metric": "euclidean",
            "cluster_selection_method": "eom"
        },
        "module": "hdbscan",
        "class": "HDBSCAN"
    },
    "DBSCAN": {
        "params": {
            "eps": 0.5,
            "min_samples": 3,
            "metric": "cosine"
        },
        "module": "sklearn.cluster",
        "class": "DBSCAN"
    }
}

# 评估不同聚类模型
results = {}

for name, config in cluster_models.items():
    try:
        # 动态导入模型类
        module = __import__(config["module"], fromlist=[config["class"]])
        ModelClass = getattr(module, config["class"])
        
        # 初始化聚类模型
        cluster_model = ModelClass(**config["params"])
        
        # 初始化BERTopic模型
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            ctfidf_model=ctfidf_model,
            language="chinese",
            verbose=True
        )
        
        # 执行聚类
        topics, probs = topic_model.fit_transform(weibo_texts)
        
        # 计算聚类评估指标
        if hasattr(cluster_model, 'labels_'):
            labels = cluster_model.labels_
        else:
            labels = topics
            
        # 计算轮廓系数（Silhouette Score）
        try:
            # 确保有足够的聚类数量
            if len(set(labels)) > 1:
                # 标准化嵌入向量用于评估
                embeddings = topic_model.umap_model.embedding_
                scaled_embeddings = StandardScaler().fit_transform(embeddings)
                score = silhouette_score(scaled_embeddings, labels)
            else:
                score = -1  # 只有一个聚类时轮廓系数为-1
        except Exception as e:
            score = float('nan')
            print(f"计算{name}轮廓系数时出错: {e}")
        
        # 保存结果
        results[name] = {
            "model": topic_model,
            "topics": topics,
            "probs": probs,
            "labels": labels,
            "score": score,
            "n_topics": len(set(labels)) - (1 if -1 in labels else 0)  # 排除噪声点
        }
        
        print(f"\n{name} 聚类结果:")
        print(f"话题数量: {results[name]['n_topics']}")
        print(f"轮廓系数: {results[name]['score']:.4f}")
        
        # 打印前3个话题的关键词
        for i in range(min(3, results[name]['n_topics'])):
            print(f"话题 {i} 关键词: {topic_model.get_topic(i)[:5]}")
            
    except Exception as e:
        print(f"运行{name}模型时出错: {e}")

# 可视化不同模型的聚类效果对比
if results:
    model_names = list(results.keys())
    scores = [results[name]['score'] for name in model_names]
    n_topics = [results[name]['n_topics'] for name in model_names]
    
    plt.figure(figsize=(12, 5))
    
    # 轮廓系数对比
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, scores, color=['#4CAF50', '#2196F3'])
    plt.title('聚类质量对比 (轮廓系数)')
    plt.ylabel('分数')
    plt.ylim(-1, 1)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 话题数量对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, n_topics, color=['#FF9800', '#9C27B0'])
    plt.title('话题数量对比')
    plt.ylabel('话题数')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()