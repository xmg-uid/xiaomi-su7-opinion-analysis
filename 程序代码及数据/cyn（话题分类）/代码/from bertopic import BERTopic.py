from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP

# 示例：微博文本数据（预处理后）
weibo_texts = [
   
]

# 自定义微博话题的种子词
seed_words = []  # 根据实际需求调整

# 降维模型配置（固定随机状态确保结果可复现）
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# 使用种子词增强特定话题表示
ctfidf_model = ClassTfidfTransformer(
    seed_words=seed_words,
    seed_multiplier=2  # 增强种子词在话题表示中的权重
)

# 初始化并训练BERTopic模型
topic_model = BERTopic(
    umap_model=umap_model,
    min_topic_size=15,  # 最小话题大小，可根据数据量调整
    ctfidf_model=ctfidf_model,
    language="chinese",  # 如果是中文微博，指定语言为中文
    verbose=True  # 显示训练进度
)

# 执行聚类
topics, probs = topic_model.fit_transform(weibo_texts)

# 查看聚类结果
print(topic_model.get_topic_info())  # 打印话题摘要
print(topic_model.get_topic(0))     # 查看第一个话题的关键词