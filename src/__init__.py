# src/__init__.py
# 重新导出所有需要的函数
try:
    # 尝试相对导入
    from .base.data_processing import initialize_spark, load_data, preprocess_data, split_data
    from .base.collaborative_filtering import build_als_model, evaluate_als_model, tune_als_parameters
    from .base.content_based import build_tfidf_model, get_content_recommendations
    from .base.hybrid_model import build_hybrid_recommender
    from .base.evaluation import compare_recommenders, visualize_comparison, save_evaluation_results
    from .base.cold_start import build_popularity_model, build_content_based_cold_start
except ImportError:
    # 如果相对导入失败，尝试从src导入
    from src.base.data_processing import initialize_spark, load_data, preprocess_data, split_data
    from src.base.collaborative_filtering import build_als_model, evaluate_als_model, tune_als_parameters
    from src.base.content_based import build_tfidf_model, get_content_recommendations
    from src.base.hybrid_model import build_hybrid_recommender
    from src.base.evaluation import compare_recommenders, visualize_comparison, save_evaluation_results
    from src.base.cold_start import build_popularity_model, build_content_based_cold_start