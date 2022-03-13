from utils.metrics import save_conf_matrix
import os

save_conf_matrix(os.path.join('results/video/last_layer', "conf_matrix.csv"), os.path.join('results/video/last_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/video/all_layer', "conf_matrix.csv"), os.path.join('results/video/all_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/audio/last_layer', "conf_matrix.csv"), os.path.join('results/audio/last_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/audio/all_layer', "conf_matrix.csv"), os.path.join('results/audio/all_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/video/last_layer', "conf_matrix.csv"), os.path.join('results/video/last_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/video/last_layer', "conf_matrix.csv"), os.path.join('results/video/last_layer', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/feature_fusion', "conf_matrix.csv"), os.path.join('results/multimodal/feature_fusion', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/feature_fusion_norm', "conf_matrix.csv"), os.path.join('results/multimodal/feature_fusion_norm', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/pred_fusion_0.1', "conf_matrix.csv"), os.path.join('results/multimodal/pred_fusion_0.1', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/pred_fusion_0.3', "conf_matrix.csv"), os.path.join('results/multimodal/pred_fusion_0.3', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/pred_fusion_0.5', "conf_matrix.csv"), os.path.join('results/multimodal/pred_fusion_0.5', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/pred_fusion_0.7', "conf_matrix.csv"), os.path.join('results/multimodal/pred_fusion_0.7', "conf_matrix_img.png"))
save_conf_matrix(os.path.join('results/multimodal/pred_fusion_0.9', "conf_matrix.csv"), os.path.join('results/multimodal/pred_fusion_0.9', "conf_matrix_img.png"))
