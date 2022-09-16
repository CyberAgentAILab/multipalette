import os
from tensorflow.keras import regularizers

representation = 'lab_bins_16'
bin_range = 16
kmeansType = '_sklearn'

PROJECT_PATH = '../data/training_data/data_bert'

Config = {
    'project_path': PROJECT_PATH,
    'bin_range': bin_range,
    'representation': representation,
    'corpus_file_path': os.path.join(PROJECT_PATH, f'data_color/color_corpus_{representation}_train{kmeansType}.txt'),
    'vocabulary_file_path': os.path.join(PROJECT_PATH, f'data_color/color_vocab_{representation}_train{kmeansType}.txt'),
    'log_dir': os.path.join(PROJECT_PATH, 'logs'),
    'saved_weight': os.path.join(PROJECT_PATH, 'saved_weight_256d_16bins'),
    'character_frequency_threshold': 1,  # 3 may be better for large dataset
    'segment_size': 3,
    'batch_size': 64, # 2048 for training on GPU
    'max_palette_length': [5, 5, 5],
    'max_sequence_length': 18,
    'mask_rate': 0.1,
    'mask_token_rate': 0.3,
    'mask_num': 1,
    'mask_position': [], # fix mask position: 8, 9: cta colors
    'vocab_size': 800,  # fix vocab_size? len(color_freq)+4 (SEP, MASK, PAD, UNK)
    'embedding_size': 256, # for NLP 256 is better?
    'num_transformer_layers': 3, # 3
    'num_attention_heads': 8, # 8
    'intermediate_size': 1024,
    'initializer_variance': 0.02,
    'bias_regularizer': 1e-5,
}
