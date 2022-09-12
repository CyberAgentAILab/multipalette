import os
from tensorflow.keras import regularizers

representation = 'lab_bins_16'
bin_range = 16
kmeansType = '_sklearn'

PROJECT_PATH = 'data_bert/'

Config = {
    'Project_path': PROJECT_PATH,
    'Bin_range': bin_range,
    'Representation': representation
    'Corpus_File_Path': os.path.join(PROJECT_PATH, f'Data_color/color_corpus_{representation}_train{kmeansType}.txt'),
    'Vocabulary_File_Path': os.path.join(PROJECT_PATH, f'Data_color/color_vocab_{representation}_train{kmeansType}.txt'),
    # 'Labels_File_Path': os.path.join(PROJECT_PATH, f'Data_color/color_labels.txt'),
    'Log_Dir': os.path.join(PROJECT_PATH, 'Logs'),
    'Saved_Weight': os.path.join(PROJECT_PATH, 'Saved_Weight_256d_16bins'),
    'Character_Frequency_Threshold': 1,  # 3 may be better for large dataset
    'Segment_Size': 3,
    'Batch_Size': 64, # 2048 for training on GPU
    'Max_Palette_Length': [5, 5, 5],
    'Max_Sequence_Length': 18,
    'Mask_Rate': 0.1,
    'Mask_Token_Rate': 0.3,
    'Mask_num': 1,
    'Mask_position': [], # fix mask position: 8, 9: cta colors
    'Vocab_Size': 800,  # fix vocab_size? len(color_freq)+4 (SEP, MASK, PAD, UNK)
    'Embedding_Size': 256, # for NLP 256 is better?
    'Num_Transformer_Layers': 3, # 3
    'Num_Attention_Heads': 8, # 8
    'Intermediate_Size': 1024,
    'Initializer_Variance': 0.02,
    'bias_regularizer': 1e-5,
}
