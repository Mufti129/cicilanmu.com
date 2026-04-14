from src.preprocessing import load_data, preprocess
from src.prediction import train_model

df = load_data('data/pawn_data.csv')
df_clean = preprocess(df)

train_model(df_clean)

print("Model berhasil dibuat!")
