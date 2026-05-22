import pandas as pd
df = pd.read_csv(r"D:\Home\esalasvilla\Documents\Stage\Breast_Project\Data\datasets\prediction\PET_CT_features_with_pcr.csv")
df.to_excel(r"D:\Home\esalasvilla\Documents\Stage\Breast_Project\Data\datasets\prediction\PET_CT_features_with_pcr.xlsx", index=False)
