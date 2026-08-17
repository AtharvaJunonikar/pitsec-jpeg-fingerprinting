import pandas as pd

FEATURE_COLS = [
    # SSIM (8)
    'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
    'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3',
    # DCT (15)
    'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
    'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
    'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb',
    # YCbCr (8)
    'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
    'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std',
    # Chroma (4)
    'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std',
    'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
]
df = pd.read_csv("output/all_features_combined.csv")
df_cleaned = df.drop_duplicates(subset=FEATURE_COLS, keep='first')

df_cleaned.to_csv("output/all_features_combined_ssim_fixed_cleaned.csv", index=False)
print(f"Cleaned dataset saved to output/all_features_combined_ssim_fixed_cleaned.csv, {len(df_cleaned)} rows remaining after removing duplicates.")