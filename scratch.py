import pandas as pd

results_df = pd.read_csv('faithfulness_results.csv')

print('\n=== EXAMPLES OF CONTRADICTED CLAIMS ===')
contra = results_df[results_df['verdict'] == 'contradicted'].head(3)
for _, row in contra.iterrows():
    print(f'Claim: {row["claim"]}')
    print(f'Reason: {row["reason"]}\n')
    
print('\n=== EXAMPLES OF NEUTRAL CLAIMS ===')
neutral = results_df[results_df['verdict'] == 'neutral'].head(3)
for _, row in neutral.iterrows():
    print(f'Claim: {row["claim"]}')
    print(f'Reason: {row["reason"]}\n')
