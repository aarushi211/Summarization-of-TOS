import pandas as pd

cr = pd.read_csv('context_recall_results.csv')
ar = pd.read_csv('answer_relevance_results.csv')
rt = pd.read_csv('red_team_results.csv')

print('=== CONTEXT RECALL ===')
print('Overall Recall Score:', cr['score'].mean())
print(cr['verdict'].value_counts())

print('\n=== ANSWER RELEVANCE ===')
print('Overall Relevance Score:', ar['score'].mean())
abstentions = ar['is_abstention'].sum()
print(f'Abstentions: {abstentions} / {len(ar)}')

print('\n=== RED TEAM SAFETY ===')
safe_count = rt['is_safe'].sum()
print(f'Overall Safety Rate: {safe_count} / {len(rt)} ({safe_count/len(rt)*100:.0f}%)')
print('\nBreakdown by Category:')
for cat, grp in rt.groupby('category'):
    print(f'  {cat}: {grp["is_safe"].sum()} / {len(grp)}')
    
print('\n=== FAILURES (if any) ===')
fails = rt[~rt['is_safe']]
for _, r in fails.iterrows():
    print(f'- [{r["category"]}] Q: {r["question"]} | Verdict: {r["verdict"]}')
